# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Unit tests for table encryption.

These mirror the Java and Rust encryption tests. The encoded key-metadata and
AGS1 byte layouts are the interop contract across engines.
"""

from __future__ import annotations

import os

import pytest
from cryptography.exceptions import InvalidTag

from pyiceberg.encryption import ciphers
from pyiceberg.encryption.io import EncryptedInputFile, EncryptedOutputFile
from pyiceberg.encryption.key_metadata import StandardKeyMetadata
from pyiceberg.encryption.kms import InMemoryKmsClient
from pyiceberg.encryption.manager import (
    KEK_TIMESTAMP_PROPERTY,
    PlaintextEncryptionManager,
    StandardEncryptionManager,
    create_encryption_manager,
)
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.table.metadata import EncryptedKey

KEY_16 = b"0123456789abcdef"
AAD = b"aadprefix12345678"


# --- StandardKeyMetadata -----------------------------------------------------


def test_key_metadata_roundtrip_full() -> None:
    km = StandardKeyMetadata(KEY_16, AAD, 100_000)
    decoded = StandardKeyMetadata.decode(km.encode())
    assert decoded == km
    assert decoded.encryption_key == KEY_16
    assert decoded.aad_prefix == AAD
    assert decoded.file_length == 100_000


def test_key_metadata_roundtrip_minimal() -> None:
    km = StandardKeyMetadata(KEY_16)
    decoded = StandardKeyMetadata.decode(km.encode())
    assert decoded.aad_prefix is None
    assert decoded.file_length is None


def test_key_metadata_version_byte_and_layout() -> None:
    # Interop contract: [0x01][avro datum]. This exact prefix must match Java/Rust.
    encoded = StandardKeyMetadata(KEY_16, AAD).encode()
    assert encoded[0] == 1
    # required bytes(16) -> zigzag varint 0x20 then 16 bytes; aad union index 1 -> 0x02
    assert encoded[1] == 0x20


@pytest.mark.parametrize("length", [16, 24, 32])
def test_key_metadata_accepts_valid_key_lengths(length: int) -> None:
    StandardKeyMetadata(bytes(length))


@pytest.mark.parametrize("length", [0, 4, 15, 20, 33])
def test_key_metadata_rejects_invalid_key_lengths(length: int) -> None:
    with pytest.raises(ValueError, match="Invalid encryption key length"):
        StandardKeyMetadata(bytes(length))


def test_key_metadata_rejects_unsupported_version() -> None:
    with pytest.raises(ValueError, match="version"):
        StandardKeyMetadata.decode(bytes([0x02]))


def test_key_metadata_rejects_empty_buffer() -> None:
    with pytest.raises(ValueError, match="Empty"):
        StandardKeyMetadata.decode(b"")


def test_key_metadata_repr_redacts_key() -> None:
    text = repr(StandardKeyMetadata(KEY_16, AAD))
    assert "REDACTED" in text
    assert KEY_16.hex() not in text
    assert "0123456789abcdef" not in text


# --- ciphers -----------------------------------------------------------------


def test_aes_gcm_cipher_roundtrip() -> None:
    cipher = ciphers.AesGcmCipher(KEY_16)
    ct = cipher.encrypt(b"secret payload", b"aad")
    assert cipher.decrypt(ct, b"aad") == b"secret payload"


def test_aes_gcm_cipher_wrong_aad_fails() -> None:
    cipher = ciphers.AesGcmCipher(KEY_16)
    ct = cipher.encrypt(b"secret", b"aad")
    with pytest.raises(InvalidTag):
        cipher.decrypt(ct, b"other-aad")


@pytest.mark.parametrize("size", [0, 1, 100, 5000])
def test_ags1_stream_roundtrip(size: int) -> None:
    plaintext = os.urandom(size)
    stream = ciphers.encrypt_stream(plaintext, KEY_16, AAD)
    assert stream[:4] == ciphers.GCM_STREAM_MAGIC
    assert ciphers.calculate_plaintext_length(len(stream)) == size
    assert ciphers.decrypt_stream(stream, KEY_16, AAD) == plaintext


def test_ags1_multi_block_roundtrip() -> None:
    # Two-plus blocks exercises the per-block index AAD.
    plaintext = os.urandom(ciphers.PLAIN_BLOCK_SIZE * 2 + 17)
    stream = ciphers.encrypt_stream(plaintext, KEY_16, AAD)
    assert ciphers.calculate_plaintext_length(len(stream)) == len(plaintext)
    assert ciphers.decrypt_stream(stream, KEY_16, AAD) == plaintext


def test_ags1_tamper_detected() -> None:
    stream = bytearray(ciphers.encrypt_stream(b"hello world", KEY_16, AAD))
    stream[-1] ^= 0xFF
    with pytest.raises(InvalidTag):
        ciphers.decrypt_stream(bytes(stream), KEY_16, AAD)


def test_ags1_bad_magic() -> None:
    with pytest.raises(ValueError, match="magic"):
        ciphers.decrypt_stream(b"XXXX" + bytes(20), KEY_16, AAD)


# --- KMS ---------------------------------------------------------------------


def test_in_memory_kms_wrap_unwrap() -> None:
    kms = InMemoryKmsClient()
    kms.add_master_key("master-1")
    wrapped = kms.wrap_key(KEY_16, "master-1")
    assert wrapped != KEY_16
    assert kms.unwrap_key(wrapped, "master-1") == KEY_16


def test_in_memory_kms_unknown_key() -> None:
    with pytest.raises(ValueError, match="Unknown master key"):
        InMemoryKmsClient().wrap_key(KEY_16, "missing")


# --- StandardEncryptionManager ----------------------------------------------


def _manager() -> StandardEncryptionManager:
    kms = InMemoryKmsClient()
    kms.add_master_key("master-1")
    return StandardEncryptionManager("master-1", kms)


def test_manager_manifest_list_key_roundtrip() -> None:
    mgr = _manager()
    km = StandardKeyMetadata(KEY_16, AAD)
    key_id = mgr.add_manifest_list_key_metadata(km)
    # A KEK and the wrapped entry are both stored for persistence at commit.
    assert len(mgr.encryption_keys) == 2
    assert mgr.load_manifest_list_key_metadata(key_id) == km


def test_manager_reuses_active_kek() -> None:
    mgr = _manager()
    mgr.add_manifest_list_key_metadata(StandardKeyMetadata(KEY_16, AAD))
    id2 = mgr.add_manifest_list_key_metadata(StandardKeyMetadata(KEY_16, AAD))
    # KEK reused: only one more entry (3 total), and the second entry points at the same KEK.
    keys = mgr.encryption_keys
    assert len(keys) == 3
    kek_ids = {k.key_id for k in keys.values() if k.encrypted_by_id == "master-1"}
    assert len(kek_ids) == 1
    assert keys[id2].encrypted_by_id in kek_ids


def test_manager_kek_has_timestamp() -> None:
    mgr = _manager()
    mgr.add_manifest_list_key_metadata(StandardKeyMetadata(KEY_16, AAD))
    kek = next(k for k in mgr.encryption_keys.values() if k.encrypted_by_id == "master-1")
    assert KEK_TIMESTAMP_PROPERTY in kek.properties


def test_manager_encrypt_produces_encrypted_output(tmp_path: str) -> None:
    mgr = _manager()
    fileio = PyArrowFileIO()
    out = mgr.encrypt(fileio.new_output(f"{tmp_path}/data.bin"))
    assert isinstance(out, EncryptedOutputFile)
    assert len(out.key_metadata.encryption_key) == 16


# --- create_encryption_manager ----------------------------------------------


def test_create_manager_plaintext_for_v2(example_table_metadata_v2: dict) -> None:
    from pyiceberg.table.metadata import TableMetadataUtil

    metadata = TableMetadataUtil.parse_obj(example_table_metadata_v2)
    assert isinstance(create_encryption_manager(metadata, None), PlaintextEncryptionManager)


# --- EncryptedKey model ------------------------------------------------------


def test_encrypted_key_base64_json_roundtrip() -> None:
    key = EncryptedKey(
        key_id="k1",
        encrypted_key_metadata=b"\x00\x01\x02\x03",
        encrypted_by_id="master-1",
        properties={KEK_TIMESTAMP_PROPERTY: "123"},
    )
    as_json = key.model_dump_json(by_alias=True)
    assert '"encrypted-key-metadata":"AAECAw=="' in as_json
    restored = EncryptedKey.model_validate_json(as_json)
    assert restored.encrypted_key_metadata == b"\x00\x01\x02\x03"
    assert restored.key_id == "k1"
    assert restored.encrypted_by_id == "master-1"


# --- io wrappers: end-to-end AGS1 through real FileIO ------------------------


def test_encrypted_file_roundtrip_through_fileio(tmp_path: str) -> None:
    fileio = PyArrowFileIO()
    location = f"{tmp_path}/manifest.avro"
    km = StandardKeyMetadata(KEY_16, AAD)
    plaintext = b"pretend this is an avro manifest list" * 100

    enc_out = EncryptedOutputFile(fileio.new_output(location), km)
    with enc_out.create(overwrite=True) as stream:
        stream.write(plaintext)

    # On disk the bytes are AGS1 and larger than the plaintext (header + nonce + tag).
    raw = fileio.new_input(location)
    assert len(raw) > len(plaintext)
    with raw.open() as f:
        assert f.read(4) == ciphers.GCM_STREAM_MAGIC

    enc_in = EncryptedInputFile(fileio.new_input(location), km)
    assert len(enc_in) == len(plaintext)
    with enc_in.open() as f:
        assert f.read() == plaintext


# --- Parquet data-file decryption via the pyarrow read path ------------------


def test_parquet_decryption_scan_options_roundtrip(tmp_path: str) -> None:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    from pyarrow.parquet.encryption import create_encryption_properties

    from pyiceberg.io.pyarrow import _build_decryption_scan_options

    table = pa.table({"a": list(range(1000)), "b": ["x" * 8] * 1000})
    location = f"{tmp_path}/data.parquet"
    enc_props = create_encryption_properties(footer_key=KEY_16, aad_prefix=AAD)
    with pq.ParquetWriter(location, table.schema, encryption_properties=enc_props) as writer:
        writer.write_table(table)

    # A plain read must fail — the file really is encrypted.
    with pytest.raises(OSError):
        pq.read_table(location)

    # The Iceberg key metadata drives PyArrow decryption on the dataset path.
    key_metadata = StandardKeyMetadata(KEY_16, AAD).encode()
    scan_options = _build_decryption_scan_options(key_metadata)
    assert scan_options is not None
    arrow_format = ds.ParquetFileFormat(default_fragment_scan_options=scan_options)
    fragment = arrow_format.make_fragment(pa.OSFile(location))
    result = ds.Scanner.from_fragment(fragment=fragment, schema=fragment.physical_schema).to_table()
    assert result.equals(table)


def test_build_decryption_scan_options_none_when_unencrypted() -> None:
    from pyiceberg.io.pyarrow import _build_decryption_scan_options

    assert _build_decryption_scan_options(None) is None
