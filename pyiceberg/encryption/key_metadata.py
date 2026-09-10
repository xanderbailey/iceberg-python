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
"""Avro-serialized key metadata, byte-compatible with Java and Rust.

The wire format is ``[version byte 0x01][Avro binary datum]``, where the datum
follows the Iceberg schema::

    required(0, "encryption_key", binary)
    optional(1, "aad_prefix",     binary)
    optional(2, "file_length",    long)

Optional fields are Avro unions ``["null", <type>]`` (null branch first), so
the datum is byte-identical to ``StandardKeyMetadata`` in the other engines.
"""

from __future__ import annotations

from io import BytesIO

from pyiceberg.avro.decoder import new_decoder
from pyiceberg.avro.encoder import BinaryEncoder

# Version byte prefixed to every encoded blob. Matches Java/Rust ``V1``.
KEY_METADATA_V1 = 1

# Valid AES key lengths in bytes (AES-128/192/256).
VALID_KEY_LENGTHS = (16, 24, 32)


def _validate_key_length(key: bytes) -> None:
    if len(key) not in VALID_KEY_LENGTHS:
        raise ValueError(f"Invalid encryption key length: {len(key)} (must be one of {VALID_KEY_LENGTHS})")


class StandardKeyMetadata:
    """Standard key metadata carrying a DEK, an optional AAD prefix and file length."""

    _encryption_key: bytes
    _aad_prefix: bytes | None
    _file_length: int | None

    def __init__(self, encryption_key: bytes, aad_prefix: bytes | None = None, file_length: int | None = None) -> None:
        _validate_key_length(encryption_key)
        self._encryption_key = encryption_key
        self._aad_prefix = aad_prefix
        self._file_length = file_length

    @property
    def encryption_key(self) -> bytes:
        """The plaintext data encryption key (DEK)."""
        return self._encryption_key

    @property
    def aad_prefix(self) -> bytes | None:
        """The additional authenticated data prefix, if set."""
        return self._aad_prefix

    @property
    def file_length(self) -> int | None:
        """The plaintext file length, if recorded."""
        return self._file_length

    def with_file_length(self, file_length: int) -> StandardKeyMetadata:
        """Return a copy with the given file length."""
        return StandardKeyMetadata(self._encryption_key, self._aad_prefix, file_length)

    def encode(self) -> bytes:
        """Serialize to ``[0x01][Avro datum]``."""
        buffer = BytesIO()
        encoder = BinaryEncoder(buffer)
        encoder.write(bytes([KEY_METADATA_V1]))
        encoder.write_bytes(self._encryption_key)
        if self._aad_prefix is None:
            encoder.write_int(0)  # union branch: null
        else:
            encoder.write_int(1)  # union branch: bytes
            encoder.write_bytes(self._aad_prefix)
        if self._file_length is None:
            encoder.write_int(0)  # union branch: null
        else:
            encoder.write_int(1)  # union branch: long
            encoder.write_int(self._file_length)
        return buffer.getvalue()

    @classmethod
    def decode(cls, data: bytes) -> StandardKeyMetadata:
        """Deserialize from ``[0x01][Avro datum]``."""
        if len(data) == 0:
            raise ValueError("Empty key metadata buffer")
        version = data[0]
        if version != KEY_METADATA_V1:
            raise ValueError(f"Cannot resolve key metadata schema for version: {version}")

        decoder = new_decoder(data[1:])
        encryption_key = decoder.read_bytes()
        aad_prefix = decoder.read_bytes() if decoder.read_int() == 1 else None
        file_length = decoder.read_int() if decoder.read_int() == 1 else None
        return cls(encryption_key, aad_prefix, file_length)

    def __eq__(self, other: object) -> bool:
        """Compare key metadata by value."""
        if not isinstance(other, StandardKeyMetadata):
            return NotImplemented
        return (
            self._encryption_key == other._encryption_key
            and self._aad_prefix == other._aad_prefix
            and self._file_length == other._file_length
        )

    def __repr__(self) -> str:
        """Return a redacted representation that never leaks key material."""
        aad = f"[{len(self._aad_prefix)} bytes]" if self._aad_prefix is not None else None
        return (
            f"StandardKeyMetadata(encryption_key=[{len(self._encryption_key)} bytes REDACTED], "
            f"aad_prefix={aad}, file_length={self._file_length})"
        )
