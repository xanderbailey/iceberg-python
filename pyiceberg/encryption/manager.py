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
"""File-level encryption managers implementing Iceberg's envelope model.

Two managers mirror Java/Rust:

* :class:`PlaintextEncryptionManager` — a no-op used when a table is not
  encrypted.
* :class:`StandardEncryptionManager` — two-tier envelope encryption. A master
  key in a KMS wraps a Key Encryption Key (KEK) stored in table metadata; the
  KEK wraps per-file Data Encryption Keys (DEKs) stored in ``key_metadata``.
"""

from __future__ import annotations

import os
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pyiceberg.encryption.ciphers import AesGcmCipher
from pyiceberg.encryption.io import EncryptedInputFile, EncryptedOutputFile
from pyiceberg.encryption.key_metadata import StandardKeyMetadata
from pyiceberg.encryption.kms import KmsClient
from pyiceberg.io import InputFile, OutputFile

if TYPE_CHECKING:
    from pyiceberg.table.metadata import EncryptedKey, TableMetadata

# Property recording the KEK creation time (millis since epoch). Matches Java's
# StandardEncryptionManager.KEY_TIMESTAMP; used as AAD when wrapping DEKs.
KEK_TIMESTAMP_PROPERTY = "KEY_TIMESTAMP"
# Default KEK lifespan in days (NIST SP 800-57), after which a new KEK is created.
DEFAULT_KEK_LIFESPAN_MS = 730 * 24 * 60 * 60 * 1000
# Default unwrapped-KEK cache TTL in seconds.
DEFAULT_KEK_CACHE_TTL_S = 3600
# AAD prefix length in bytes (matches TableProperties.ENCRYPTION_AAD_LENGTH_DEFAULT).
AAD_PREFIX_LENGTH = 16


class EncryptionManager(ABC):
    """Base class for file-level encryption managers."""

    @abstractmethod
    def encrypt(self, raw_output: OutputFile) -> OutputFile:
        """Return an output file that encrypts on write (or the input unchanged)."""

    @abstractmethod
    def decrypt(self, raw_input: InputFile, key_metadata: bytes | None) -> InputFile:
        """Return an input file that decrypts on read (or the input unchanged)."""


class PlaintextEncryptionManager(EncryptionManager):
    """No-op manager for unencrypted tables."""

    def encrypt(self, raw_output: OutputFile) -> OutputFile:
        """Return ``raw_output`` unchanged."""
        return raw_output

    def decrypt(self, raw_input: InputFile, key_metadata: bytes | None) -> InputFile:
        """Return ``raw_input`` unchanged."""
        return raw_input


class StandardEncryptionManager(EncryptionManager):
    """Two-tier envelope encryption manager."""

    def __init__(
        self,
        table_key_id: str,
        kms_client: KmsClient,
        encryption_keys: dict[str, EncryptedKey] | None = None,
        data_key_length: int = 16,
    ) -> None:
        if data_key_length not in (16, 24, 32):
            raise ValueError(f"Invalid encryption.data-key-length: {data_key_length} (must be 16, 24 or 32)")
        self._table_key_id = table_key_id
        self._kms_client = kms_client
        self._data_key_length = data_key_length
        self._encryption_keys: dict[str, EncryptedKey] = dict(encryption_keys or {})
        # Cache of unwrapped KEK bytes: key_id -> (plaintext_kek, inserted_at_epoch_s).
        self._kek_cache: dict[str, tuple[bytes, float]] = {}

    @property
    def encryption_keys(self) -> dict[str, EncryptedKey]:
        """The current key set (KEKs + wrapped manifest-list entries) to persist at commit."""
        return dict(self._encryption_keys)

    def encrypt(self, raw_output: OutputFile) -> EncryptedOutputFile:
        """Wrap ``raw_output`` with a fresh DEK and random AAD prefix (AGS1)."""
        dek = os.urandom(self._data_key_length)
        aad_prefix = os.urandom(AAD_PREFIX_LENGTH)
        key_metadata = StandardKeyMetadata(dek, aad_prefix)
        return EncryptedOutputFile(raw_output, key_metadata)

    def decrypt(self, raw_input: InputFile, key_metadata: bytes | None) -> InputFile:
        """Wrap ``raw_input`` with the DEK from ``key_metadata`` (AGS1)."""
        if key_metadata is None:
            return raw_input
        return EncryptedInputFile(raw_input, StandardKeyMetadata.decode(key_metadata))

    def add_manifest_list_key_metadata(self, key_metadata: StandardKeyMetadata) -> str:
        """Wrap a manifest-list DEK with the active KEK and store it.

        Returns the id of the wrapped entry, to be recorded on the snapshot's
        ``key-id`` so readers can locate it.
        """
        kek = self._find_active_kek() or self._create_kek()
        kek_bytes = self._unwrap_kek(kek)
        aad = self._kek_timestamp_aad(kek)
        wrapped = AesGcmCipher(kek_bytes).encrypt(key_metadata.encode(), aad)

        from pyiceberg.table.metadata import EncryptedKey

        entry = EncryptedKey(
            key_id=str(uuid.uuid4()),
            encrypted_key_metadata=wrapped,
            encrypted_by_id=kek.key_id,
        )
        self._encryption_keys[entry.key_id] = entry
        return entry.key_id

    def load_manifest_list_key_metadata(self, key_id: str) -> StandardKeyMetadata:
        """Reverse of :meth:`add_manifest_list_key_metadata`."""
        entry = self._encryption_keys.get(key_id)
        if entry is None:
            raise ValueError(f"Encryption key '{key_id}' not found")
        if entry.encrypted_by_id is None:
            raise ValueError(f"EncryptedKey '{key_id}' has no encrypted-by-id")

        kek = self._encryption_keys.get(entry.encrypted_by_id)
        if kek is None:
            raise ValueError(f"KEK not found in encryption keys: {entry.encrypted_by_id}")
        kek_bytes = self._unwrap_kek(kek)
        aad = self._kek_timestamp_aad(kek)
        plaintext = AesGcmCipher(kek_bytes).decrypt(entry.encrypted_key_metadata, aad)
        return StandardKeyMetadata.decode(plaintext)

    def _create_kek(self) -> EncryptedKey:
        if self._kms_client.supports_key_generation():
            generated = self._kms_client.generate_key(self._table_key_id)
            plaintext_kek, wrapped_kek = generated.key, generated.wrapped_key
        else:
            plaintext_kek = os.urandom(self._data_key_length)
            wrapped_kek = self._kms_client.wrap_key(plaintext_kek, self._table_key_id)

        key_id = str(uuid.uuid4())
        now_ms = int(time.time() * 1000)
        self._kek_cache[key_id] = (plaintext_kek, time.time())

        from pyiceberg.table.metadata import EncryptedKey

        kek = EncryptedKey(
            key_id=key_id,
            encrypted_key_metadata=wrapped_kek,
            encrypted_by_id=self._table_key_id,
            properties={KEK_TIMESTAMP_PROPERTY: str(now_ms)},
        )
        self._encryption_keys[key_id] = kek
        return kek

    def _find_active_kek(self) -> EncryptedKey | None:
        candidates = [
            kek
            for kek in self._encryption_keys.values()
            if kek.encrypted_by_id == self._table_key_id and not self._is_kek_expired(kek)
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda kek: int(kek.properties.get(KEK_TIMESTAMP_PROPERTY, "0")))

    def _is_kek_expired(self, kek: EncryptedKey) -> bool:
        raw = kek.properties.get(KEK_TIMESTAMP_PROPERTY)
        if raw is None:
            return True
        created_at_ms = int(raw)
        return (int(time.time() * 1000) - created_at_ms) >= DEFAULT_KEK_LIFESPAN_MS

    def _unwrap_kek(self, kek: EncryptedKey) -> bytes:
        cached = self._kek_cache.get(kek.key_id)
        if cached is not None and (time.time() - cached[1]) < DEFAULT_KEK_CACHE_TTL_S:
            return cached[0]
        if kek.encrypted_by_id is None:
            raise ValueError(f"KEK '{kek.key_id}' has no encrypted-by-id")
        plaintext = self._kms_client.unwrap_key(kek.encrypted_key_metadata, kek.encrypted_by_id)
        self._kek_cache[kek.key_id] = (plaintext, time.time())
        return plaintext

    @staticmethod
    def _kek_timestamp_aad(kek: EncryptedKey) -> bytes:
        raw = kek.properties.get(KEK_TIMESTAMP_PROPERTY)
        if raw is None:
            raise ValueError(f"KEK '{kek.key_id}' is missing required '{KEK_TIMESTAMP_PROPERTY}' property")
        return raw.encode("utf-8")


def create_encryption_manager(metadata: TableMetadata, kms_client: KmsClient | None) -> EncryptionManager:
    """Build an :class:`EncryptionManager` from table metadata.

    Returns a :class:`PlaintextEncryptionManager` when the format version is
    below 3 or ``encryption.key-id`` is not set. Raises when the property is set
    but no KMS client is available.
    """
    from pyiceberg.table import TableProperties

    if metadata.format_version < 3:
        return PlaintextEncryptionManager()

    table_key_id = metadata.properties.get(TableProperties.ENCRYPTION_KEY_ID)
    if table_key_id is None:
        return PlaintextEncryptionManager()

    if kms_client is None:
        raise ValueError(
            f"Table property '{TableProperties.ENCRYPTION_KEY_ID}' is set but no KMS client is configured "
            f"(set '{TableProperties.ENCRYPTION_KMS_IMPL}')"
        )

    data_key_length = int(
        metadata.properties.get(
            TableProperties.ENCRYPTION_DATA_KEY_LENGTH,
            TableProperties.ENCRYPTION_DATA_KEY_LENGTH_DEFAULT,
        )
    )
    return StandardEncryptionManager(
        table_key_id=table_key_id,
        kms_client=kms_client,
        encryption_keys={key.key_id: key for key in metadata.encryption_keys},
        data_key_length=data_key_length,
    )
