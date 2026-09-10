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
"""Pluggable Key Management System (KMS) integration.

Mirrors Java's ``KeyManagementClient`` interface and Rust's
``KeyManagementClient`` trait. A KMS holds master keys and wraps/unwraps the
key-encryption keys (KEKs) that Iceberg stores in table metadata.
"""

from __future__ import annotations

import importlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from pyiceberg.encryption.ciphers import AesGcmCipher
from pyiceberg.typedef import Properties

# Property naming the KMS client implementation (dotted path), matching Java's
# reflection-based ``encryption.kms-impl``.
KMS_IMPL = "encryption.kms-impl"


@dataclass(frozen=True)
class GeneratedKey:
    """A freshly generated key together with its wrapped (encrypted) form.

    Returned by KMS backends that support atomic server-side key generation.
    """

    key: bytes
    wrapped_key: bytes


class KmsClient(ABC):
    """Pluggable interface for key management systems (AWS KMS, Azure Key Vault, ...)."""

    @abstractmethod
    def wrap_key(self, key: bytes, wrapping_key_id: str) -> bytes:
        """Wrap (encrypt) ``key`` using the wrapping key identified by ``wrapping_key_id``."""

    @abstractmethod
    def unwrap_key(self, wrapped_key: bytes, wrapping_key_id: str) -> bytes:
        """Unwrap (decrypt) a previously wrapped key."""

    def supports_key_generation(self) -> bool:
        """Whether this KMS can generate and wrap keys server-side."""
        return False

    def generate_key(self, wrapping_key_id: str) -> GeneratedKey:
        """Generate a new key and wrap it atomically on the server side.

        Only supported when :meth:`supports_key_generation` returns ``True``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support server-side key generation")


class KmsClientFactory(ABC):
    """Factory that constructs a :class:`KmsClient` from catalog/table properties."""

    @abstractmethod
    def create_kms_client(self, properties: Properties) -> KmsClient:
        """Create a :class:`KmsClient`, called once during catalog/table initialization."""


class InMemoryKmsClient(KmsClient):
    """In-memory KMS for development and tests only.

    Master keys are held in-process and used to AES-GCM wrap/unwrap KEKs.
    Not for production use: it provides no durability or access control. Mirrors
    Rust's ``MemoryKeyManagementClient`` and Java's ``UnitestKMS``.
    """

    def __init__(self, master_keys: dict[str, bytes] | None = None) -> None:
        self._master_keys: dict[str, bytes] = dict(master_keys or {})

    def add_master_key(self, key_id: str, key: bytes | None = None) -> bytes:
        """Register a master key (generating a random 16-byte key if none given)."""
        key = key if key is not None else os.urandom(16)
        self._master_keys[key_id] = key
        return key

    def _cipher(self, wrapping_key_id: str) -> AesGcmCipher:
        try:
            master_key = self._master_keys[wrapping_key_id]
        except KeyError as exc:
            raise ValueError(f"Unknown master key id: {wrapping_key_id}") from exc
        return AesGcmCipher(master_key)

    def wrap_key(self, key: bytes, wrapping_key_id: str) -> bytes:
        """Wrap ``key`` under the master key, using the key id as AAD."""
        return self._cipher(wrapping_key_id).encrypt(key, wrapping_key_id.encode("utf-8"))

    def unwrap_key(self, wrapped_key: bytes, wrapping_key_id: str) -> bytes:
        """Unwrap a key previously wrapped by :meth:`wrap_key`."""
        return self._cipher(wrapping_key_id).decrypt(wrapped_key, wrapping_key_id.encode("utf-8"))


def load_kms_client(properties: Properties) -> KmsClient | None:
    """Instantiate the KMS client named by the ``encryption.kms-impl`` property.

    The implementation must be either a :class:`KmsClient` subclass (constructed
    with ``properties``) or a :class:`KmsClientFactory` subclass (whose
    ``create_kms_client`` is called). Returns ``None`` when the property is unset.
    """
    impl = properties.get(KMS_IMPL)
    if impl is None:
        return None

    module_name, _, class_name = impl.rpartition(".")
    if not module_name:
        raise ValueError(f"Invalid {KMS_IMPL} value (expected a dotted path): {impl}")
    cls = getattr(importlib.import_module(module_name), class_name)

    instance = cls()
    if isinstance(instance, KmsClientFactory):
        return instance.create_kms_client(properties)
    if isinstance(instance, KmsClient):
        return instance
    raise ValueError(f"{impl} is neither a KmsClient nor a KmsClientFactory")
