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
"""AGS1 stream-encrypted wrappers over ``InputFile`` / ``OutputFile``.

These wrap a plain file with transparent AGS1 encryption/decryption. Because
manifest metadata files are small and PyIceberg's Avro reader already loads the
whole file into memory, encryption and decryption are performed on the full
buffer rather than block-by-block. The on-disk bytes are still the standard
block-based AGS1 format, so files interoperate with Java and Rust.
"""

from __future__ import annotations

from io import BytesIO
from types import TracebackType

from pyiceberg.encryption.ciphers import calculate_plaintext_length, decrypt_stream, encrypt_stream
from pyiceberg.encryption.key_metadata import StandardKeyMetadata
from pyiceberg.io import InputFile, InputStream, OutputFile, OutputStream


class _EncryptingOutputStream(BytesIO):
    """Buffers plaintext writes and emits an AGS1 stream to ``inner`` on close."""

    def __init__(self, inner: OutputFile, key_metadata: StandardKeyMetadata, overwrite: bool) -> None:
        super().__init__()
        self._inner = inner
        self._key_metadata = key_metadata
        self._overwrite = overwrite
        self._closed = False

    def close(self) -> None:
        """Encrypt the buffered plaintext and write it to the underlying file."""
        if self._closed:
            return
        self._closed = True
        plaintext = self.getvalue()
        ciphertext = encrypt_stream(
            plaintext,
            self._key_metadata.encryption_key,
            self._key_metadata.aad_prefix or b"",
        )
        with self._inner.create(overwrite=self._overwrite) as out:
            out.write(ciphertext)
        super().close()

    def __exit__(
        self,
        exctype: type[BaseException] | None,
        excinst: BaseException | None,
        exctb: TracebackType | None,
    ) -> None:
        """Flush and encrypt on scope exit."""
        self.close()


class EncryptedOutputFile(OutputFile):
    """An ``OutputFile`` that transparently AGS1-encrypts on write."""

    def __init__(self, inner: OutputFile, key_metadata: StandardKeyMetadata) -> None:
        super().__init__(inner.location)
        self._inner = inner
        self._key_metadata = key_metadata

    @property
    def key_metadata(self) -> StandardKeyMetadata:
        """The key metadata (DEK + AAD prefix) used to encrypt this file."""
        return self._key_metadata

    def __len__(self) -> int:
        """Return the length of the underlying (encrypted) file."""
        return len(self._inner)

    def exists(self) -> bool:
        """Check whether the underlying file exists."""
        return self._inner.exists()

    def create(self, overwrite: bool = False) -> OutputStream:
        """Return a stream that encrypts buffered writes on close."""
        return _EncryptingOutputStream(self._inner, self._key_metadata, overwrite)

    def to_input_file(self) -> EncryptedInputFile:
        """Return an :class:`EncryptedInputFile` for this location."""
        return EncryptedInputFile(self._inner.to_input_file(), self._key_metadata)


class EncryptedInputFile(InputFile):
    """An ``InputFile`` that transparently AGS1-decrypts on read."""

    def __init__(self, inner: InputFile, key_metadata: StandardKeyMetadata) -> None:
        super().__init__(inner.location)
        self._inner = inner
        self._key_metadata = key_metadata

    @property
    def key_metadata(self) -> StandardKeyMetadata:
        """The key metadata (DEK + AAD prefix) used to decrypt this file."""
        return self._key_metadata

    def __len__(self) -> int:
        """Return the plaintext length of the file."""
        return calculate_plaintext_length(len(self._inner))

    def exists(self) -> bool:
        """Check whether the underlying file exists."""
        return self._inner.exists()

    def open(self, seekable: bool = True) -> InputStream:
        """Return a seekable stream over the decrypted plaintext."""
        with self._inner.open(seekable=False) as raw:
            ciphertext = raw.read()
        plaintext = decrypt_stream(
            ciphertext,
            self._key_metadata.encryption_key,
            self._key_metadata.aad_prefix or b"",
        )
        return BytesIO(plaintext)
