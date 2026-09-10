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
"""Core AES-GCM primitives and the AGS1 stream format.

``AesGcmCipher`` provides single-shot AES-GCM (used to wrap DEKs with a KEK).
``encrypt_stream`` / ``decrypt_stream`` implement the block-based AGS1 format
used for encrypted Avro manifest lists and manifest files, byte-compatible with
Java's ``AesGcmOutputStream`` / ``AesGcmInputStream`` and Rust's ``stream.rs``.

AGS1 layout::

    Header (8 bytes): magic "AGS1" + plaintext block size (u32 little-endian)
    Block N: nonce (12B) + ciphertext (<= block size) + GCM tag (16B)
    Per-block AAD = aad_prefix || block_index (u32 little-endian)
"""

from __future__ import annotations

import os

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError(
        "Table encryption requires the 'cryptography' package. Install with: pip install 'pyiceberg[encryption]'"
    ) from exc

# Default plaintext block size (1 MiB), matching Java's Ciphers.PLAIN_BLOCK_SIZE.
PLAIN_BLOCK_SIZE = 1024 * 1024
# AES-GCM nonce length in bytes (96 bits).
NONCE_LENGTH = 12
# AES-GCM authentication tag length in bytes (128 bits).
GCM_TAG_LENGTH = 16
# Cipher block size = plaintext block + nonce + tag.
CIPHER_BLOCK_SIZE = PLAIN_BLOCK_SIZE + NONCE_LENGTH + GCM_TAG_LENGTH
# AGS1 stream magic bytes.
GCM_STREAM_MAGIC = b"AGS1"
# AGS1 header length: 4-byte magic + 4-byte block size.
GCM_STREAM_HEADER_LENGTH = 8

VALID_KEY_LENGTHS = (16, 24, 32)


class AesGcmCipher:
    """Single-shot AES-GCM cipher for wrapping/unwrapping keys.

    Produces ``nonce || ciphertext || tag`` and reverses it, matching the
    layout used by the other engines for DEK wrapping.
    """

    def __init__(self, key: bytes) -> None:
        if len(key) not in VALID_KEY_LENGTHS:
            raise ValueError(f"Invalid AES key length: {len(key)} (must be one of {VALID_KEY_LENGTHS})")
        self._aesgcm = AESGCM(key)

    def encrypt(self, plaintext: bytes, aad: bytes | None = None) -> bytes:
        """Encrypt ``plaintext``, returning ``nonce || ciphertext || tag``."""
        nonce = os.urandom(NONCE_LENGTH)
        # cryptography appends the tag to the ciphertext.
        return nonce + self._aesgcm.encrypt(nonce, plaintext, aad)

    def decrypt(self, data: bytes, aad: bytes | None = None) -> bytes:
        """Decrypt ``nonce || ciphertext || tag`` back to plaintext."""
        if len(data) < NONCE_LENGTH + GCM_TAG_LENGTH:
            raise ValueError("Ciphertext too short to contain nonce and tag")
        nonce, ciphertext = data[:NONCE_LENGTH], data[NONCE_LENGTH:]
        return self._aesgcm.decrypt(nonce, ciphertext, aad)


def _block_aad(aad_prefix: bytes, block_index: int) -> bytes:
    """Return ``aad_prefix || block_index`` (u32 little-endian)."""
    index_bytes = block_index.to_bytes(4, "little")
    return aad_prefix + index_bytes if aad_prefix else index_bytes


def calculate_plaintext_length(ciphertext_length: int) -> int:
    """Return the plaintext length for an AGS1 stream of the given total size."""
    stream_length = ciphertext_length - GCM_STREAM_HEADER_LENGTH
    if stream_length < 0:
        raise ValueError(f"Invalid AGS1 stream length: {ciphertext_length}")
    if stream_length == 0:
        return 0

    num_full_blocks, remainder = divmod(stream_length, CIPHER_BLOCK_SIZE)
    overhead = NONCE_LENGTH + GCM_TAG_LENGTH
    plaintext = num_full_blocks * PLAIN_BLOCK_SIZE
    if remainder > 0:
        if remainder < overhead:
            raise ValueError(f"Invalid AGS1 final block size: {remainder}")
        plaintext += remainder - overhead
    return plaintext


def encrypt_stream(plaintext: bytes, key: bytes, aad_prefix: bytes = b"") -> bytes:
    """Encrypt ``plaintext`` into a complete AGS1 stream."""
    aesgcm = AESGCM(key)
    out = bytearray()
    out += GCM_STREAM_MAGIC
    out += PLAIN_BLOCK_SIZE.to_bytes(4, "little")

    # An empty payload still emits a single (empty) block, matching the readers.
    offset = 0
    block_index = 0
    total = len(plaintext)
    while True:
        chunk = plaintext[offset : offset + PLAIN_BLOCK_SIZE]
        nonce = os.urandom(NONCE_LENGTH)
        out += nonce
        out += aesgcm.encrypt(nonce, chunk, _block_aad(aad_prefix, block_index))
        offset += len(chunk)
        block_index += 1
        if offset >= total:
            break
    return bytes(out)


def decrypt_stream(ciphertext: bytes, key: bytes, aad_prefix: bytes = b"") -> bytes:
    """Decrypt a complete AGS1 stream back to plaintext."""
    if len(ciphertext) < GCM_STREAM_HEADER_LENGTH:
        raise ValueError("AGS1 stream shorter than header")
    if ciphertext[:4] != GCM_STREAM_MAGIC:
        raise ValueError(f"Invalid AGS1 magic bytes: {ciphertext[:4]!r}")

    aesgcm = AESGCM(key)
    out = bytearray()
    offset = GCM_STREAM_HEADER_LENGTH
    block_index = 0
    overhead = NONCE_LENGTH + GCM_TAG_LENGTH
    total = len(ciphertext)
    while offset < total:
        block = ciphertext[offset : offset + CIPHER_BLOCK_SIZE]
        if len(block) < overhead:
            raise ValueError(f"Truncated AGS1 block at offset {offset}")
        nonce, payload = block[:NONCE_LENGTH], block[NONCE_LENGTH:]
        out += aesgcm.decrypt(nonce, payload, _block_aad(aad_prefix, block_index))
        offset += len(block)
        block_index += 1
    return bytes(out)
