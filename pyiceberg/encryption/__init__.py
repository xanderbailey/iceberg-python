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
"""Table encryption for PyIceberg.

This package implements Iceberg's two-tier envelope encryption, matching the
Java (``org.apache.iceberg.encryption``) and Rust (``iceberg::encryption``)
implementations so that encrypted tables interoperate across engines.

The optional ``cryptography`` dependency is required; install with the
``pyiceberg[encryption]`` extra.
"""

from __future__ import annotations

from pyiceberg.encryption.io import EncryptedInputFile, EncryptedOutputFile
from pyiceberg.encryption.key_metadata import StandardKeyMetadata
from pyiceberg.encryption.kms import GeneratedKey, InMemoryKmsClient, KmsClient, KmsClientFactory
from pyiceberg.encryption.manager import (
    EncryptionManager,
    PlaintextEncryptionManager,
    StandardEncryptionManager,
    create_encryption_manager,
)

__all__ = [
    "EncryptedInputFile",
    "EncryptedOutputFile",
    "EncryptionManager",
    "GeneratedKey",
    "InMemoryKmsClient",
    "KmsClient",
    "KmsClientFactory",
    "PlaintextEncryptionManager",
    "StandardEncryptionManager",
    "StandardKeyMetadata",
    "create_encryption_manager",
]
