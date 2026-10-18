"""Secrets at rest: Fernet (AES-128-CBC + HMAC) under a key from the app's .env.

Orange, Mango, SDE Prep and Resume Maker each wrapped `cryptography.fernet` the same way:
check the key, encrypt a string, decrypt it, and say something useful when the key is
missing or has changed. This module is that wrapper. What stays in each app is its
policy: which env var holds the key, whether a missing key disables a feature (Orange),
is generated into .env (Mango) or is an error (SDE Prep), and what a failed decrypt
means to its users.

The format is plain Fernet, byte for byte what the apps already stored, so existing
ciphertexts keep decrypting: no re-encryption, no new envelope. `derive_key` is Resume
Maker's existing derivation, kept identical for the same reason.

`cryptography` is imported only when a box is made. Nothing here logs a key or a value.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Any

GENERATE_HINT = (
    'python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"'
)


class SecretKeyError(RuntimeError):
    """The key is missing or malformed, or a value won't decrypt under it."""


def generate_key() -> str:
    """A new random Fernet key (urlsafe base64, 32 bytes)."""
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def derive_key(secret: str, purpose: str) -> str:
    """A Fernet key derived from an app secret: urlsafe_b64(sha256("<purpose>:<secret>")).

    Resume Maker's scheme for user API keys (purpose "api-keys"). Changing the secret
    makes everything under the derived key unreadable, so prefer a dedicated key from
    `generate_key` for anything new.
    """
    return base64.urlsafe_b64encode(
        hashlib.sha256(f"{purpose}:{secret}".encode()).digest()
    ).decode()


class SecretBox:
    """Encrypt and decrypt under one Fernet key.

    `name` is the env var the key came from; it goes into error messages so the fix is
    obvious ("CREDENTIAL_KEY is not set …").
    """

    def __init__(self, key: str | bytes | None, *, name: str = "the encryption key") -> None:
        self.name = name
        if not key:
            raise SecretKeyError(f"{name} is not set. Generate one with: {GENERATE_HINT}")
        from cryptography.fernet import Fernet

        try:
            self._fernet: Any = Fernet(key.encode() if isinstance(key, str) else key)
        except (ValueError, TypeError) as e:
            raise SecretKeyError(
                f"{name} is not a valid Fernet key (32 bytes, urlsafe base64). "
                f"Generate one with: {GENERATE_HINT}"
            ) from e

    def encrypt(self, plaintext: str) -> str:
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        """Raises SecretKeyError when the value won't decrypt (the key changed, or the
        stored value is corrupt)."""
        return self.decrypt_bytes(token.encode()).decode()

    def decrypt_or_none(self, token: str | None) -> str | None:
        """None for an empty or undecryptable value, for callers that fall back (e.g. to
        a fresh login) instead of failing."""
        if not token:
            return None
        try:
            return self.decrypt(token)
        except SecretKeyError:
            return None

    def encrypt_bytes(self, data: bytes) -> bytes:
        return bytes(self._fernet.encrypt(data))

    def decrypt_bytes(self, token: bytes) -> bytes:
        from cryptography.fernet import InvalidToken

        try:
            return bytes(self._fernet.decrypt(token))
        except (InvalidToken, ValueError) as e:
            raise SecretKeyError(
                f"A stored value can't be decrypted: {self.name} changed since it was saved, "
                "or the value is corrupt"
            ) from e
