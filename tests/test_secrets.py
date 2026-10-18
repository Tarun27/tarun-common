"""tarun_common.secrets — including that ciphertexts made by the apps' old code still
decrypt, which is the whole point of keeping plain Fernet."""

import base64
import hashlib
import subprocess
import sys

import pytest
from cryptography.fernet import Fernet

from tarun_common import secrets
from tarun_common.secrets import SecretBox, SecretKeyError


def test_round_trip():
    box = SecretBox(secrets.generate_key(), name="CREDENTIAL_KEY")
    token = box.encrypt("sk-ant-abc")
    assert token != "sk-ant-abc"
    assert box.decrypt(token) == "sk-ant-abc"
    assert box.decrypt_bytes(box.encrypt_bytes(b"\x00pdf")) == b"\x00pdf"


def test_reads_what_plain_fernet_wrote_and_vice_versa():
    key = Fernet.generate_key()
    old_token = Fernet(key).encrypt(b"stored before the move").decode()
    box = SecretBox(key.decode())
    assert box.decrypt(old_token) == "stored before the move"
    assert Fernet(key).decrypt(box.encrypt("x").encode()) == b"x"


def test_derive_key_matches_resume_makers_old_derivation():
    secret = "jwt-signing-secret"
    old = base64.urlsafe_b64encode(hashlib.sha256(f"api-keys:{secret}".encode()).digest())
    assert secrets.derive_key(secret, "api-keys") == old.decode()
    old_token = Fernet(old).encrypt(b"user key").decode()
    assert SecretBox(secrets.derive_key(secret, "api-keys")).decrypt(old_token) == "user key"


@pytest.mark.parametrize("key", [None, ""])
def test_missing_key_says_which_and_how(key):
    with pytest.raises(SecretKeyError, match=r"CREDENTIAL_KEY is not set.*Fernet\.generate_key"):
        SecretBox(key, name="CREDENTIAL_KEY")


def test_malformed_key():
    with pytest.raises(SecretKeyError, match="MANGO_SECRET_KEY is not a valid Fernet key"):
        SecretBox("not-a-key", name="MANGO_SECRET_KEY")


def test_wrong_key_or_corrupt_value():
    token = SecretBox(secrets.generate_key()).encrypt("x")
    other = SecretBox(secrets.generate_key(), name="INTEGRATIONS_ENCRYPTION_KEY")
    with pytest.raises(SecretKeyError, match="INTEGRATIONS_ENCRYPTION_KEY changed"):
        other.decrypt(token)
    with pytest.raises(SecretKeyError):
        other.decrypt("garbage")
    assert other.decrypt_or_none(token) is None
    assert other.decrypt_or_none(None) is None
    assert other.decrypt_or_none("") is None


def test_importing_the_module_does_not_import_cryptography():
    code = "import sys, tarun_common.secrets; print('cryptography' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False", out.stderr
