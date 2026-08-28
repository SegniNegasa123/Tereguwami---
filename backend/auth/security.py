"""
Authentication, Token Issuance & Security Primitives
Part of Tereguwami (ተርጓሚ) Backend Auth Tier (§11, §12)
"""

import os
import time
import hmac
import hashlib
import base64
import json
from typing import Dict, Optional, Any


class SecurityManager:
    """
    Handles cryptographic password hashing, API key verification, and
    stateless HMAC-SHA256 signed bearer tokens.
    """

    SECRET_KEY = os.environ.get("TEREGUWAMI_SECRET_KEY", "tereguwami_super_secret_production_key_2026")
    TOKEN_EXPIRATION_SECONDS = 86400 * 7  # 7 days

    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> str:
        """Secure PBKDF2-HMAC-SHA256 password hashing."""
        if salt is None:
            salt = base64.b64encode(os.urandom(16)).decode("utf-8")
        key = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            100000
        )
        return f"{salt}${base64.b64encode(key).decode('utf-8')}"

    @staticmethod
    def verify_password(password: str, hashed_value: str) -> bool:
        """Verify candidate password against salt$hash record."""
        try:
            salt, key_b64 = hashed_value.split("$")
            candidate_key = hashlib.pbkdf2_hmac(
                "sha256",
                password.encode("utf-8"),
                salt.encode("utf-8"),
                100000
            )
            return hmac.compare_digest(base64.b64encode(candidate_key).decode("utf-8"), key_b64)
        except Exception:
            return False

    def create_access_token(self, user_id: str, role: str = "registered_signer") -> str:
        """Create an HMAC-SHA256 signed JSON Web Token."""
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": user_id,
            "role": role,
            "exp": int(time.time()) + self.TOKEN_EXPIRATION_SECONDS,
            "iat": int(time.time())
        }

        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        signing_input = f"{header_b64}.{payload_b64}"

        signature = hmac.new(
            self.SECRET_KEY.encode(),
            signing_input.encode(),
            hashlib.sha256
        ).digest()
        sig_b64 = base64.urlsafe_b64encode(signature).decode().rstrip("=")

        return f"{signing_input}.{sig_b64}"

    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode signed bearer token."""
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            header_b64, payload_b64, sig_b64 = parts
            signing_input = f"{header_b64}.{payload_b64}"

            expected_sig = hmac.new(
                self.SECRET_KEY.encode(),
                signing_input.encode(),
                hashlib.sha256
            ).digest()
            expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).decode().rstrip("=")

            if not hmac.compare_digest(sig_b64, expected_sig_b64):
                return None

            # Restore base64 padding
            payload_padded = payload_b64 + "=" * (-len(payload_b64) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_padded.encode()).decode())

            # Check expiration
            if payload.get("exp", 0) < time.time():
                return None

            return payload
        except Exception:
            return None


security_manager = SecurityManager()
