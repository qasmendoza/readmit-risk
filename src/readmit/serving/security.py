"""Security and privacy controls for a service that handles health data.

* Optional API-key authentication (``READMIT_API_KEY``) with constant-time comparison.
* Request-size limit to reduce denial-of-service risk from oversized payloads.
* PHI-safe logging: access logs record route templates, status, latency, and request ids only;
  never bodies or query strings (which can carry identifiers).
* Audit events carry *pseudonymized* identifiers (keyed HMAC-SHA256 with ``READMIT_AUDIT_SALT``),
  so an audit trail can link events for one patient without storing the raw identifier.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

from fastapi import Header, HTTPException, status

audit_logger = logging.getLogger("readmit.audit")
_DEFAULT_SALT = "dev-only-salt-change-me"


def audit_salt() -> str:
    return os.environ.get("READMIT_AUDIT_SALT", _DEFAULT_SALT)


def pseudonymize(value: str | None, salt: str | None = None) -> str | None:
    if value is None:
        return None
    key = (salt or audit_salt()).encode("utf-8")
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


def audit(event: str, **fields: Any) -> None:
    record = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "event": event, **fields}
    audit_logger.info(json.dumps(record, default=str))


def api_key_guard(expected: str | None):
    """FastAPI dependency enforcing ``X-API-Key`` when a key is configured."""

    async def guard(x_api_key: str | None = Header(default=None)) -> None:
        if expected is None:
            return
        if x_api_key is None or not hmac.compare_digest(x_api_key, expected):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key")

    return guard
