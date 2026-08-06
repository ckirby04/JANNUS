"""
API key authentication middleware for BrainMetScan.
Supports API key validation via header, rate limiting, and permission checking.
"""

import time
from collections import defaultdict

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from .database import Database

# API key header scheme
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In-memory rate limit tracking: {key_id: [(timestamp, ...)] }
_rate_limit_windows: dict = defaultdict(list)


def _get_db() -> Database:
    """Lazy database initialization."""
    if not hasattr(_get_db, "_instance"):
        _get_db._instance = Database()
    return _get_db._instance


def set_db(db: Database):
    """Set the database instance (for testing or custom configuration)."""
    _get_db._instance = db


#: Requests per minute allowed to an unauthenticated caller, keyed by client IP.
#: Only reachable when AUTH_REQUIRED is explicitly disabled.
ANONYMOUS_RATE_LIMIT = 10

#: Permissions that require a real API key even when anonymous access has been
#: explicitly enabled. These read or mutate cross-patient state — /admin/predictions
#: lists case identifiers — so "the site turned auth off" is not sufficient
#: justification for granting them.
ALWAYS_AUTHENTICATED = frozenset({"admin"})


def auth_required() -> bool:
    """Whether an API key is mandatory.

    Defaults to True. A site that genuinely wants an open instance sets
    AUTH_REQUIRED=false explicitly and accepts the consequences.
    """
    import os

    return os.environ.get("AUTH_REQUIRED", "true").lower() != "false"


async def get_api_key_info(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
) -> dict | None:
    """
    Dependency that validates the API key.

    v1.50 SECURITY CHANGE — the default is now `AUTH_REQUIRED=true`.

    v1.40 defaulted to `false`, so a site that followed the README and ran
    `python run_server.py` got an unauthenticated service handling patient
    imaging, with every `/admin/*` endpoint world-readable. Sites that
    deliberately want an open instance (an isolated research VM, say) must now
    opt in with AUTH_REQUIRED=false rather than getting it by accident.
    """
    if api_key is None or api_key == "":
        if auth_required():
            raise HTTPException(status_code=401, detail="API key required. Provide X-API-Key header.")
        # Anonymous access was explicitly enabled. Still rate-limit by client
        # IP: /predict is GPU-bound, and in v1.40 the anonymous path bypassed
        # the limiter entirely, making denial of service trivial.
        client_host = request.client.host if request.client else "unknown"
        _check_rate_limit(f"anon:{client_host}", ANONYMOUS_RATE_LIMIT)
        return None

    db = _get_db()
    key_info = db.validate_api_key(api_key)

    if key_info is None:
        raise HTTPException(status_code=401, detail="Invalid or expired API key")

    # Rate limiting
    _check_rate_limit(key_info["key_id"], key_info["rate_limit_per_minute"])

    return key_info


def _check_rate_limit(key_id: str, limit_per_minute: int):
    """Simple sliding window rate limiter."""
    now = time.time()
    window_start = now - 60

    # Clean old entries
    _rate_limit_windows[key_id] = [
        t for t in _rate_limit_windows[key_id] if t > window_start
    ]

    if len(_rate_limit_windows[key_id]) >= limit_per_minute:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Maximum {limit_per_minute} requests per minute.",
        )

    _rate_limit_windows[key_id].append(now)


def check_permission(key_info: dict | None, permission: str):
    """Check that the caller holds `permission`, denying anonymous callers.

    v1.50 SECURITY CHANGE — an anonymous caller (`key_info is None`) is now
    denied instead of silently permitted.

    In v1.40 this function returned early on None, which made every
    `check_permission(..., "admin")` guard a no-op for unauthenticated
    requests. Combined with the old `AUTH_REQUIRED=false` default, that left
    `/admin/stats`, `/admin/predictions` (which lists case identifiers) and
    `/admin/keys` readable by anyone who could reach the port.

    Anonymous callers are now refused unless a site has *explicitly* set
    AUTH_REQUIRED=false, and even then anything in `ALWAYS_AUTHENTICATED`
    still demands a real key.
    """
    if key_info is None:
        if permission in ALWAYS_AUTHENTICATED or auth_required():
            raise HTTPException(
                status_code=401,
                detail=(
                    f"Authentication required for '{permission}'. "
                    f"Provide an X-API-Key header."
                ),
            )
        # Anonymous access was deliberately enabled for this deployment.
        return

    if permission not in key_info.get("permissions", []):
        raise HTTPException(
            status_code=403,
            detail=f"API key does not have '{permission}' permission",
        )
