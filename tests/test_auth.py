"""Tests for API key authentication."""

import pytest

# Requires the optional `api` extra. This guard must precede every other
# import: a bare `import fastapi` at module scope fails collection outright on a
# core install (`pip install -e ".[dev]"`), which is what CI verifies.
pytest.importorskip("fastapi", reason="install jannus[api]")

from jannus.api.auth import _check_rate_limit, _rate_limit_windows
from jannus.api.database import Database


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "auth_test.db")
    return Database(db_path)

class TestRateLimiting:
    def test_allows_within_limit(self):
        _rate_limit_windows.clear()
        # Should not raise
        _check_rate_limit("test_key", 10)

    def test_blocks_over_limit(self):
        from fastapi import HTTPException

        _rate_limit_windows.clear()
        _rate_limit_windows["test_key_2"] = [__import__("time").time()] * 5

        with pytest.raises(HTTPException) as exc_info:
            _check_rate_limit("test_key_2", 5)
        assert exc_info.value.status_code == 429

class TestAuthPermissions:
    def test_anonymous_caller_is_denied_by_default(self, monkeypatch):
        """v1.50 SECURITY CHANGE — anonymous callers are refused by default.

        This test previously asserted the opposite: that `check_permission`
        returned silently for `key_info=None`. That behaviour made every
        `check_permission(..., "admin")` guard a no-op for unauthenticated
        requests, leaving /admin/stats, /admin/predictions (which lists case
        identifiers) and /admin/keys readable by anyone who could reach the
        port. Denying here is the fix; the test is inverted deliberately.
        """
        from fastapi import HTTPException

        from jannus.api.auth import check_permission

        monkeypatch.delenv("AUTH_REQUIRED", raising=False)
        with pytest.raises(HTTPException) as exc_info:
            check_permission(None, "predict")
        assert exc_info.value.status_code == 401

    def test_anonymous_predict_allowed_when_explicitly_opted_in(self, monkeypatch):
        from jannus.api.auth import check_permission

        monkeypatch.setenv("AUTH_REQUIRED", "false")
        check_permission(None, "predict")  # must not raise

    def test_admin_always_requires_a_key(self, monkeypatch):
        """Even with auth disabled, admin endpoints demand authentication.

        /admin/predictions enumerates case identifiers across patients, so
        "the site turned auth off" is not sufficient justification.
        """
        from fastapi import HTTPException

        from jannus.api.auth import check_permission

        monkeypatch.setenv("AUTH_REQUIRED", "false")
        with pytest.raises(HTTPException) as exc_info:
            check_permission(None, "admin")
        assert exc_info.value.status_code == 401

    def test_check_permission_valid(self):
        from jannus.api.auth import check_permission
        key_info = {"permissions": ["predict", "admin"]}
        check_permission(key_info, "predict")

    def test_check_permission_denied(self):
        from fastapi import HTTPException

        from jannus.api.auth import check_permission
        key_info = {"permissions": ["predict"]}
        with pytest.raises(HTTPException) as exc_info:
            check_permission(key_info, "admin")
        assert exc_info.value.status_code == 403
