"""JANNUS API server entry point.

    python run_server.py                       # localhost only (default)
    python run_server.py --host 0.0.0.0        # all interfaces — see below

The API is OPTIONAL. External validation sites do not need it: use the `jannus`
CLI, which needs no server, no credentials and no network.

v1.50 SECURITY CHANGE — the default bind address is now 127.0.0.1.

v1.40 defaulted to 0.0.0.0, which combined with authentication being disabled
by default meant the out-of-the-box deployment was an unauthenticated PHI
service listening on every interface. On a flat hospital network that is
reachable from any workstation.

Before exposing this beyond localhost:

  * keep AUTH_REQUIRED=true (the default) and issue API keys;
  * put a reverse proxy in front that terminates TLS — JANNUS provides none,
    so PHI would otherwise transit in cleartext;
  * set CORS_ORIGINS explicitly, or leave it empty to disable cross-origin
    access entirely;
  * disable FastAPI's /docs and /openapi.json.

See SECURITY.md and docs/PHI_AND_DEIDENTIFICATION.md.
"""

from __future__ import annotations

import argparse
import os
import sys

import uvicorn


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the JANNUS API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host", default=os.environ.get("JANNUS_HOST", "127.0.0.1"),
        help="Bind address (default: 127.0.0.1, localhost only)",
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("JANNUS_PORT", "8000")),
        help="Bind port (default: 8000)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Worker processes. Keep at 1: the model holds GPU memory, and the "
             "rate limiter is per-process (default: 1)",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Auto-reload for development. Never use on a host holding patient data.",
    )
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        auth_disabled = os.environ.get("AUTH_REQUIRED", "true").lower() == "false"
        print(
            f"WARNING: binding to {args.host} exposes this server beyond localhost.\n"
            f"         Ensure TLS termination and authentication are in place.\n"
            f"         See SECURITY.md.",
            file=sys.stderr,
        )
        if auth_disabled:
            # Refuse the one combination that is indefensible: reachable from
            # the network with authentication explicitly turned off.
            print(
                "ERROR: AUTH_REQUIRED=false with a non-localhost bind address would "
                "expose an unauthenticated service handling patient imaging.\n"
                "       Remove AUTH_REQUIRED=false, or bind to 127.0.0.1.",
                file=sys.stderr,
            )
            return 2

    uvicorn.run(
        "jannus.api.server:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
