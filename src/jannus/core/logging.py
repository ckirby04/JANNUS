"""PHI-safe structured logging.

JANNUS runs inside hospital networks on identifiable imaging. Two rules follow
from that, and this module exists to enforce them mechanically rather than by
asking every call site to remember:

1. **Patient identifiers never reach a log file.** Case identifiers are
   pseudonymised to a stable, salted digest before they are emitted. The same
   case maps to the same token within a run, so logs remain debuggable, but the
   token cannot be reversed to a MRN without the salt.

2. **Logs are machine-readable.** `--log-format json` emits one JSON object per
   line so a site can hand a run log to their IT/security reviewer, or to us,
   without exporting anything else.

The redaction is defence in depth, not a licence to log identifiers. Prefer
passing `case_token=` over `case_id=` at the call site.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

LOGGER_NAME = "jannus"

#: Salt for case pseudonymisation. A site may set this to a secret value so
#: tokens cannot be correlated across sites; the default keeps tokens stable
#: within an installation, which is enough for debugging.
_SALT_ENV = "JANNUS_LOG_SALT"

# Patterns that must never appear in a log line. Deliberately conservative:
# a false positive costs a redacted token in a debug line, a false negative
# costs a PHI disclosure.
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Long digit runs — MRNs, accession numbers, DICOM UIDs with 8+ digits.
    (re.compile(r"\b\d{8,}\b"), "<redacted:digits>"),
    # DICOM UID form (dotted numerics, 3+ groups).
    (re.compile(r"\b\d+(?:\.\d+){3,}\b"), "<redacted:uid>"),
    # Dates that look like DOB (YYYYMMDD in a DICOM tag context).
    (re.compile(r"\b(?:19|20)\d{2}[01]\d[0-3]\d\b"), "<redacted:date>"),
    # Anything that looks like an API key.
    (re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{16,})\b"), "<redacted:key>"),
)


def pseudonymise(case_id: str, *, salt: str | None = None, length: int = 10) -> str:
    """Map a case identifier to a stable, non-reversible token.

    Not a substitute for de-identifying the *data*. This only protects the
    logs; see docs/PHI_AND_DEIDENTIFICATION.md for the data-side requirements.
    """
    salt = salt if salt is not None else os.environ.get(_SALT_ENV, "jannus-default-salt")
    digest = hashlib.sha256(f"{salt}:{case_id}".encode()).hexdigest()
    return f"case-{digest[:length]}"


def redact(text: str) -> str:
    """Strip anything matching a PHI-shaped pattern from a log message."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


class RedactingFilter(logging.Filter):
    """Applies :func:`redact` to every record before a handler formats it."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: _redact_value(v) for k, v in record.args.items()}
            else:
                record.args = tuple(_redact_value(a) for a in record.args)
        return True


def _redact_value(value: Any) -> Any:
    return redact(value) if isinstance(value, str) else value


class JsonFormatter(logging.Formatter):
    """One JSON object per line."""

    #: LogRecord attributes that are not caller-supplied context.
    _STANDARD = frozenset(
        ["name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs", "relativeCreated", "thread", "threadName", "processName", "process", "taskName"]
    )

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._STANDARD and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console output."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")


def configure_logging(
    level: str = "INFO",
    *,
    log_file: str | Path | None = None,
    fmt: str = "console",
) -> logging.Logger:
    """Set up the `jannus` logger. Idempotent — safe to call per CLI invocation.

    Args:
        level: standard level name.
        log_file: if given, additionally write JSON lines here. Parent
            directories are created. This is the file a site attaches to a
            support request.
        fmt: ``"console"`` or ``"json"`` for the stream handler.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Reconfiguring must not stack duplicate handlers across repeated calls.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    redactor = RedactingFilter()

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    stream.addFilter(redactor)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        # The file is always JSON regardless of console format — it is the
        # artefact that gets shared, so it needs to be parseable.
        file_handler.setFormatter(JsonFormatter())
        file_handler.addFilter(redactor)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the `jannus` logger."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")
