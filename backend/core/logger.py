"""
ORBITAL Structured JSON Logger
Outputs each log entry as a single JSON line with IST timestamps.
"""

import json
import logging
import sys
from datetime import datetime, timezone, timedelta


# IST timezone offset: UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30))


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON with IST timestamps."""

    def format(self, record: logging.LogRecord) -> str:
        # Build the base log entry
        log_entry = {
            "timestamp": datetime.now(IST).isoformat(),
            "level": record.levelname,
            "module": record.name,
            "message": record.getMessage(),
        }

        # Merge any extra kwargs passed via the `extra` dict
        # We look for a special key '_extra_fields' that our LoggerAdapter injects
        if hasattr(record, "_extra_fields") and isinstance(record._extra_fields, dict):
            log_entry.update(record._extra_fields)

        # Also capture exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False, default=str)


class OrbitalLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that allows passing extra kwargs directly in log calls.
    
    Usage:
        logger = get_logger(__name__)
        logger.info("Pipeline started", filename="rbi.pdf", source="RBI")
    """

    def process(self, msg, kwargs):
        # Extract extra fields from kwargs (anything that isn't a standard logging kwarg)
        standard_keys = {"exc_info", "stack_info", "stacklevel", "extra"}
        extra_fields = {k: v for k, v in kwargs.items() if k not in standard_keys}

        # Remove our custom keys so logging doesn't complain
        for k in extra_fields:
            kwargs.pop(k, None)

        # Inject extra fields into the record via the 'extra' dict
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"]["_extra_fields"] = extra_fields

        return msg, kwargs


def get_logger(name: str) -> OrbitalLoggerAdapter:
    """
    Create and return a structured JSON logger for the given module name.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        An OrbitalLoggerAdapter that outputs JSON log lines to stderr.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(JSONFormatter())

        logger.addHandler(handler)

    # Prevent propagation to root logger to avoid duplicate output
    logger.propagate = False

    return OrbitalLoggerAdapter(logger, {})
