"""Disk-based caching decorator.

Provides @cached(ttl=...), which transparently caches a function's
JSON-serializable return value to disk for a given time-to-live. Cache
files are keyed by function name plus a hash of the arguments, so different
arguments cache independently. Cache read/write failures degrade gracefully:
on any cache error the wrapped function is simply called directly.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import time
from typing import Any, Callable

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)

DEFAULT_TTL = 3600


def _make_key(func_name: str, args: tuple, kwargs: dict) -> str:
    """Build a unique, filesystem-safe cache filename from a call signature.

    Args:
        func_name: The name of the cached function.
        args: Positional arguments of the call.
        kwargs: Keyword arguments of the call.

    Returns:
        A filename of the form "<func_name>_<hash>.json".
    """
    raw = f"{func_name}:{args}:{kwargs}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{func_name}_{digest}.json"


def cached(ttl: int = DEFAULT_TTL) -> Callable:
    """Decorator that caches a function's JSON-serializable result on disk.

    Args:
        ttl: Time-to-live in seconds. A cached result older than this is
            ignored and the function is re-run.

    Returns:
        A decorator that wraps the target function with disk caching.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            os.makedirs(CACHE_DIR, exist_ok=True)
            path = os.path.join(CACHE_DIR, _make_key(func.__name__, args, kwargs))

            # Try to serve from a fresh cache file.
            if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning("Cache read failed for %s (%s); recomputing.", path, e)

            # Cache miss, stale, or unreadable: run the real function.
            result = func(*args, **kwargs)

            # Best-effort cache write; never let a write failure break the call.
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(result, f)
            except (OSError, TypeError) as e:
                logger.warning("Cache write failed for %s (%s); continuing.", path, e)

            return result

        return wrapper
    return decorator