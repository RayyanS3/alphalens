from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)
DEFAULT_TTL = 3600

def _make_key(func: Callable, args: tuple, kwargs: dict) -> str:
    """Build a unique cache filename from a normalized call signature.

    Arguments are bound to their parameter names and defaults applied, so
    positional and keyword forms of the same call produce the same key.
    """
    try:
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        signature = repr(sorted(bound.arguments.items()))
    except TypeError:
        signature = f"{args}:{kwargs}"

    raw = f"{func.__name__}:{signature}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{func.__name__}_{digest}.json"

def cached(ttl: int = DEFAULT_TTL) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            os.makedirs(CACHE_DIR, exist_ok=True)
            path = os.path.join(CACHE_DIR, _make_key(func, args, kwargs))
            
            if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        return json.load(f)
                except (OSError, json.JSONDecodeError) as e:
                    logger.warning("Cache read failed for %s (%s); recomputing.", path, e)

            result = func(*args, **kwargs)
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(result, f)
            except (OSError, TypeError) as e:
                logger.warning("Cache write failed for %s (%s); continuing.", path, e)

            return result
        return wrapper
    return decorator