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
    raw = f"{func_name}:{args}:{kwargs}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{func_name}_{digest}.json"


def cached(ttl: int = DEFAULT_TTL) -> Callable:
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            os.makedirs(CACHE_DIR, exist_ok=True)
            path = os.path.join(CACHE_DIR, _make_key(func.__name__, args, kwargs))

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