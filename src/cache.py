# src/cache.py — a simple disk-based caching decorator

import os
import json
import time
import hashlib
import functools

CACHE_DIR = "data/cache"
DEFAULT_TTL = 3600   # cache lifetime in seconds (1 hour)


def _make_key(func_name, args, kwargs):
    """Build a unique, safe filename from the function name and its arguments."""
    raw = f"{func_name}:{args}:{kwargs}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{func_name}_{digest}.json"


def cached(ttl=DEFAULT_TTL):
    """Decorator: cache a function's JSON-serializable result on disk for `ttl` seconds."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(CACHE_DIR, exist_ok=True)

            key = _make_key(func.__name__, args, kwargs)
            path = os.path.join(CACHE_DIR, key)

            # If a fresh cache file exists, use it
            if os.path.exists(path):
                age = time.time() - os.path.getmtime(path)
                if age < ttl:
                    with open(path, "r") as f:
                        return json.load(f)

            # Otherwise run the real function and cache the result
            result = func(*args, **kwargs)
            with open(path, "w") as f:
                json.dump(result, f)
            return result

        return wrapper
    return decorator