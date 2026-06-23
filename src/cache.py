import os
import json
import time
import hashlib
import functools


CACHE_DIR = "data/cache"
DEFAULT_TTL = 3600


def _make_key(func_name, args, kwargs):
    raw = f"{func_name}:{args}:{kwargs}"
    digest = hashlib.md5(raw.encode()).hexdigest()
    return f"{func_name}_{digest}.json"


def cached(ttl=DEFAULT_TTL):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            os.makedirs(CACHE_DIR, exist_ok=True)

            key = _make_key(func.__name__, args, kwargs)
            path = os.path.join(CACHE_DIR, key)

            if os.path.exists(path):
                age = time.time() - os.path.getmtime(path)
                if age < ttl:
                    with open(path, "r") as f:
                        return json.load(f)

            result = func(*args, **kwargs)
            with open(path, "w") as f:
                json.dump(result, f)
            return result

        return wrapper
    return decorator