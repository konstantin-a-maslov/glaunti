import functools


def cache(use_cache=True, maxsize=None, typed=False):
    def decorator(retrieve_func):
        if not use_cache:
            return retrieve_func
        return functools.lru_cache(maxsize=maxsize, typed=typed)(retrieve_func)
    return decorator
