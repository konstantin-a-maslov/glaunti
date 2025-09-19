def cache(use_cache=True):
    def decorator(retrieve_func):
        if not use_cache:
            return retrieve_func
            
        cache_obj = {}
        
        def wrapper(*args):
            if args in cache_obj:
                return cache_obj[args]
            output = retrieve_func(*args)
            cache_obj[args] = output
            return output
            
        wrapper.cache_obj = cache_obj
        return wrapper

    return decorator
