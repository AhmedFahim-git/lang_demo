def set_docstring(docstring):
    def decorator(func):
        func.__doc__ = docstring
        return func

    return decorator
