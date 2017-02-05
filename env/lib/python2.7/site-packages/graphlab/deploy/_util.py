import warnings

def deprecated(msg = None):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    def innerFunc(func):
        def newFunc(*args, **kwargs):
            message = msg
            if not message:
                message = "Function %s is going to be deprecated." % func.__name__
            warnings.warn(message, category=DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        newFunc.__name__ = func.__name__
        newFunc.__doc__ = func.__doc__
        newFunc.__dict__.update(func.__dict__)
        return newFunc

    return innerFunc
