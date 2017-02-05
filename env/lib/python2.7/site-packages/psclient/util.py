from decorator import decorator
from time import time

from warnings import warn as _warn
def deprecated(message, allow=True):
    """Decorates a deprecated method or function with a handy warning
    message. Will log a warning exactly once."""
    warned = [False]

    @decorator
    def deprecated(func, *args, **kwargs):
        _warn(message, DeprecationWarning, stacklevel=2)
        if not warned[0]:
            _logger.warn(message)
            warned[0] = True
        if allow:
            return func(*args, **kwargs)
        else:
            raise DeprecationWarning(message)
    return deprecated
