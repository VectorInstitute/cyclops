"""File which contains generic registery code
"""


def register_with_dictionary(register_dict, func=None, *, name=None):
    def wrap(func):
        register_dict[func.__name__ if name is None else name] = func
        return func

    # called with params
    if func is None:
        return wrap

    return wrap(func)
