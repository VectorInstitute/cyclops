"""Model catalog."""

from functools import partial


MODEL_CATALOG = {}


def register_with_dictionary(register_dict, func=None, *, name=None):
    def wrap(func):
        register_dict[func.__name__ if name is None else name] = func
        return func

    if func is None:
        return wrap

    return wrap(func)


def get_model(name):
    return MODEL_CATALOG[name]


register = partial(register_with_dictionary, MODEL_CATALOG)
