from functools import wraps


class FeaturePlugin:
    """
    Utility to register a function that operates on an image.
    """

    _functions = dict()

    def __init__(self):
        pass

    @classmethod
    def register(cls, feature_name, cons = None): #, result_names):
        def inner(f):
            def wrapped(*args, **kwargs):
                value = f(*args, **kwargs)
                if cons is not None and value is not None:
                    return cons(*value)
                else:
                    return value

            # inner function
            func = wraps(f)(wrapped)

            # keep in registry
            FeaturePlugin._functions[feature_name] = func

            # return to decorator
            return func

        return inner

    @classmethod
    def get(cls, feature_name):

        if feature_name in FeaturePlugin._functions:
            return FeaturePlugin._functions[feature_name]
        else:
            return None