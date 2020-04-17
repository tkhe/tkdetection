import cloudpickle


class PicklableWrapper(object):
    def __init__(self, obj):
        self._obj = obj

    def __reduce__(self):
        s = cloudpickle.dumps(self._obj)
        return cloudpickle.loads, (s,)

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)

    def __getattr__(self, attr):
        if attr not in ["_obj"]:
            return getattr(self._obj, attr)
        return getattr(self, attr)
