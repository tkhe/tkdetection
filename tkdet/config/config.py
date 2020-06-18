import functools
import inspect
import logging

from fvcore.common.config import CfgNode
from fvcore.common.file_io import PathManager

__all__ = ["get_cfg", "configurable"]


def get_cfg() -> CfgNode:
    from .defaults import _C

    return _C.clone()


def configurable(init_func):
    assert init_func.__name__ == "__init__", "@configurable should only be used for __init__!"
    
    if init_func.__module__.startswith("tkdet."):
        assert init_func.__doc__ is not None and "experimental" in init_func.__doc__, \
            f"configurable {init_func} should be marked experimental"

    @functools.wraps(init_func)
    def wrapped(self, *args, **kwargs):
        try:
            from_config_func = type(self).from_config
        except AttributeError:
            raise AttributeError("Class with @configurable must have a 'from_config' classmethod.")
        if not inspect.ismethod(from_config_func):
            raise TypeError("Class with @configurable must have a 'from_config' classmethod.")

        if _called_with_cfg(*args, **kwargs):
            explicit_args = _get_args_from_config(from_config_func, *args, **kwargs)
            init_func(self, **explicit_args)
        else:
            init_func(self, *args, **kwargs)

    return wrapped


def _get_args_from_config(from_config_func, *args, **kwargs):
    signature = inspect.signature(from_config_func)
    if list(signature.parameters.keys())[0] != "cfg":
        raise TypeError(
            f"{from_config_func.__self__}.from_config must take 'cfg' as the first argument!"
        )
    support_var_arg = any(
        param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]
        for param in signature.parameters.values()
    )
    if support_var_arg:
        ret = from_config_func(*args, **kwargs)
    else:
        supported_arg_names = set(signature.parameters.keys())
        extra_kwargs = {}
        for name in list(kwargs.keys()):
            if name not in supported_arg_names:
                extra_kwargs[name] = kwargs.pop(name)
        ret = from_config_func(*args, **kwargs)
        ret.update(extra_kwargs)
    return ret


def _called_with_cfg(*args, **kwargs):
    if len(args) and isinstance(args[0], CfgNode):
        return True
    if isinstance(kwargs.pop("cfg", None), CfgNode):
        return True
    return False
