from typing import Dict
from typing import KeysView
from typing import Optional

__all__ = ["Registry"]


class Registry(object):
    def __init__(self, name):
        self._name = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        assert name not in self._obj_map, \
            f"An object named '{name}' was already registered in '{self._name}' registry!"

        self._obj_map[name] = obj

    def register(self, name: str = None, obj: object = None) -> Optional[object]:
        if obj is None:
            def deco(func_or_class):
                if name is None:
                    new_name = func_or_class.__name__
                    self._do_register(new_name, func_or_class)
                else:
                    self._do_register(name, func_or_class)
                return func_or_class

            return deco

        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                f"No object named '{name}' found in '{self._name}' registry!"
            )
        return ret

    def keys(self) -> KeysView[str]:
        return self._obj_map.keys()
