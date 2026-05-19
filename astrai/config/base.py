import json
from dataclasses import MISSING, dataclass, fields
from typing import Any, Dict, Optional, Self, get_type_hints


@dataclass
class BaseConfig:
    def to_dict(self) -> Dict[str, Any]:
        d = {}
        for fld in fields(self):
            v = getattr(self, fld.name)
            if isinstance(v, (str, int, float, bool)):
                d[fld.name] = v
            elif v is None:
                d[fld.name] = None
            elif isinstance(v, (dict, list)):
                try:
                    json.dumps(v)
                    d[fld.name] = v
                except (TypeError, ValueError):
                    pass
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Self:
        hints = get_type_hints(cls)
        inst = cls.__new__(cls)
        for fld in fields(cls):
            if fld.name in d:
                v = d[fld.name]
                target = cls._unwrap_optional(hints.get(fld.name))
                if target is not None:
                    try:
                        v = cls._coerce(v, target)
                    except (TypeError, ValueError):
                        pass
                object.__setattr__(inst, fld.name, v)
            elif fld.default is not MISSING:
                object.__setattr__(inst, fld.name, fld.default)
            elif fld.default_factory is not MISSING:
                object.__setattr__(inst, fld.name, fld.default_factory())
            else:
                object.__setattr__(inst, fld.name, None)
        return inst

    @staticmethod
    def _unwrap_optional(tp) -> Optional[type]:
        if tp is None:
            return None
        origin = getattr(tp, "__origin__", None)
        if origin is not None:
            args = getattr(tp, "__args__", ())
            non_none = [a for a in args if a is not type(None)]
            return non_none[0] if non_none else None
        return tp

    @staticmethod
    def _coerce(value: Any, target_type: type) -> Any:
        if target_type is bool and isinstance(value, bool):
            return value
        if (
            target_type is int
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return int(value)
        if (
            target_type is float
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            return float(value)
        if target_type is str and isinstance(value, str):
            return value
        if isinstance(value, target_type):
            return value
        raise TypeError
