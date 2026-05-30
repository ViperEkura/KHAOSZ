import json
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Self, Union, get_type_hints


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
            elif isinstance(v, (dict, list, tuple)):
                try:
                    val = list(v) if isinstance(v, tuple) else v
                    json.dumps(val)
                    d[fld.name] = val
                except (TypeError, ValueError):
                    pass
            elif isinstance(v, BaseConfig):
                d[fld.name] = v.to_dict()
            elif hasattr(v, "__dataclass_fields__"):
                sub = {}
                for f in fields(v):
                    a = getattr(v, f.name)
                    sub[f.name] = list(a) if isinstance(a, tuple) else a
                d[fld.name] = sub
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
        if isinstance(value, dict) and issubclass(target_type, BaseConfig):
            return target_type.from_dict(value)
        raise TypeError

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> Self:
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_json(self, path: Union[str, Path]):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
