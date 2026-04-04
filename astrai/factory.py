"""Base factory class for extensible component registration."""

from abc import ABC
from typing import Callable, Dict, Generic, Type, TypeVar, Optional, List, Tuple

T = TypeVar("T")


class Registry:
    """Flexible registry for component classes with category and priority support.

    This registry stores component classes with optional metadata (category, priority).
    It provides methods for registration, retrieval, and listing with filtering.
    """

    def __init__(self):
        self._entries = {}  # name -> (component_cls, category, priority)

    def register(
        self,
        name: str,
        component_cls: Type,
        category: Optional[str] = None,
        priority: int = 0,
    ) -> None:
        """Register a component class with optional category and priority."""
        if name in self._entries:
            raise ValueError(f"Component '{name}' is already registered")
        self._entries[name] = (component_cls, category, priority)

    def get(self, name: str) -> Type:
        """Get component class by name."""
        if name not in self._entries:
            raise KeyError(f"Component '{name}' not found in registry")
        return self._entries[name][0]

    def get_with_metadata(self, name: str) -> Tuple[Type, Optional[str], int]:
        """Get component class with its metadata."""
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"Component '{name}' not found in registry")
        return entry

    def contains(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._entries

    def list_names(self) -> List[str]:
        """Return list of registered component names."""
        return sorted(self._entries.keys())

    def list_by_category(self, category: str) -> List[str]:
        """Return names of components belonging to a specific category."""
        return sorted(
            name for name, (_, cat, _) in self._entries.items() if cat == category
        )

    def list_by_priority(self, reverse: bool = False) -> List[str]:
        """Return names sorted by priority (default ascending)."""
        return sorted(
            self._entries.keys(),
            key=lambda name: self._entries[name][2],
            reverse=reverse,
        )

    def entries(self) -> Dict[str, Tuple[Type, Optional[str], int]]:
        """Return raw entries dictionary."""
        return self._entries.copy()


class BaseFactory(ABC, Generic[T]):
    """Generic factory class for component registration and creation.

    This base class provides a decorator-based registration pattern
    for creating extensible component factories.

    Example usage:
        class MyFactory(BaseFactory[MyBaseClass]):
            pass

        @MyFactory.register("custom")
        class CustomComponent(MyBaseClass):
            ...

        component = MyFactory.create("custom", *args, **kwargs)
    """

    _registry: Registry

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry = Registry()

    @classmethod
    def register(
        cls, name: str, category: Optional[str] = None, priority: int = 0
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a component class with optional category and priority.

        Args:
            name: Registration name for the component
            category: Optional category for grouping components
            priority: Priority for ordering (default 0)

        Returns:
            Decorator function that registers the component class

        Raises:
            TypeError: If the decorated class doesn't inherit from the base type
        """

        def decorator(component_cls: Type[T]) -> Type[T]:
            cls._validate_component(component_cls)
            cls._registry.register(
                name, component_cls, category=category, priority=priority
            )
            return component_cls

        return decorator

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> T:
        """Create a component instance by name.

        Args:
            name: Registered name of the component
            *args: Positional arguments passed to component constructor
            **kwargs: Keyword arguments passed to component constructor

        Returns:
            Component instance

        Raises:
            ValueError: If the component name is not registered
        """
        if not cls._registry.contains(name):
            raise ValueError(
                f"Unknown component: '{name}'. "
                f"Supported types: {sorted(cls._registry.list_names())}"
            )
        component_cls = cls._registry.get(name)
        return component_cls(*args, **kwargs)

    @classmethod
    def _validate_component(cls, component_cls: Type[T]) -> None:
        """Validate that the component class is valid for this factory.

        Override this method in subclasses to add custom validation.

        Args:
            component_cls: Component class to validate

        Raises:
            TypeError: If the component class is invalid
        """
        pass

    @classmethod
    def list_registered(cls) -> list:
        """List all registered component names.

        Returns:
            List of registered component names
        """
        return cls._registry.list_names()

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a component name is registered.

        Args:
            name: Component name to check

        Returns:
            True if registered, False otherwise
        """
        return cls._registry.contains(name)

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """List registered component names in a category."""
        return cls._registry.list_by_category(category)

    @classmethod
    def list_by_priority(cls, reverse: bool = False) -> List[str]:
        """List registered component names sorted by priority."""
        return cls._registry.list_by_priority(reverse)
