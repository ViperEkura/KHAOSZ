"""Base factory class for extensible component registration."""

from abc import ABC
from typing import Callable, Dict, Generic, Type, TypeVar

T = TypeVar("T")


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

    _registry: Dict[str, Type[T]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[T]], Type[T]]:
        """Decorator to register a component class.

        Args:
            name: Registration name for the component

        Returns:
            Decorator function that registers the component class

        Raises:
            TypeError: If the decorated class doesn't inherit from the base type
        """

        def decorator(component_cls: Type[T]) -> Type[T]:
            cls._validate_component(component_cls)
            cls._registry[name] = component_cls
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
        if name not in cls._registry:
            raise ValueError(
                f"Unknown component: '{name}'. "
                f"Supported types: {sorted(cls._registry.keys())}"
            )
        component_cls = cls._registry[name]
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
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a component name is registered.

        Args:
            name: Component name to check

        Returns:
            True if registered, False otherwise
        """
        return name in cls._registry
