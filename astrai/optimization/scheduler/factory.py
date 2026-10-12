"""Registry for training learning-rate schedulers."""

from astrai.factory import BaseFactory

from .base import BaseScheduler


class SchedulerFactory(BaseFactory[BaseScheduler]):
    """Factory for creating registered training schedulers."""
