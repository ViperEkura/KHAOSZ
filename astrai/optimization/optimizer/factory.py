"""Factory for registered training optimizers."""

from torch.optim import Optimizer

from astrai.factory import BaseFactory


class OptimizerFactory(BaseFactory[Optimizer]):
    """Factory for creating registered training optimizers."""
