"""DTensor gather and local reshard policy for sharded Muon updates."""

from typing import Protocol

from torch import Tensor
from torch.distributed.tensor import DTensor, distribute_tensor


class DTensorTransferPolicy(Protocol):
    """Policy for gathering a sharded matrix and restoring its local layout."""

    def gather(self, value: DTensor) -> Tensor: ...

    def redistribute(self, full_tensor: Tensor, like: DTensor) -> DTensor: ...


class FullTensorTransferPolicy:
    """Gather the full value, then distribute the result onto the same shards."""

    def gather(self, value: DTensor) -> Tensor:
        return value.full_tensor()

    def redistribute(self, full_tensor: Tensor, like: DTensor) -> DTensor:
        return distribute_tensor(
            full_tensor, like.device_mesh, like.placements, src_data_rank=None
        )
