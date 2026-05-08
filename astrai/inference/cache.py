"""KV cache slot allocation and prefix cache management.

Provides:
  - SlotAllocator: Object Pool pattern for O(1) KV cache slot alloc/free via bitmask.
  - PrefixCacheManager: Radix-tree prefix cache with LRU eviction for KV cache reuse.
"""

import time
from collections import OrderedDict
from typing import Dict, List, Tuple

from torch import Tensor

_STOP = object()


class _RadixNode:
    """Internal node for the radix tree prefix cache.

    Attributes:
        children: Mapping from token ID to child node.
        slot: KV cache slot index for the prefix ending at this node.
        slot_ver: Version counter of the slot at insertion time.
        ref_count: Number of tasks currently referencing this node.
        last_access: Timestamp of the most recent access (for LRU ordering).
    """

    __slots__ = ("children", "slot", "slot_ver", "ref_count", "last_access")

    def __init__(self):
        self.children: Dict[int, "_RadixNode"] = {}
        self.slot: int = -1
        self.slot_ver: int = 0
        self.ref_count: int = 0
        self.last_access: float = 0.0


class SlotAllocator:
    """KV cache slot allocator using bitmask for O(1) alloc/free.

    Implements the Object Pool pattern: pre-allocated KV cache slots
    are managed via a bitmask, providing constant-time allocation and
    deallocation with version counters for staleness detection.
    """

    def __init__(self, max_slots: int):
        self._max_slots = max_slots
        self._free_mask = (1 << max_slots) - 1
        self._versions: List[int] = [0] * max_slots

    def alloc(self) -> int:
        """Allocates a free slot.

        Returns:
            Slot index on success, -1 if all slots are occupied.
        """
        lsb = self._free_mask & -self._free_mask
        if lsb == 0:
            return -1
        idx = lsb.bit_length() - 1
        self._free_mask ^= lsb
        self._versions[idx] += 1
        return idx

    def free(self, idx: int) -> None:
        """Releases a slot back to the free pool."""
        self._free_mask |= 1 << idx

    def occupy(self, idx: int) -> None:
        """Marks a currently free slot as occupied without bumping its version.

        Used for direct slot reuse when a prefix-cached slot is still valid.
        """
        self._free_mask ^= 1 << idx

    def is_free(self, idx: int) -> bool:
        """Checks whether a slot is currently free."""
        return (self._free_mask >> idx) & 1 == 1

    def version(self, idx: int) -> int:
        """Returns the current version counter for a slot."""
        return self._versions[idx]

    @property
    def free_count(self) -> int:
        """Returns the number of currently free slots."""
        return self._free_mask.bit_count()


class PrefixCacheManager:
    """Radix-tree prefix cache with LRU eviction.

    Maps token ID sequences to KV cache slots. Intermediate tree nodes
    also store slot information, allowing direct slot reuse when the
    cached slot is free and its version matches (no intervening writes).
    """

    def __init__(self, max_capacity: int = 1000):
        """Initializes the prefix cache.

        Args:
            max_capacity: Maximum number of nodes in the LRU list.
        """
        self.root = _RadixNode()
        self.max_capacity = max_capacity
        self._lru: OrderedDict[int, _RadixNode] = OrderedDict()

    def insert(self, token_ids: Tuple[int, ...], slot: int, slot_ver: int) -> None:
        """Inserts a token sequence into the prefix cache.

        Every node along the path records the slot and its version,
        enabling direct slot reuse for partial prefix matches.

        Args:
            token_ids: The token ID sequence to cache.
            slot: The KV cache slot containing this prefix's computed keys/values.
            slot_ver: The slot version at insertion time, used for staleness detection.
        """
        node = self.root
        for tid in token_ids:
            nxt = node.children.get(tid)
            if nxt is None:
                nxt = _RadixNode()
                node.children[tid] = nxt
            node = nxt
            node.slot = slot
            node.slot_ver = slot_ver
            node.last_access = time.time()
            self._lru[id(node)] = node
        node.ref_count += 1
        self._evict_if_needed()

    def find(self, token_ids: List[int]) -> Tuple[int, int, int]:
        """Finds the longest matching prefix in the cache.

        Walks the radix tree token by token, recording the deepest match.

        Args:
            token_ids: The token sequence to match against.

        Returns:
            Tuple of (prefix_len, slot, slot_ver):
                prefix_len: Number of matching tokens (0 if no match).
                slot: KV cache slot of the matched prefix (-1 if no match).
                slot_ver: Version of that slot when the prefix was inserted.
        """
        node = self.root
        best_len, best_slot, best_ver = 0, -1, 0
        for i, tid in enumerate(token_ids):
            nxt = node.children.get(tid)
            if nxt is None:
                break
            node = nxt
            best_len, best_slot, best_ver = i + 1, node.slot, node.slot_ver
            node.last_access = time.time()
            self._lru.move_to_end(id(node))
        return best_len, best_slot, best_ver

    def pin(self, token_ids: Tuple[int, ...]) -> None:
        """Increments the reference count of a cached prefix.

        Called when a task reuses a cached prefix to prevent eviction.

        Args:
            token_ids: The token sequence whose node's ref_count to increment.
        """
        node = self.root
        for tid in token_ids:
            nxt = node.children.get(tid)
            if nxt is None:
                return
            node = nxt
        node.ref_count += 1

    def release(self, token_ids: Tuple[int, ...]) -> None:
        """Decrements the reference count of a cached prefix.

        The node's slot is preserved even when ref_count reaches zero,
        allowing future tasks to reuse the slot directly if it remains free.

        Args:
            token_ids: The token sequence whose node's ref_count to decrement.
        """
        node = self.root
        for tid in token_ids:
            nxt = node.children.get(tid)
            if nxt is None:
                return
            node = nxt
        if node.ref_count > 0:
            node.ref_count -= 1

    def copy_kv(
        self,
        token_ids: Tuple[int, ...],
        target_slot: int,
        kv_cache: Tuple[Tensor, Tensor],
        n_layers: int,
    ) -> None:
        """Copies cached KV data from the source slot to a target slot.

        Args:
            token_ids: The prefix token sequence identifying the source cache node.
            target_slot: The destination KV cache slot to copy into.
            kv_cache: Tuple of (k_cache, v_cache) tensors.
            n_layers: Number of transformer layers to copy.
        """
        node = self.root
        for tid in token_ids:
            nxt = node.children.get(tid)
            if nxt is None:
                return
            node = nxt
        src_slot = node.slot
        if src_slot < 0:
            return
        prefix_len = len(token_ids)
        k_cache, v_cache = kv_cache
        for li in range(n_layers):
            k_cache[target_slot, :prefix_len, li].copy_(
                k_cache[src_slot, :prefix_len, li]
            )
            v_cache[target_slot, :prefix_len, li].copy_(
                v_cache[src_slot, :prefix_len, li]
            )

    def _evict_if_needed(self) -> None:
        """Evicts least-recently-used nodes until under capacity.

        Skips nodes with ref_count > 0 (still in use by active tasks).
        Evicted nodes have their slot and children cleared.
        """
        while len(self._lru) > self.max_capacity:
            key, node = next(iter(self._lru.items()))
            if node.ref_count > 0:
                self._lru.move_to_end(key)
                continue
            self._lru.pop(key)
            node.slot = -1
            node.slot_ver = 0
            node.children.clear()
