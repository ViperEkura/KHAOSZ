"""GEMM plan configuration, probe, and launch hook.

The runtime autotuner lives in autotune.py; kernel/gemm.py exposes the raw
binding shapes used by benchmark tools.
"""

import os
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import torch

from astrai.extension.runtime.loader import get_module

# The autotune hook: the adapter calls note_launch on every launch, so the
# off path costs one flag check. Nothing is installed unless the deprecated
# ASTR_GEMM_AUTOTUNE=1 seed or enable() put a tuner there.
_autotuner: Optional[object] = None
_autotuner_resolved = False


def set_autotuner(tuner: object) -> None:
    """Install (or, with None, remove) the runtime autotune hook."""
    global _autotuner
    _autotuner = tuner


def note_launch(a, b, a_scale, b_scale, trans_a, trans_b, bias) -> None:
    """Record one quant_gemm launch for the installed tuner (the adapter's
    per-launch hook)."""
    global _autotuner_resolved
    if not _autotuner_resolved:
        _autotuner_resolved = True
        if os.environ.get("ASTR_GEMM_AUTOTUNE", "") == "1":  # deprecated seed
            enable()
    if _autotuner is not None:
        _autotuner.note(a, b, a_scale, b_scale, trans_a, trans_b, bias)


# Planner configuration — the runtime owner of every launch-time knob that
# used to live in environment variables (and, for the plan rows, in a
# hand-pasted plan_table.h block that needed a rebuild). Values set here
# win; anything unset falls to a one-time seed from the deprecated
# ASTR_GEMM_* variables, never to a per-call env read. Sweeps toggle these
# knobs between launches instead of rewriting the environment.
# ---------------------------------------------------------------------------

PLANNER_MODES = ("table", "hybrid", "model", "heuristic")
ROW_TIERS = ("override", "injected")

Rows = Union[str, Path]


# ---------------------------------------------------------------------------
# The plan surface: one value (``plan.config``), one writer
# (``plan.configure``), one scope (``plan.override``) — over the same bindings
# the legacy ``set_*`` / ``state`` / ``probe`` / ``facts`` functions call.
# Those keep returning the raw wire dict/list (the csrc/bench tools parse
# those keys); the records here are for new code and stay answerable to the
# old idioms — attribute access, ``d["key"]``, ``row[3]`` and 12-tuple
# unpacking all work — so a call site migrates one line at a time.
# ---------------------------------------------------------------------------


class _Record:
    """A frozen dataclass that still reads like the dict/list it replaced."""

    __slots__ = ()

    def _values(self) -> Tuple:
        return tuple(getattr(self, f.name) for f in fields(self))

    def __getitem__(self, key):
        return getattr(self, key) if isinstance(key, str) else self._values()[key]

    def __iter__(self):
        return iter(self._values())

    def __len__(self) -> int:
        return len(fields(self))

    def keys(self) -> Tuple:
        return tuple(f.name for f in fields(self))

    def items(self) -> Tuple:
        return tuple(zip(self.keys(), self._values()))

    def get(self, key, default=None):
        return getattr(self, key, default)


@dataclass(frozen=True)
class Staging(_Record):
    """The staging A/B switches; both default to enabled."""

    tma: bool
    mx: bool


@dataclass(frozen=True)
class PlanConfig(_Record):
    """The whole planner configuration as one value.

    ``planner_mode`` is the raw knob (-1 = unset: the one-time env seed
    decides) while ``planner`` is the resolved name. Each row tier carries the
    source spec it was installed from, which is what makes the value
    re-installable verbatim.
    """

    planner: str
    planner_mode: int
    log: bool
    table_off: bool
    override_rows: int
    override_source: str
    injected_rows: int
    injected_source: str
    staging: Staging

    @classmethod
    def _of(cls, wire: Dict) -> "PlanConfig":
        table = wire["table"]
        return cls(
            planner=wire["planner"],
            planner_mode=wire["planner_mode"],
            log=wire["log"],
            table_off=table["off"],
            override_rows=table["override_rows"],
            override_source=table["override_source"],
            injected_rows=table["injected_rows"],
            injected_source=table["injected_source"],
            staging=Staging(**wire["staging"]),
        )


@dataclass(frozen=True)
class Decision(_Record):
    """The dispatch decision for one problem: ``source`` names the row tier
    (or the model / heuristic) that answered."""

    source: str
    cta: int
    k_stages: int
    raster: int
    k_tile: int
    perf_class: int
    crosswise: int
    tma: bool = False
    resources: Tuple[Tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class DeviceFacts(_Record):
    """The device geometry the planner prices against."""

    sms: int
    smem_max: int
    smem_per_sm: int
    regs_per_sm: int
    l2_bytes: int
    cc: int


@dataclass(frozen=True)
class Tile(_Record):
    """One ladder recipe. The field order is the vocabulary row's, so
    ``row[3]`` is still the CTA class and a 12-tuple still unpacks."""

    crosswise: int
    ba: int
    bb: int
    cta: int
    k_stages: int
    k_tile: int
    bm: int
    bn: int
    wm: int
    wn: int
    threads: int
    smem: int

    @property
    def cta_name(self) -> str:
        return _tile_class_names()[self.cta]

    @property
    def name(self) -> str:
        return f"Tile_{self.bm}x{self.bn}x{self.k_tile}_W{self.wm}x{self.wn}_S{self.k_stages}"


_TILE_NAMES: Optional[Tuple[str, ...]] = None


def _tile_class_names() -> Tuple[str, ...]:
    """The CTA class names in ordinal order (static for the process)."""
    global _TILE_NAMES
    if _TILE_NAMES is None:
        _TILE_NAMES = tuple(get_module("gemm").tile_class_names())
    return _TILE_NAMES


# This module IS the plan surface — read it as one value, write it with one
# call, scope it for an A/B:
#
#     cfg = plan.config                      # the whole state
#     plan.configure(planner="model", tma=False)
#     with plan.override(rows=rows, tier="override"):
#         ...                                # restored on exit, always
#
# Every configure() argument defaults to "leave unchanged", and the value it
# returns is fully re-installable — feeding a saved config's fields back
# restores that exact state (both row tiers ride their source spec).


def configure(
    planner: Union[str, int, None] = None,
    log: Optional[bool] = None,
    tma: Optional[bool] = None,
    mx: Optional[bool] = None,
    table_off: Optional[bool] = None,
    rows: Optional[Rows] = None,
    tier: Optional[str] = None,
) -> PlanConfig:
    """Apply a patch; returns the resulting configuration.

    Every argument defaults to "leave it alone", and only the ones given ride
    the patch: the binding takes a dict keyed by these names, so an unset
    argument is an absent key rather than a second spelling of "no change".

    ``planner`` is a mode name, an int, or ``""`` to restore the unset state
    (the env seed decides). ``rows`` is a row-file path or inline row text;
    the empty string clears the tier named by ``tier`` (``"override"``, the
    default, or ``"injected"``).
    """
    if tier is not None and tier not in ROW_TIERS:
        raise ValueError(f"tier must be one of {ROW_TIERS}, got {tier!r}")
    if isinstance(planner, str) and planner and planner not in PLANNER_MODES:
        raise ValueError(f"planner must be one of {PLANNER_MODES}, got {planner!r}")
    patch = {
        "planner": planner,
        "log": log,
        "tma": tma,
        "mx": mx,
        "table_off": table_off,
        "rows": str(rows) if isinstance(rows, Path) else rows,
        "tier": tier,
    }
    wire = get_module("gemm").configure(
        {key: value for key, value in patch.items() if value is not None}
    )
    return PlanConfig._of(wire)


def _config() -> PlanConfig:
    return PlanConfig._of(get_module("gemm").config_state())


def _facts() -> DeviceFacts:
    return DeviceFacts(**get_module("gemm").device_facts_info())


@contextmanager
def override(**patch) -> Iterator[PlanConfig]:
    """Apply a patch for the duration of the block, then restore.

    The restore re-installs the saved value wholesale, so the knobs and both
    row tiers come back — whether the block returns or raises.
    """
    saved = _config()
    configure(**patch)
    try:
        yield _config()
    finally:
        configure(
            planner=saved.planner_mode,
            log=saved.log,
            tma=saved.staging.tma,
            mx=saved.staging.mx,
            table_off=saved.table_off,
            rows=saved.override_source,
            tier="override",
        )
        configure(rows=saved.injected_source, tier="injected")


def probe(
    m: int,
    n: int,
    k: int,
    dt_a: torch.dtype = torch.bfloat16,
    dt_b: torch.dtype = torch.bfloat16,
    trans_a: bool = False,
    trans_b: bool = True,
    batch: int = 1,
) -> Decision:
    """Inspect dispatch without launching; heuristic mode queries CUDA metadata."""
    info = get_module("gemm").plan_probe(
        m, n, k, dt_a, dt_b, trans_a=trans_a, trans_b=trans_b, batch=batch
    )
    info["resources"] = tuple(tuple(row) for row in info["resources"])
    return Decision(**info)


def tiles() -> List[Tile]:
    """Every recipe the launch ladders instantiate, each carrying its CTA
    class name and canonical ``Tile_...`` spelling."""
    return [Tile(*row) for row in get_module("gemm").tile_vocabulary()]


def __getattr__(name: str):  # PEP 562: the two read-only values of the surface
    if name == "config":
        return _config()
    if name == "facts":
        return _facts()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The tuning implementation owns candidate measurement and row persistence.
# These names remain on the plan surface for existing callers.
from astrai.extension.policy.gemm.autotune import (  # noqa: E402
    GemmAutotuner,
    Problem,
    Row,
    _problem_key,
    device_signature,
    enable,
)

__all__ = [
    "PlanConfig",
    "Staging",
    "Decision",
    "DeviceFacts",
    "Tile",
    "PLANNER_MODES",
    "ROW_TIERS",
    "Rows",
    "GemmAutotuner",
    "Problem",
    "Row",
    "device_signature",
    "_problem_key",
    "set_autotuner",
    "note_launch",
    "enable",
]
