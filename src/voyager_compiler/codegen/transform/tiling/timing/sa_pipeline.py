"""SA build profile and compact final-output loop descriptions.

The shared scheduler consumes these stages and output events. Geometry and
loop bounds determine work; the profile supplies synthesized stage timing.
"""

from dataclasses import dataclass
from functools import lru_cache

from .output import STAGE_NAMES, TARGET_FIELDS
from .sa_output import REDUCTIONS


@dataclass(frozen=True)
class SAOutputPipeline:
    target: tuple
    stages: tuple
    writeback_restart_cycles: int
    direct_store_forward_cycles: int

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {
            "target",
            "stages",
            "writeback_restart_cycles",
            "direct_store_forward_cycles",
        }:
            raise ValueError(
                "SA output pipeline requires target, stages, writeback and store delays"
            )
        target = value["target"]
        if not isinstance(target, dict) or set(target) != set(TARGET_FIELDS):
            raise ValueError(
                "SA output pipeline target must specify geometry and widths"
            )
        stages = value["stages"]
        if not isinstance(stages, list) or len(stages) != len(STAGE_NAMES):
            raise ValueError(
                "SA output pipeline requires the eight ordered hardware stages"
            )
        records = []
        for name, stage in zip(STAGE_NAMES, stages):
            if (
                not isinstance(stage, dict)
                or set(stage) != {"name", "capacity_vectors", "forward_cycles"}
                or stage["name"] != name
            ):
                raise ValueError(f"Expected SA output stage {name}")
            records.append((stage["capacity_vectors"], stage["forward_cycles"]))
        positive = [*target.values(), *(n for pair in records for n in pair)]
        if any(type(n) is not int or n <= 0 for n in positive):
            raise ValueError(
                "SA geometry, capacities and delays must be positive integers"
            )
        restart = value["writeback_restart_cycles"]
        if type(restart) is not int or restart < 1:
            raise ValueError("SA writeback restart must be a positive integer")
        forward = value["direct_store_forward_cycles"]
        if type(forward) is not int or forward < 0:
            raise ValueError(
                "SA direct-store delay must be a nonnegative integer"
            )
        return cls(
            tuple(target[k] for k in TARGET_FIELDS),
            tuple(records),
            restart,
            forward,
        )


@dataclass(frozen=True)
class OutputPattern:
    """A repeated loop body, without expanding its output vectors.

    Each of ``repeat`` iterations waits ``gap`` cycles, then either emits
    one vector (a leaf) or runs ``child``. Gaps include reduction work that
    does not produce a final vector.
    """

    repeat: int = 1
    gap: int = 1
    child: "OutputPattern | None" = None

    @property
    def vectors(self):
        return self.repeat * (self.child.vectors if self.child else 1)

    @property
    def cycles(self):
        return self.repeat * (
            self.gap + (self.child.cycles if self.child else 0)
        )


@dataclass(frozen=True)
class PipelineEstimate:
    cycles: int
    producer_cycles: int
    nominal_cycles: int
    commands: int


def _delay(pattern, cycles):
    if pattern.repeat == 1:
        return OutputPattern(1, pattern.gap + cycles, pattern.child)
    return OutputPattern(1, cycles, pattern) if cycles else pattern


def _repeat_pattern(pattern, count):
    if count == 1:
        return pattern
    return OutputPattern(pattern.repeat * count, pattern.gap, pattern.child)


@lru_cache(maxsize=256)
def output_events(levels, array_rows):
    """Return nominal work and a compact final-output loop schedule."""
    spatial, prefix = 1, 0
    for name, bound in levels[0]:
        if name not in ("OX", "OY"):
            break
        spatial *= bound
        prefix += 1
    inner, outer = map(dict, levels)
    outer_names = [name for name, _ in levels[1]]
    held = [
        name
        for name in outer_names[: outer_names.index("OC")]
        if name in ("OX", "OY")
    ]
    if (
        held
        and spatial == inner["OX"] * inner["OY"]
        and all(inner[name] == 1 for name in (*REDUCTIONS, "OC"))
        and all(outer[name] == 1 for name in REDUCTIONS)
    ):
        # MatrixProcessor retains weights across these outer spatial loops.
        total = spatial * outer["OX"] * outer["OY"] * outer["OC"]
        for name in held:
            spatial *= outer[name]
        work = max(array_rows, spatial)
        pattern = _delay(OutputPattern(spatial), work - spatial)
        return total // spatial * work, _repeat_pattern(
            pattern, total // spatial
        )
    work, pattern = 1, OutputPattern()
    for level, loops in enumerate(levels):
        for index, (name, bound) in enumerate(loops):
            if level == 0 and index == prefix:
                gap = max(0, array_rows - spatial)
                pattern = _delay(pattern, gap)
                work += gap
            if name in REDUCTIONS:
                pattern = _delay(pattern, (bound - 1) * work)
            else:
                pattern = _repeat_pattern(pattern, bound)
            work *= bound
    return work, pattern
