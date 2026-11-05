"""Architectural SA final-output FIFO timing, composed in hardware loop order.

Only MatrixProcessor::accum_output_fifo is credited as final-output storage.
Intermediate partial-sum queues and uncharacterized downstream registers are
excluded. This is a deterministic queue estimate, not an RTL simulator.
"""

from dataclasses import dataclass
from functools import lru_cache

from .backpressure import StreamSummary, idle

OUTPUT_FIFO_VECTORS = 8  # MatrixProcessor.h: accum_output_fifo
REDUCTIONS = {"IC", "FX", "FY"}


@dataclass(frozen=True)
class OutputEstimate:
    nominal_cycles: int
    producer_cycles: int
    consumer_cycles: int
    stall_cycles: int
    drain_cycles: int
    capacity_vectors: int


@lru_cache(maxsize=4096)
def output_timing(levels, array_rows, cycles_per_vector):
    """Time an L1/L2 sweep, retaining queue contents across every loop.

    ``levels`` contains (loop name, bound) pairs inner-to-outer at each level.
    One nominal compute cycle completes one vector. A reduction emits only
    on its last iteration; other iterations allow earlier output to drain.
    Weight loading adds max(0, array_rows - inner_spatial_vectors) per tile.

    The FIFO starts empty. A vector can enter when a slot is available, and
    service starts after the vector becomes visible on the following cycle.
    A dequeue releases its slot for enqueue on the following cycle. These
    registered FIFO rules follow Connections::Fifo, not measured delays;
    service occupies the consumer for the derived cycles_per_vector. This
    architectural schedule omits additional pipeline storage/delays and
    variable downstream waits. It requires no timing-profile fields.
    """
    for value in (array_rows, cycles_per_vector):
        if type(value) is not int or value <= 0:
            raise ValueError(
                "SA rows and service cycles must be positive integers"
            )
    if len(levels) != 2:
        raise ValueError("SA output timing requires L1 and L2 loops")
    for loops in levels:
        if {name for name, _ in loops} != {
            "OX",
            "OY",
            "OC",
            "IC",
            "FX",
            "FY",
        } or len(loops) != 6:
            raise ValueError(
                "Each SA level must contain the six matrix loops once"
            )
        if any(type(n) is not int or n <= 0 for _, n in loops):
            raise ValueError("SA loop bounds must be positive integers")

    service = cycles_per_vector
    # Connections::Fifo registers head/tail/full: no empty bypass and no
    # full-slot reuse on the dequeue edge. Both rules follow its source.
    # For a full FIFO, the oldest dequeue is consumer_finish - 8*service.
    # If service has gaps, producer+1 already dominates that bound.
    # p' = max(p+1, d-8*s+1); d' = max(p'+1+s, d+s).
    summary = StreamSummary(
        1, (1, 1 - OUTPUT_FIFO_VECTORS * service, 2 + service, service)
    )
    spatial = 1
    prefix = 0
    for loop, bound in levels[0]:
        if loop not in ("OX", "OY"):
            break
        spatial *= bound
        prefix += 1
    for level, loops in enumerate(levels):
        for index, (loop, bound) in enumerate(loops):
            if level == 0 and index == prefix:
                summary = idle(max(0, array_rows - spatial)).then(summary)
            if bound != 1:
                if loop in REDUCTIONS:
                    summary = idle((bound - 1) * summary.work_cycles).then(
                        summary
                    )
                else:
                    summary = summary.repeated(bound)
    timing = summary.timing()
    return OutputEstimate(
        summary.work_cycles,
        timing.producer_cycles,
        timing.consumer_cycles,
        timing.stall_cycles,
        timing.backlog_cycles,
        OUTPUT_FIFO_VECTORS,
    )
