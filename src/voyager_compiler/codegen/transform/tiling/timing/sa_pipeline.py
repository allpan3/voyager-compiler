"""Schedule the characterized SA-to-vector path and command retirement.

Capacities and forward delays describe synthesized stages. Loop completion,
output beats, FIFO occupancy and command waits are calculated here. This does
not simulate input arbitration or the array's internal elastic registers.
"""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

from .sa_output import OUTPUT_FIFO_VECTORS, REDUCTIONS

STAGE_NAMES = (
    "matrix_output",
    "router",
    "stage_0",
    "stage_1",
    "stage_2",
    "stage_3",
    "vector_output",
    "store",
)
TARGET_FIELDS = (
    "array_rows",
    "array_columns",
    "vector_lanes",
    "input_bits",
    "weight_bits",
    "accum_bits",
    "output_bits",
    "port_bits",
)


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


@dataclass(frozen=True)
class CommandTransfers:
    """DMA durations and reuse periods derived from the L3 loop nest.

    A zero period denotes an operand held for the whole sweep, loaded in the
    compiler prologue. Other operands load at multiples of their reuse period.
    Durations come from payload bytes, DRAM bandwidth and request latency.
    """

    loads: tuple
    store: float


class _Transfers:
    def __init__(self, spec, slots):
        self.spec = spec
        self.slots = slots
        self.index = 0
        self.now = self.dma = self.store_done = 0
        self.ready = [0] * len(spec.loads)
        for i, (cycles, period) in enumerate(spec.loads):
            if period == 0:
                self.dma += cycles
                self.ready[i] = self.dma

    def dispatch(self):
        for i, (cycles, period) in enumerate(self.spec.loads):
            if period and self.index % period == 0:
                self.dma = max(self.dma, self.now) + cycles
                self.ready[i] = self.dma
        # Input semaphores gate parameter dispatch, so L1 filling starts only
        # after these loads complete. Reusing an output slot also waits its DMA.
        return max(self.now, self.store_done, *self.ready)

    def retire(self, finished):
        # AsyncPipelinedKernel dispatches i, then retires and stores i-1.
        # Its next iteration cannot prefetch until that retirement. A single
        # command slot instead stores the current tile before advancing.
        if self.slots == 1 or self.index:
            self.now = max(self.now, finished[-self.slots])
            self.dma = max(self.dma, self.now) + self.spec.store
            self.store_done = self.dma
        self.index += 1

    def state(self, origin):
        return (
            tuple(self.index % p if p else 0 for _, p in self.spec.loads),
            (
                self.now - origin,
                *(
                    max(self.now, t) - origin
                    for t in (self.dma, self.store_done, *self.ready)
                ),
            ),
        )

    def advance(self, cycles, commands):
        self.now += cycles
        self.dma += cycles
        self.store_done += cycles
        self.ready = [t + cycles for t in self.ready]
        self.index += commands

    def finish(self, done):
        if self.slots == 1:
            return self.store_done
        return max(self.dma, self.now, done) + self.spec.store


@dataclass(frozen=True)
class PipelineEstimate:
    cycles: int
    producer_cycles: int
    nominal_cycles: int
    commands: int


class _Schedule:
    """Timing state at a loop boundary, shared by direct and vector routes."""

    def __init__(
        self, beats, slots, fill, skew, profile, vector, transfers=None
    ):
        self.stages = (
            ((OUTPUT_FIFO_VECTORS, 1), *profile.stages) if vector else ()
        )
        self.beats = beats
        self.finished = deque(maxlen=slots)
        capacity = (
            max(c for c, _ in self.stages) if vector else OUTPUT_FIFO_VECTORS
        )
        self.histories = [
            deque(maxlen=capacity) for _ in range(len(self.stages) + 1)
        ]
        self.producer = fill + skew
        self.gate = 0
        self.vector = vector
        self.transfers = _Transfers(transfers, slots) if transfers else None

    def emit(self):
        if not self.vector:
            dequeues = self.histories[0]
            if len(dequeues) == OUTPUT_FIFO_VECTORS:
                self.producer = max(self.producer, dequeues[0] + 1)
            dequeues.append(
                max(
                    self.producer + 1,
                    dequeues[-1] + self.beats if dequeues else 0,
                    self.gate,
                )
            )
            return
        time = self.producer
        for stage, (capacity, forward) in enumerate(self.stages):
            store = stage == len(self.stages) - 1
            interval = self.beats if store else 1
            if self.histories[stage]:
                time = max(time, self.histories[stage][-1] + interval)
            if len(self.histories[stage + 1]) >= capacity:
                time = max(
                    time,
                    self.histories[stage + 1][-capacity] + (0 if store else 1),
                )
            if stage == 2:
                # Router arithmetic waits for command start. Interface
                # registers before it can prefetch the next command's data.
                time = max(time, self.gate)
            self.histories[stage].append(time)
            if stage == 0:
                self.producer = time
            time += forward + (self.beats - 1 if store else 0)
        if self.histories[-1]:
            time = max(time, self.histories[-1][-1] + self.beats)
        self.histories[-1].append(time)
        self.gate += 1

    def state(self, commands=False):
        # All transfer constraints use timestamp differences. Retain every
        # history entry and command slot, but subtract a common time origin.
        origin = self.producer
        state = tuple(
            tuple(t - origin for t in history) for history in self.histories
        )
        if commands:
            return (
                state,
                tuple(t - origin for t in self.finished),
                self.transfers.state(origin) if self.transfers else None,
            )
        # Once this command's start gate falls behind the producer, it can
        # never constrain a later vector: producer advances by at least one
        # cycle per vector, as does the router gate (direct-store gate is fixed).
        return state, max(0, self.gate - origin)

    def advance(self, cycles, vectors, commands=False, count=0):
        self.producer += cycles
        for history in self.histories:
            for index in range(len(history)):
                history[index] += cycles
        if commands:
            for index in range(len(self.finished)):
                self.finished[index] += cycles
            if self.transfers:
                self.transfers.advance(cycles, count)
        elif self.vector:
            self.gate += vectors

    def repeat(self, count, body, vectors, commands=False, stride=1):
        """Skip only after an identical relative state recurs at this phase.

        The same loop body from the same relative state must produce the
        same next state and elapsed time. Repeating that transition advances
        all retained timestamps by its elapsed time; no average-rate or
        convergence tolerance is involved. Transients execute normally.
        """
        if count == 1:
            body()
            return
        seen = {}
        index = 0
        while index < count:
            if index % stride == 0:
                state = self.state(commands)
                prior = seen.get(state)
                if prior is not None:
                    old_index, old_time = prior
                    period = index - old_index
                    repeats = (count - index) // period
                    if repeats:
                        self.advance(
                            (self.producer - old_time) * repeats,
                            vectors * period * repeats,
                            commands,
                            period * repeats,
                        )
                        index += period * repeats
                        continue
                else:
                    seen[state] = index, self.producer
            body()
            index += 1

    def pattern(self, pattern):
        def body():
            self.producer += pattern.gap
            if pattern.child:
                self.pattern(pattern.child)
            else:
                self.emit()

        vectors = pattern.child.vectors if pattern.child else 1
        # Checking after each history window avoids copying the pipeline
        # state for every vector in a long uniform burst.
        stride = 1 if pattern.child else self.histories[0].maxlen
        self.repeat(pattern.repeat, body, vectors, stride=stride)


def _timing(
    work,
    arrivals,
    commands,
    slots,
    fill,
    skew,
    beats,
    profile,
    vector,
    transfers=None,
):
    state = _Schedule(beats, slots, fill, skew, profile, vector, transfers)
    # The measured gap ends at the next command's first output enqueue.
    # The output pattern counts that first issue, so subtract its cycle here.
    restart = max(work - arrivals.cycles, profile.writeback_restart_cycles - 1)
    tail = 0 if vector else profile.direct_store_forward_cycles + beats - 1

    def command():
        # Approximate completion at the last store. The compiler retires the
        # oldest occupied command slot before filling a new input buffer.
        state.gate = state.finished[-1] + 1 if state.finished else 0
        ready = (
            state.finished[0] + fill + skew
            if len(state.finished) == slots
            else 0
        )
        if state.transfers:
            ready = max(ready, state.transfers.dispatch() + fill + skew)
        if state.finished:
            state.producer += restart
        state.producer = max(state.producer, ready)
        state.pattern(arrivals)
        state.finished.append(state.histories[-1][-1] + tail)
        if state.transfers:
            state.transfers.retire(state.finished)

    state.repeat(commands, command, arrivals.vectors, commands=True)
    return PipelineEstimate(
        (
            state.transfers.finish(state.finished[-1])
            if state.transfers
            else state.finished[-1]
        ),
        state.producer,
        commands * work,
        commands,
    )


@lru_cache(maxsize=256)
def pipeline_timing(
    work, arrivals, commands, slots, fill, skew, beats, profile, transfers=None
):
    """Time the vector pipeline, batching only exact repeated timing states."""
    return _timing(
        work,
        arrivals,
        commands,
        slots,
        fill,
        skew,
        beats,
        profile,
        True,
        transfers,
    )


@lru_cache(maxsize=256)
def direct_timing(
    work, arrivals, commands, slots, fill, skew, beats, profile, transfers=None
):
    """Time direct stores and their FIFO, including command retirement."""
    return _timing(
        work,
        arrivals,
        commands,
        slots,
        fill,
        skew,
        beats,
        profile,
        False,
        transfers,
    )
