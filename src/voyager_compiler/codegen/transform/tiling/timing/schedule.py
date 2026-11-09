"""Shared finite-storage and command scheduling rules.

Backends supply ordered work, storage lifetimes and build-specific stage delays.
Repeated work skips only when its complete relative timing state repeats.
"""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from math import ceil

# MatrixProcessor and CIMProcessor both use an eight-vector Connections FIFO.
OUTPUT_FIFO_VECTORS = 8


@dataclass(frozen=True)
class CommandTransfers:
    """DMA durations and reuse periods derived from the L3 loop nest.

    A zero period denotes an operand held for the whole sweep, loaded in the
    compiler prologue. Other operands load at multiples of their reuse period.
    Durations come from payload bytes, DRAM bandwidth and request latency.
    """

    loads: tuple
    store: float

    @classmethod
    def from_loads(cls, loads, store, commands):
        return cls(
            tuple(
                (ceil(cycles), commands // count if count > 1 else 0)
                for cycles, count in loads
            ),
            ceil(store),
        )


class Transfers:
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


class _RepeatedSchedule:
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


class OutputSchedule(_RepeatedSchedule):
    """Timing state at a loop boundary, shared by direct and vector routes."""

    def __init__(
        self, beats, slots, skew, stages=(), direct_forward=0, transfers=None
    ):
        vector = bool(stages)
        self.stages = ((OUTPUT_FIFO_VECTORS, 1), *stages) if vector else ()
        self.beats = beats
        self.finished = deque(maxlen=slots)
        capacity = (
            max(c for c, _ in self.stages) if vector else OUTPUT_FIFO_VECTORS
        )
        self.histories = [
            deque(maxlen=capacity) for _ in range(len(self.stages) + 1)
        ]
        self.producer = skew
        self.direct_forward = direct_forward
        self.gate = 0
        self.vector = vector
        self.transfers = Transfers(transfers, slots) if transfers else None

    def store(self, start, forward):
        """Return the stage departure and final beat on a dedicated output bank."""
        return start, start + forward + self.beats - 1

    def emit(self):
        if not self.vector:
            dequeues = self.histories[0]
            if len(dequeues) == OUTPUT_FIFO_VECTORS:
                self.producer = max(self.producer, dequeues[0] + 1)
            start = max(
                self.producer + 1,
                dequeues[-1] + self.beats if dequeues else 0,
                self.gate,
            )
            start, _ = self.store(start, self.direct_forward)
            dequeues.append(start)
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
            if store:
                time, finish = self.store(time, forward)
            self.histories[stage].append(time)
            if stage == 0:
                self.producer = time
            time = finish if store else time + forward
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


class _CommandSchedule(_RepeatedSchedule):
    def __init__(self, transfers, slots):
        self.producer = 0
        self.finished = deque(maxlen=slots)
        self.transfers = Transfers(transfers, slots)

    def state(self, commands=False):
        return (
            tuple(t - self.producer for t in self.finished),
            self.transfers.state(self.producer),
        )

    def advance(self, cycles, vectors, commands=False, count=0):
        self.producer += cycles
        self.finished = deque(
            (t + cycles for t in self.finished), maxlen=self.finished.maxlen
        )
        self.transfers.advance(cycles, count)


@lru_cache(maxsize=256)
def command_timing(command_cycles, commands, slots, transfers):
    """Apply compiler dispatch and retirement to an architectural command cost.

    Local execution is represented by its command duration. DMA and command
    slot dependencies use the same rules as the characterized stage schedule.
    This does not infer overlap between uncharacterized matrix commands.
    """
    if transfers is None:
        return commands * command_cycles
    if all(period in (0, 1) for _, period in transfers.loads):
        # Prologue, first prefetch, repeated load/store overlap, and epilogue
        # of the same compiler sequence. No state walk is needed here.
        held = sum(c for c, p in transfers.loads if p == 0)
        load = sum(c for c, p in transfers.loads if p == 1)
        store = transfers.store
        if slots == 1 or commands == 1:
            return held + commands * (load + command_cycles + store)
        return (
            held
            + load
            + max(command_cycles, load)
            + (commands - 2) * max(command_cycles, load + store)
            + max(command_cycles, store)
            + store
        )
    state = _CommandSchedule(transfers, slots)

    def command():
        state.producer = (
            max(state.producer, state.transfers.dispatch()) + command_cycles
        )
        state.finished.append(state.producer)
        state.transfers.retire(state.finished)

    state.repeat(commands, command, 0, commands=True)
    return state.transfers.finish(state.producer)
