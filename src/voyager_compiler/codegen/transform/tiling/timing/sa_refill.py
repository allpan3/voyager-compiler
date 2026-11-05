"""Local-buffer dependencies for dense SA commands.

The memory planner separates pipeline slots into whole-bank regions. A held
operand occupies region zero. Reads and stores reserve service in those
regions; an input refill waits for its local ping-pong buffer to be released.
Array propagation remains the architectural skew approximation used by the
output scheduler, rather than a simulation of every processing element.
"""

from bisect import bisect_right
from dataclasses import dataclass
from functools import lru_cache

from .sa_pipeline import OutputPattern, PipelineEstimate, _Schedule


@dataclass(frozen=True)
class InputTile:
    reads: tuple
    work: int
    outputs: OutputPattern | None
    weight_rows: int = 0


class _Banks:
    """Nonoverlapping service intervals, retaining gaps usable by prefetch.

    A store can be scheduled ahead of a later input-buffer release. Keeping
    its interval instead of only a busy-until timestamp lets that read use
    idle time before the store. Reservation order approximates requester
    arbitration; it does not add a workload-dependent arbitration penalty.
    """

    def __init__(self):
        self.spans = {}
        self.independent = set()

    def reserve(self, bank, start, count):
        if bank[0] in self.independent:
            # This role has its own bank region and its request port already
            # serializes these beats. No other stream can fill a gap here.
            return start, start + count - 1
        spans = self.spans.setdefault(bank, [])
        remaining, time, first = count, start, None
        added = []
        begin = max(0, bisect_right(spans, (start, float("inf"))) - 1)
        stop = begin
        for index in range(begin, len(spans)):
            left, right = spans[index]
            stop = index + 1
            if right <= time:
                continue
            if left > time:
                length = min(remaining, left - time)
                if first is None:
                    first = time
                added.append((time, time + length))
                time += length
                remaining -= length
                if not remaining:
                    break
            time = max(time, right)
        if remaining:
            if first is None:
                first = time
            added.append((time, time + remaining))
            time += remaining
        merged = []
        for left, right in sorted(spans[begin:stop] + added):
            if merged and left <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
            else:
                merged.append((left, right))
        spans[begin:stop] = merged
        return first, time - 1

    def discard_before(self, floors):
        # All subsequent refill releases and stores are at or after this
        # boundary. Past reservations cannot constrain another request.
        for bank, spans in self.spans.items():
            time = floors[bank[0]]
            self.spans[bank] = [
                (max(left, time), right)
                for left, right in spans
                if right > time
            ]

    def state(self, origin):
        return tuple(
            (bank, tuple((a - origin, b - origin) for a, b in spans))
            for bank, spans in sorted(self.spans.items())
            if spans
        )

    def advance(self, cycles):
        for bank, spans in self.spans.items():
            self.spans[bank] = [(a + cycles, b + cycles) for a, b in spans]


class _RefillSchedule(_Schedule):
    def __init__(self, beats, slots, skew, profile, vector, transfers):
        super().__init__(beats, slots, 0, skew, profile, vector, transfers)
        self.skew = skew
        self.profile = profile
        self.banks = _Banks()
        self.output_bank = None
        self.shared_output = False
        self.store_end = 0
        self.ports = [0, 0]
        self.ready = [[0, 0], [0, 0]]
        self.local_free = [[0, 0], [0, 0]]
        self.groups = ()
        self.periods = ()
        self.command_index = 0
        self.held_weights = False
        self.output_only = False

    def pattern(self, pattern):
        previous = self.output_only
        self.output_only = not self.shared_output
        try:
            super().pattern(pattern)
        finally:
            self.output_only = previous

    def emit(self):
        if not self.shared_output:
            super().emit()
            return
        if not self.vector:
            dequeues = self.histories[0]
            if len(dequeues) == dequeues.maxlen:
                self.producer = max(self.producer, dequeues[0] + 1)
            forward = self.profile.direct_store_forward_cycles
            earliest = max(
                self.producer + 1,
                self.store_end + 1 - forward,
                self.gate,
            )
            first, self.store_end = self.banks.reserve(
                self.output_bank, earliest + forward, self.beats
            )
            dequeues.append(max(earliest, first - forward))
            return
        time = self.producer
        for stage, (capacity, forward) in enumerate(self.stages):
            store = stage == len(self.stages) - 1
            if self.histories[stage]:
                time = max(
                    time,
                    self.histories[stage][-1] + (self.beats if store else 1),
                )
            if len(self.histories[stage + 1]) >= capacity:
                time = max(
                    time,
                    self.histories[stage + 1][-capacity] + (0 if store else 1),
                )
            if stage == 2:
                time = max(time, self.gate)
            if store:
                time = max(time, self.store_end + 1 - forward)
                first, last = self.banks.reserve(
                    self.output_bank, time + forward, self.beats
                )
                time = max(time, first - forward)
                self.store_end = last
            self.histories[stage].append(time)
            if stage == 0:
                self.producer = time
            time = last if store else time + forward
        self.histories[-1].append(time)
        self.gate += 1

    def state(self, commands=False):
        if self.output_only:
            return super().state(commands)
        origin = self.producer
        # A future read must follow both this requester's last beat and the
        # release of its target local slot. A store follows the previous
        # store. These lower bounds also hold across command boundaries.
        floors = {}
        floor = origin - self.skew
        ports = list(self.ports)
        free = [list(role) for role in self.local_free]
        if self.held_weights and not commands:
            # No more weight fills in this command. The held slot's release
            # is still ahead, at its final read, so older port times cannot
            # constrain another request before this source boundary.
            ports[1] = max(ports[1], floor)
            free[1] = [max(t, floor) for t in free[1]]
        gates = [max(port, min(role)) for port, role in zip(ports, free)]
        for group, floor in zip(self.groups, (*gates, self.store_end + 1)):
            floors[group] = min(floors.get(group, floor), floor)
        self.banks.discard_before(floors)
        return (
            super().state(commands),
            self.banks.state(origin),
            tuple(
                self.command_index % (2 * p) if p else 0 for p in self.periods
            )
            if commands
            else (),
            max(origin, self.store_end) - origin,
            tuple(t - origin for t in gates),
            tuple(
                tuple(max(ports[i], t) - origin for t in role)
                for i, role in enumerate(free)
            ),
            ()
            if commands
            else tuple(
                tuple(max(origin - self.skew, t) - origin for t in role)
                for role in self.ready
            ),
        )

    def advance(self, cycles, vectors, commands=False, count=0):
        super().advance(cycles, vectors, commands, count)
        if self.output_only:
            return
        self.banks.advance(cycles)
        if commands:
            self.command_index += count
        self.store_end += cycles
        self.ports = [t + cycles for t in self.ports]
        self.ready = [[t + cycles for t in role] for role in self.ready]
        self.local_free = [
            [t + cycles for t in role] for role in self.local_free
        ]


def _period(values):
    """Period of a sequence, including a possible unfinished last repeat."""
    prefix = [0] * len(values)
    for index in range(1, len(values)):
        match = prefix[index - 1]
        while match and values[index] != values[match]:
            match = prefix[match - 1]
        if values[index] == values[match]:
            match += 1
        prefix[index] = match
    return len(values) - prefix[-1] if values else 1


@lru_cache(maxsize=256)
def refill_timing(
    tiles,
    commands,
    slots,
    groups,
    periods,
    skew,
    parameter_cycles,
    read_forward,
    read_latency,
    beats,
    profile,
    vector,
    transfers,
):
    """Schedule local ping-pong fills, output service and command/DMA waits.

    ``groups`` identifies whole-bank regions; ``periods`` gives the number
    of commands reusing each input, weight and output slot. Geometry, loops,
    payloads and ownership are supplied by the compiler, not by the profile.
    """
    state = _RefillSchedule(beats, slots, skew, profile, vector, transfers)
    state.groups = groups
    state.banks.independent = {g for g in groups if groups.count(g) == 1}
    state.periods = tuple(
        p if p < commands else 0
        for g, p in zip(groups, periods)
        if slots > 1 and groups.count(g) > 1
    )
    # Restart spans output enqueue to output enqueue; the pattern adds the
    # first issue cycle. This repeated gate may overlap input/DMA readiness.
    restart = profile.writeback_restart_cycles - 1
    count = len(tiles)

    def bank(role, command):
        period = periods[role]
        slot = command // period % slots if commands > period else 0
        return groups[role], slot

    weight_tiles = [i for i, tile in enumerate(tiles) if tile.reads[1]]
    period = _period(tiles[1:])

    def command_body():
        command = state.command_index
        state.output_bank = bank(2, command)
        state.shared_output = any(
            state.output_bank == bank(role, command) for role in (0, 1)
        )
        dispatch = state.finished[0] if len(state.finished) == slots else 0
        if state.transfers:
            dispatch = max(dispatch, state.transfers.dispatch())
        state.ready = [[0, 0], [0, 0]]
        weight_group = 0

        def load(role, index, release):
            tile_index = weight_tiles[index] if role else index
            words = tiles[tile_index].reads[role]
            slot = index % 2
            _, last = state.banks.reserve(
                bank(role, command),
                max(release, state.ports[role], state.local_free[role][slot]),
                words,
            )
            state.ports[role] = last + 1
            state.ready[role][slot] = last + 1 + read_forward[role]

        # Omit short controller request startup; refill and resource waits
        # still determine when each operand becomes available.
        for index in range(min(2, count)):
            for role in (0, 1):
                if role == 1 and index >= len(weight_tiles):
                    continue
                load(
                    role,
                    index,
                    dispatch + parameter_cycles,
                )
        state.producer = max(
            state.producer + (restart if command else 0),
            max(role[0] for role in state.ready) + skew,
        )
        state.gate = state.finished[-1] + 1 if state.finished else 0
        index = 0
        seen = {}
        # Leave the final prefetch and weight-release transitions explicit.
        limit = min(
            count - 2, weight_tiles[-2] if len(weight_tiles) > 2 else count - 2
        )
        while index < count:
            tile = tiles[index]
            state.held_weights = len(weight_tiles) == 1 and index + 1 < count
            if (
                weight_group + 1 < len(weight_tiles)
                and weight_tiles[weight_group + 1] == index
            ):
                weight_group += 1
            if 1 <= index < limit and (index - 1) % period == 0:
                key = (index % 2, weight_group % 2, state.state())
                prior = seen.get(key)
                if prior is not None:
                    old_index, old_weight, old_time = prior
                    width = index - old_index
                    repeats = (limit - index) // width
                    if repeats:
                        vectors = sum(
                            t.outputs.vectors
                            for t in tiles[old_index:index]
                            if t.outputs
                        )
                        state.advance(
                            (state.producer - old_time) * repeats,
                            vectors * repeats,
                        )
                        weight_group += (weight_group - old_weight) * repeats
                        index += width * repeats
                        seen.clear()
                        continue
                else:
                    seen[key] = index, weight_group, state.producer
            state.producer = max(
                state.producer,
                state.ready[0][index % 2] + skew,
                state.ready[1][weight_group % 2] + skew,
            )
            source_start = state.producer - skew
            last_weight_use = index + 1 == count or tiles[index + 1].reads[1]
            if last_weight_use:
                # WeightController keeps this local tile across its reuse
                # loops. Free it only after the final row stream using it.
                weight = tiles[weight_tiles[weight_group]]
                rows = weight.weight_rows or weight.reads[1]
                release = source_start + rows - read_latency[1]
                state.local_free[1][weight_group % 2] = release
                if weight_group + 2 < len(weight_tiles):
                    load(1, weight_group + 2, release)
            if tile.outputs is not None:
                state.pattern(tile.outputs)
            else:
                state.producer += tile.work
            release = state.producer - skew - read_latency[0] + 1
            state.local_free[0][index % 2] = release
            if index + 2 < count:
                load(0, index + 2, release)
            index += 1
        # Approximate command completion at its final accepted store beat.
        if not state.shared_output:
            tail = (
                0 if vector else profile.direct_store_forward_cycles + beats - 1
            )
            state.store_end = state.histories[-1][-1] + tail
        done = state.store_end
        state.finished.append(done)
        if state.transfers:
            state.transfers.retire(state.finished)
        if count == 1:
            state.local_free[0][1] = state.local_free[0][0]
        if len(weight_tiles) == 1:
            state.local_free[1][1] = state.local_free[1][0]
        state.command_index += 1

    state.repeat(
        commands,
        command_body,
        sum(t.outputs.vectors for t in tiles if t.outputs),
        commands=True,
    )
    return PipelineEstimate(
        state.transfers.finish(state.finished[-1])
        if state.transfers
        else state.finished[-1],
        state.producer,
        commands * sum(tile.work for tile in tiles),
        commands,
    )
