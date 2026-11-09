"""CIM banked-path schedule: eligibility, lower bounds and event timing.

The event schedule joins buffer ownership, word arbitration, the output
pipeline and compiler DMA. It skips only complete repeated states. Source
parameters define work and storage; measured build timing supplies pipeline
delays. The lower bounds live beside the schedule they must not exceed.
"""

import math
from collections import deque
from functools import lru_cache
from heapq import heappop, heappush
from math import lcm, prod

import interstellar
from voyager_compiler.codegen.transform.tiling.timing.buffers import (
    buffer_completion,
)
from voyager_compiler.codegen.transform.tiling.timing.schedule import (
    CommandTransfers,
    Transfers,
    command_timing,
)

le = interstellar.le
REDUCTIONS = ("IC", "FX", "FY")



def mapping_levels(mapping):
    """L1 and L2 bounds in inner-to-outer order, excluding the unit batch loop."""
    return tuple(
        tuple(
            (le.table[d], mapping.loop_blockings[d][level])
            for d in sorted(
                (d for d in range(le.NUM) if d != le.ON),
                key=lambda d: mapping.loop_orders[d][level],
            )
        )
        for level in (1, 2)
    )


def banked_profile(rc):
    """Use measured bank timing only for the characterized dense datapath."""
    profile = rc.timing_options.banked_pipeline
    if (
        profile is None
        or not rc.config.double_buffered_accum_buffer
        or rc.single_k_tail_extra_pass
        or rc.tail_specs
        or rc.bias_width
        or rc.input_scale_width
        or rc.weight_scale_width
        or rc.output_scale_width
        or (not rc.has_tail and rc.output_dtype_width != rc.accum_dtype_width)
    ):
        return None
    target = (
        *rc.config.pe_array_size,
        rc.config.vector_lanes,
        rc.input_dtype_width,
        rc.weight_dtype_width,
        rc.accum_dtype_width,
        profile.target[6],
        rc.sram_bandwidth,
    )
    if target != profile.target or (
        rc.has_tail and rc.output_dtype_width != profile.target[6]
    ):
        return None
    if rc.output_dtype_width * rc.config.pe_array_size[1] <= rc.sram_bandwidth:
        return None
    return profile


def banked_schedule(
    rc, mapping, groups, loads, store, commands, runtime_limit=math.inf
):
    """Join local buffer ownership and compiler DMA on one event timeline.

    Bounds relax dependencies and may reject only a strictly slower candidate.
    Unsupported routes retain their architectural estimate; they do not borrow
    storage or delays from this measured build.
    """
    profile = rc._banked_pipeline
    if (
        profile is None
        or not rc.uses_banked_output(mapping)
        or mapping.loop_blockings[le.IC][3] != 1
    ):
        return None
    timing = rc.timing(mapping, groups)
    if timing is None:
        return math.inf
    r, traffic = timing.readiness, timing.traffic
    transition = r["weight_ready_cycles"]
    assert transition == r["weight_release_cycles"]
    if (
        traffic is None
        or r.get("input_max_fill_bound")
        or r.get("input_bank_switch_cycles")
        or r.get("weight_bank_switch_cycles")
    ):
        return None
    if r.get("accumulation_feedback_safe") is False:
        return math.inf
    bound = (
        max(
            timing.compute_cycles,
            traffic.input_external_beats,
            traffic.weight_external_beats,
            traffic.output_vectors,
        )
        * commands
    )
    if bound > runtime_limit:
        return math.inf
    work = rc.evaluate(mapping)
    policy = work.policy
    levels = mapping_levels(mapping)
    input_tiles = prod(b for _, b in levels[1])
    vectors = prod(b for n, b in levels[0] if n not in REDUCTIONS)
    output_tiles = work.output_vectors // vectors
    args = dict(
        levels=levels,
        policy=(
            policy.sequence_sets if policy.fits else 1,
            policy.compute_replays if policy.fits else 1,
            policy.macs_per_set_use,
        ),
        capacity=rc.config.cim_weight_sets,
        input_reads=traffic.input_external_beats // input_tiles,
        input_fill=timing.resource_cycles["input"] // input_tiles,
        input_first=r["input_first_fill_cycles"],
        weight_reads=traffic.weight_external_beats // policy.full_set_loads,
        weight_fill=r["weight_fill_cycles"],
        weight_first=r["weight_first_fill_cycles"],
        interval=timing.resource_cycles["result_slots"] // work.mac_requests,
        latency=r["result_latency_cycles"] + profile.forward_cycles,
        beats=(
            timing.resource_cycles["output"]
            - output_tiles * r["output_bank_drain_cycles"]
        )
        // work.output_vectors,
        drain=r["output_bank_drain_cycles"],
        groups=tuple(
            next(i for i, g in enumerate(groups) if role in g)
            for role in ("input", "weight", "output")
        ),
        handoff=profile.producer_handoff_cycles,
        pipeline=profile.stages if rc.has_tail else (),
        readahead=(
            profile.input_readahead_beats,
            profile.weight_readahead_beats,
        ),
    )
    # Independent input/weight ports and ready operands are optimistic. The
    # last completed bank must still pass through the output port and DMA.
    first_dispatch = (
        0 if rc.dram_untimed else sum(math.ceil(c) for c, _ in loads)
    )
    final_store = 0 if rc.dram_untimed else math.ceil(store)
    last_output = vectors * args["beats"] + args["latency"] + args["drain"]
    compute_min = (
        work.mac_requests * args["interval"]
        + max(0, output_tiles - 1) * args["handoff"]
    )
    bounds = [bound, compute_min * commands]
    for finish, _, conservative in (
        buffer_completion(
            args["policy"][0],
            args["policy"][1],
            (policy.sequence_count if policy.fits else policy.full_set_loads)
            * commands,
            args["capacity"],
            args["weight_fill"],
            transition,
            policy.macs_per_set_use * args["interval"],
            args["input_first"],
            release_delay=transition,
            load_start=0,
        ),
        buffer_completion(
            1,
            1,
            input_tiles * commands,
            2,
            args["input_fill"],
            0,
            work.mac_requests // input_tiles * args["interval"],
            args["weight_first"] + transition,
            load_start=0,
        ),
    ):
        if not conservative:
            bounds.append(finish)
    bound = max(bounds) + last_output + first_dispatch + final_store
    if args["pipeline"]:
        # Vector storage can release the bank before its final memory write.
        bound -= args["drain"]
    else:
        finish, _, conservative = buffer_completion(
            1,
            1,
            output_tiles * commands,
            2,
            work.mac_requests // output_tiles * args["interval"],
            args["latency"],
            vectors * args["beats"],
            0,
            release_delay=args["drain"],
        )
        if not conservative:
            bound = max(
                bound, finish + args["drain"] + first_dispatch + final_store
            )
    slots = 2 if rc.double_buffered_l2 else 1
    transfers = (
        None
        if rc.dram_untimed
        else CommandTransfers.from_loads(loads, store, commands)
    )
    if transfers is not None:
        bound = max(
            bound,
            sum(math.ceil(c) * n for c, n in loads)
            + math.ceil(store) * commands,
            command_timing(compute_min, commands, slots, transfers),
        )
    if bound > runtime_limit:
        return math.inf
    args.update(
        commands=commands,
        slots=slots,
        periods=tuple(commands // count for _, count in loads) + (1,),
        transfers=transfers,
    )
    result = estimate(**args, cutoff=runtime_limit)
    assert bound <= result, "CIM lower bound exceeded the complete schedule"
    return result


def estimate(
    levels,
    policy,
    capacity,
    input_reads,
    input_fill,
    input_first,
    weight_reads,
    weight_fill,
    weight_first,
    interval,
    latency,
    beats,
    drain,
    groups,
    *,
    commands=1,
    slots=1,
    periods=(1, 1, 1),
    transfers=None,
    batch=True,
    pipeline=(),
    handoff=0,
    readahead=(0, 0),
    cutoff=float("inf"),
):
    run = policy[2]
    tile_steps = prod(b for n, b in levels[0]) // run
    output_steps = tile_steps * prod(b for n, b in levels[1] if n in REDUCTIONS)
    size = prod(b for level in levels for n, b in level) // run
    vectors = prod(b for n, b in levels[0] if n not in REDUCTIONS)
    # Only equality of bank membership affects arbitration.
    roles = tuple(dict.fromkeys(groups))
    groups = tuple(roles.index(g) for g in groups)
    return _estimate(
        run,
        size,
        tile_steps,
        output_steps,
        vectors,
        tuple(policy),
        capacity,
        input_reads,
        input_fill,
        input_first,
        weight_reads,
        weight_fill,
        weight_first,
        interval,
        latency,
        beats,
        drain,
        groups,
        batch,
        commands,
        slots,
        tuple(periods),
        transfers,
        tuple(pipeline),
        handoff,
        tuple(readahead),
        cutoff,
    )


@lru_cache(maxsize=8192)
def _estimate(
    run,
    size,
    tile_steps,
    output_steps,
    final_vectors,
    policy,
    capacity,
    input_reads,
    input_fill,
    input_first,
    weight_reads,
    weight_fill,
    weight_first,
    interval,
    latency,
    beats,
    drain,
    groups,
    batch,
    commands,
    slots,
    periods,
    transfers,
    pipeline,
    handoff,
    readahead,
    cutoff,
):
    command_size = size
    queues = [deque() for _ in pipeline]
    produced = consumed = routed = 0
    source_ready = 0
    vector_gate = 0
    stage_seen = {}
    size *= commands
    sets, replays, _ = policy
    input_tiles, weight_loads, output_tiles = (
        size // tile_steps,
        size // replays,
        size // output_steps,
    )

    def coordinates(index):
        return (
            index // tile_steps,
            index // (sets * replays) * sets + index % sets,
            index // output_steps,
        )

    repeat = lcm(tile_steps, output_steps, policy[0])
    # Leave command tails and prefetch transitions explicit.
    limit = max(
        0,
        size
        - max(2 * tile_steps, 2 * output_steps, capacity * policy[0])
        - repeat,
    )

    now = 0
    # Roles are input, weight and output; a job holds words left, tile index,
    # and earliest controller completion. Events add the ready/release edges.
    jobs = [None] * 3
    cursor = {(g, s): 0 for g in groups for s in range(slots)}
    # Event kinds: input ready, weight ready, MAC complete, output ready,
    # accumulation bank free, weight slot free, and command dispatch.
    events = []
    order = 0
    dma = Transfers(transfers or CommandTransfers((), 0), slots)
    dispatched = -1
    waiting = None
    command_done = {}
    per_command = (
        command_size // tile_steps,
        command_size // policy[1],
        command_size // output_steps,
    )

    def bank(role, index):
        command = index // per_command[role]
        return groups[role], (
            command // periods[role]
        ) % slots if commands > periods[role] else 0

    ready = [set(), set()]
    free = [[True] * 2, [True] * capacity]
    accum_free = [True, True]
    next_load = [0, 0]
    counts = [input_tiles, weight_loads]
    reads = [input_reads, weight_reads]
    fill = [input_fill, weight_fill]
    first = [input_first, weight_first]
    compute_finish = 0
    compute = 0
    computing = False
    out_queue = []
    seen = {}

    def event(at, kind, index):
        nonlocal order
        order += 1
        heappush(events, (at, order, kind, index))

    def claim(role, slot):
        # A load that read ahead takes the slot as it frees. Its buffered
        # beats enter storage only now, so readiness cannot precede a fresh
        # fill from this edge; read-ahead saves only shared-port service.
        job = jobs[role]
        if job is not None and job[3] is not None and job[1] % len(free[role]) == slot:
            free[role][slot] = False
            job[3] = None
            job[2] = max(
                job[2],
                now
                + (first[role] if job[1] % per_command[role] == 0 else fill[role]),
            )

    event(dma.dispatch(), 6, 0)
    while (
        compute < size
        or computing
        or events
        or any(jobs)
        or out_queue
        or any(queues)
        or waiting is not None
    ):
        while events and events[0][0] <= now:
            at, _, kind, index = heappop(events)
            stage_seen.clear()
            if kind in (0, 1):
                ready[kind].add(index)
            elif kind == 2:
                t, w, o = coordinates(index)
                if index // sets % replays == replays - 1:
                    event(now + 1, 5, w)
                if ((t + 1) * tile_steps - 1) == index:
                    free[0][t % 2] = True
                    ready[0].remove(t)
                    claim(0, t % 2)
                if ((o + 1) * output_steps - 1) == index:
                    accum_free[o % 2] = False
                    event(now + latency, 3, o)
                compute += 1
                computing = False
            elif kind == 3:
                out_queue.append(index)
            elif kind == 4:
                accum_free[index % 2] = True
                if (
                    not pipeline
                    and index % per_command[2] == per_command[2] - 1
                ):
                    command_done[index // per_command[2]] = now
            elif kind == 5:
                free[1][index % capacity] = True
                ready[1].remove(index)
                claim(1, index % capacity)
            else:
                dispatched = index
                waiting = index - (slots - 1)
        if waiting is not None and (waiting < 0 or waiting in command_done):
            dma.retire([command_done.get(waiting, 0)] + [0] * (slots - 1))
            waiting = None
            if dma.index < commands:
                event(max(now, dma.dispatch()), 6, dma.index)
        if cutoff < float("inf") and compute < size:
            # Even with all future operands ready and all resources independent,
            # the remaining MACs, bank handoffs and final output must complete.
            remaining = size - compute
            handoffs = output_tiles - compute // output_steps
            compute_bound = remaining * run * interval + handoffs * handoff
            if computing:
                compute_bound += (
                    compute_finish
                    - now
                    - run * interval
                    - (handoff if (compute + 1) % output_steps == 0 else 0)
                )
            if (
                now
                + compute_bound
                + latency
                + final_vectors * beats
                + dma.spec.store
                > cutoff
            ):
                return float("inf")
        check_limit = 0
        phase = None
        if batch and not computing and 0 < compute < size:
            t, w, o = coordinates(compute)
            command = compute // command_size
            boundary = (command + 1) * command_size
            if compute % command_size == 0:
                phase = (
                    "command",
                    tuple(
                        command % (slots * p) if p < commands else 0
                        for p in periods
                    ),
                    tuple(command % p if p else 0 for _, p in dma.spec.loads),
                )
                check_limit = max(0, size - 2 * command_size)
            elif compute < limit and compute % repeat == 0:
                phase, check_limit = (
                    (
                        "outer",
                        command,
                        compute // (policy[0] * policy[1])
                        if policy[1] > 1
                        else 0,
                    ),
                    min(
                        limit,
                        boundary
                        - max(
                            2 * tile_steps,
                            2 * output_steps,
                            capacity * policy[0]
                            if policy[1] == 1
                            else policy[0],
                        ),
                    ),
                )
                sets, replays, _ = policy
                if replays > 1:
                    # Interior replays retain the same resident sequence.
                    # Stop before the final replay releases its first set.
                    check_limit = min(
                        check_limit,
                        (compute // (sets * replays) + 1) * sets * replays
                        - sets,
                    )
            else:
                sets, replays, _ = policy
                replay = compute // sets % replays
                if replays == 1:
                    check_limit = min(
                        ((t + 1) * tile_steps - 1) - 1, size - capacity * sets
                    )
                elif replay < replays - 1:
                    check_limit = min(
                        ((t + 1) * tile_steps - 1) - 1,
                        (compute // (sets * replays) + 1) * sets * replays
                        - sets,
                    )
                phase = ("inner", t, compute % sets)
                check_limit = min(check_limit, boundary - 1)
        if check_limit > compute:
            t, w, o = coordinates(compute)
            bases = (t, w, o)
            command = compute // command_size
            command_done = {
                c: v for c, v in command_done.items() if c >= command - slots
            }

            def base(kind):
                return (t, w, compute, o, o, w, command)[kind]

            state = (
                dispatched - command,
                None if waiting is None else waiting - command,
                dma.index - command,
                tuple(
                    max(0, v - now)
                    for v in (dma.now, dma.dma, dma.store_done, *dma.ready)
                ),
                tuple(
                    (c - command, max(0, v - now))
                    for c, v in sorted(command_done.items())
                ),
                phase,
                t % 2,
                o % 2,
                (
                    produced - o * final_vectors,
                    routed - o * final_vectors,
                    consumed - o * final_vectors,
                )
                if pipeline
                else (),
                tuple(
                    tuple(v - now if v > now else 0 for v in q) for q in queues
                ),
                max(0, source_ready - now),
                max(0, vector_gate - now),
                tuple(
                    tuple(sorted(v - bases[r] for v in values))
                    for r, values in enumerate(ready)
                ),
                tuple(free[0]),
                tuple(free[1][(w + i) % capacity] for i in range(capacity)),
                tuple(accum_free),
                tuple(next_load[r] - bases[r] for r in (0, 1)),
                tuple(
                    (j[0], j[1] - bases[r], max(0, j[2] - now), j[3])
                    if j
                    else None
                    for r, j in enumerate(jobs)
                ),
                tuple(
                    (at - now, kind, index - base(kind))
                    for at, _, kind, index in sorted(events)
                ),
                tuple(v - o for v in out_queue),
                tuple(cursor[g] for g in sorted(cursor)),
            )
            prior = seen.get(state)
            if prior and compute > prior[0]:
                old_i, old_time, old_base = prior
                span = compute - old_i
                repeats = (check_limit - compute) // span
                if repeats:
                    delta = (now - old_time) * repeats
                    shifts = tuple(
                        (b - a) * repeats for a, b in zip(old_base, bases)
                    )
                    previous_command = compute // command_size
                    compute += span * repeats
                    now += delta
                    command_shift = compute // command_size - previous_command
                    shift = shifts[1] % capacity
                    if shift:
                        free[1] = free[1][-shift:] + free[1][:-shift]
                    produced += shifts[2] * final_vectors
                    routed += shifts[2] * final_vectors
                    consumed += shifts[2] * final_vectors
                    queues = [deque(v + delta for v in q) for q in queues]
                    source_ready += delta
                    vector_gate += delta
                    dispatched += command_shift
                    if waiting is not None:
                        waiting += command_shift
                    dma.advance(delta, command_shift)
                    command_done = {
                        c + command_shift: v + delta
                        for c, v in command_done.items()
                    }
                    ready = [
                        {v + shifts[r] for v in values}
                        for r, values in enumerate(ready)
                    ]
                    next_load = [v + shifts[r] for r, v in enumerate(next_load)]
                    jobs = [
                        [j[0], j[1] + shifts[r], j[2] + delta, j[3]]
                        if j
                        else None
                        for r, j in enumerate(jobs)
                    ]
                    events = [
                        (
                            at + delta,
                            seq,
                            kind,
                            index
                            + (
                                shifts[0],
                                shifts[1],
                                span * repeats,
                                shifts[2],
                                shifts[2],
                                shifts[1],
                                command_shift,
                            )[kind],
                        )
                        for at, seq, kind, index in events
                    ]
                    out_queue = [v + shifts[2] for v in out_queue]
                    seen.clear()
                    stage_seen.clear()
                    continue
            else:
                seen[state] = (compute, now, bases)
        if batch and pipeline and not any(j and j[3] is not None for j in jobs):
            # Within a fixed tile, skip complete repeated service rotations.
            # Stop before every compute, load, tile or command boundary.
            stream_state = (
                tuple(
                    tuple(v - now if v > now else 0 for v in q) for q in queues
                ),
                max(0, source_ready - now),
                max(0, vector_gate - now),
                tuple((j[1], max(0, j[2] - now)) if j else None for j in jobs),
                tuple(cursor.values()),
            )
            prior = stage_seen.get(stream_state)
            counts_now = (produced, routed, consumed)
            amounts = tuple(j[0] if j else 0 for j in jobs)
            if prior:
                old_time, old_counts, old_amounts = prior
                dt = now - old_time
                growth = tuple(b - a for a, b in zip(old_counts, counts_now))
                service = tuple(a - b for a, b in zip(old_amounts, amounts))
                if dt > 0 and min(service) >= 0:
                    bounds = [(events[0][0] - now - 1) // dt] if events else []
                    bounds.extend(
                        (final_vectors - v % final_vectors - 1) // d
                        for v, d in zip(counts_now, growth)
                        if d > 0
                    )
                    bounds.extend(
                        (v - 1) // d for v, d in zip(amounts, service) if d > 0
                    )
                    cycles = min(bounds) if bounds else 0
                    if cycles > 0:
                        delta = dt * cycles
                        now += delta
                        produced += growth[0] * cycles
                        routed += growth[1] * cycles
                        consumed += growth[2] * cycles
                        queues = [deque(v + delta for v in q) for q in queues]
                        source_ready += delta
                        vector_gate += delta
                        for r, j in enumerate(jobs):
                            if j:
                                j[0] -= service[r] * cycles
                                j[2] += delta
                        stage_seen.clear()
                        continue
            else:
                stage_seen[stream_state] = (now, counts_now, amounts)
        for role in (0, 1):
            index = next_load[role]
            if (
                jobs[role] is None
                and index < counts[role]
                and index // per_command[role] <= dispatched
            ):
                slot = index % len(free[role])
                # An occupied slot admits only the controller's own read-ahead;
                # the final beat always waits for the slot (see claim).
                ahead = None if free[role][slot] else min(
                    readahead[role], reads[role] - 1
                )
                if ahead is None or ahead > 0:
                    stage_seen.clear()
                    if ahead is None:
                        free[role][slot] = False
                    jobs[role] = [
                        reads[role],
                        index,
                        now
                        + (
                            first[role]
                            if index % per_command[role] == 0
                            else fill[role]
                        ),
                        ahead,
                    ]
                    next_load[role] += 1
        if pipeline:
            # A source vector belongs to its accumulation tile until the
            # matrix-to-vector handshake, independently of its later store.
            # Full stages return credit on the following clock edge.
            space = [len(q) < stage[0] for q, stage in zip(queues, pipeline)]
            if jobs[2] is None and queues[-1] and queues[-1][0] <= now:
                queues[-1].popleft()
                jobs[2] = [beats, consumed // final_vectors, 0, None]
            for st in range(len(queues) - 1, 0, -1):
                if space[st] and queues[st - 1] and queues[st - 1][0] <= now:
                    # The vector instruction for the next command cannot
                    # enter the router until the previous output completes.
                    if st == 1 and (
                        routed // (final_vectors * per_command[2])
                        > consumed // (final_vectors * per_command[2])
                        or now < vector_gate
                    ):
                        continue
                    queues[st - 1].popleft()
                    queues[st].append(now + pipeline[st][1])
                    if st == 1:
                        routed += 1
            if out_queue and space[0] and now >= source_ready:
                queues[0].append(now + pipeline[0][1])
                produced += 1
                if produced % final_vectors == 0:
                    stage_seen.clear()
                    index = out_queue.pop(0)
                    event(now + drain, 4, index)
                    source_ready = now + drain + 1
        elif jobs[2] is None and out_queue:
            jobs[2] = [final_vectors * beats, out_queue[0], 0, None]
        if not computing and compute < size:
            t, w, o = coordinates(compute)
            if (
                compute // command_size <= dispatched
                and t in ready[0]
                and w in ready[1]
                and accum_free[o % 2]
            ):
                stage_seen.clear()
                computing = True
                compute_finish = (
                    now
                    + run * interval
                    + (
                        handoff
                        if ((o + 1) * output_steps - 1) == compute
                        else 0
                    )
                )
                event(compute_finish, 2, compute)
        active = {
            g: [
                r
                for r, j in enumerate(jobs)
                if j and bank(r, j[1]) == g and j[3] != 0
            ]
            for g in cursor
        }
        ordered = {
            g: sorted(rs, key=lambda r: (r - cursor[g]) % 3)
            for g, rs in active.items()
            if rs
        }
        # Stop when a job finishes its words or exhausts its read-ahead.
        width = min(
            (
                len(rs)
                * ((jobs[r][0] if jobs[r][3] is None else jobs[r][3]) - 1)
                + pos
                + 1
                for rs in ordered.values()
                for pos, r in enumerate(rs)
            ),
            default=float("inf"),
        )
        until = events[0][0] - now if events else float("inf")
        span = min(width, until)
        if pipeline and (any(queues) or out_queue or jobs[2]):
            span = min(span, 1)
        if span == float("inf") and compute == size and not out_queue:
            break
        if span == float("inf"):
            raise RuntimeError(
                (
                    "deadlock",
                    compute,
                    coordinates(compute) if compute < size else None,
                    now,
                    ready,
                    free,
                    accum_free,
                    next_load,
                    jobs,
                    events,
                    dispatched,
                    waiting,
                )
            )
        if span == 0:
            continue
        for g, rs in ordered.items():
            turns, extra = divmod(span, len(rs))
            for pos, r in enumerate(rs):
                served = turns + (pos < extra)
                jobs[r][0] -= served
                if jobs[r][3] is not None:
                    jobs[r][3] -= served
            cursor[g] = (rs[(span - 1) % len(rs)] + 1) % 3
        now += span
        for r in range(3):
            if jobs[r] is not None and jobs[r][0] == 0:
                index, ready_at = jobs[r][1], jobs[r][2]
                jobs[r] = None
                if r < 2:
                    stage_seen.clear()
                    event(max(now, ready_at) + (r == 1), r, index)
                elif pipeline:
                    consumed += 1
                    if consumed % final_vectors == 0:
                        stage_seen.clear()
                    if consumed % (final_vectors * per_command[2]) == 0:
                        command_done[
                            consumed // (final_vectors * per_command[2]) - 1
                        ] = now
                        vector_gate = now + 1
                else:
                    out_queue.pop(0)
                    event(now + drain, 4, index)
    return dma.finish(command_done[commands - 1])
