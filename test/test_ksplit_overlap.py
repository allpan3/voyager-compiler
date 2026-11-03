"""K-split routing and semaphore regressions; run with unittest."""

import unittest

import torch

from voyager_compiler.codegen.subgraph import create_and_insert_subgraph
from voyager_compiler.codegen.transform.bufferize.emit import (
    gen_code_bufferized,
)
from voyager_compiler.codegen.transform.bufferize.pipeline import (
    _gemm_plan,
    _gemm_scratch_and_kernel,
    build_conv2d,
    build_gemm,
)
from voyager_compiler.shape_prop import ShapeProp


def walk(ops):
    for op in ops:
        yield op
        kind = op.WhichOneof("op_type")
        if kind == "loop":
            yield from walk(op.loop.for_loop.body.ops)
        elif kind == "cond":
            yield from walk(op.cond.true_region.ops)
            yield from walk(op.cond.false_region.ops)
        elif kind == "async":
            yield from walk(getattr(op, "async").body.ops)


def make_case(
    rounds=2,
    logical_dtype=None,
    tail=False,
    rows=8,
    dtype=torch.float32,
    columns=8,
    reduction=8,
    output_tiles=None,
):
    graph = torch.fx.Graph()
    a = graph.placeholder("input")
    b = graph.placeholder("other")
    scale = graph.placeholder("scale") if tail == "dequantize" else None
    matrix = graph.call_function(torch.ops.aten.matmul.default, (a, b))
    if tail == "dequantize":
        result = graph.call_function(
            torch.ops.quantized_ops.dequantize.default, (matrix, scale)
        )
    else:
        result = (
            graph.call_function(torch.ops.aten.relu.default, (matrix,))
            if tail
            else matrix
        )
    graph.output(result)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    inputs = (
        torch.arange(rows * reduction).reshape(rows, reduction).to(dtype) % 3
        - 1,
        torch.arange(reduction * columns).reshape(reduction, columns).to(dtype)
        % 3
        - 1,
    )
    if scale is not None:
        inputs += (torch.tensor(0.5, dtype=dtype),)
    ShapeProp(gm).propagate(*inputs)
    output_tiles = output_tiles or (1 if rows == 1 else 2, 2)
    matrix.meta["l2_tiling"] = (*output_tiles, rounds)
    if logical_dtype is not None:
        matrix.meta["dtype"] = logical_dtype
    expected = gm(*inputs)
    node = matrix
    if tail:
        node = create_and_insert_subgraph([matrix, result], gm)
    return node, inputs, expected


def compile_case(synchronous=False, accumulate_fp32=False, **options):
    node, inputs, expected = make_case(**options)
    lowered = build_gemm(
        node, async_pipeline=not synchronous, accumulate_fp32=accumulate_fp32
    )
    # Eager execution exercises all rounds and reused slots with semaphore
    # assertions enabled, rather than only inspecting one symbolic iteration.
    actual = lowered(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    model = gen_code_bufferized(lowered, inputs)
    return routing_and_waits(model)


def routing_and_waits(model):
    routed = []
    waits = []
    for op in walk(model.ops):
        if op.HasField("async"):
            async_op = getattr(op, "async")
            if async_op.HasField("post") and any(
                dep.box.node == async_op.post.box.node
                for dep in async_op.dependencies
            ):
                waits.append(op)
        prims = [op.prim] if op.HasField("prim") else op.fused.op_list
        for prim in prims:
            if prim.kwargs.get("output_via_vector", None) is not None:
                assert prim.kwargs["output_via_vector"].scalar.bool_value
                routed.append((op, prim))
    return routed, waits


class KSplitOverlapTest(unittest.TestCase):
    def test_float_rounds_keep_overlap_and_balanced_semaphores(self):
        for rounds in (2, 4):
            for tail in (False, True):
                with self.subTest(rounds=rounds, tail=tail):
                    routed, waits = compile_case(rounds=rounds, tail=tail)
                    self.assertEqual(len(routed), 1)
                    self.assertFalse(waits)
                    op, prim = routed[0]
                    self.assertIn("matmul", prim.target)
                    self.assertTrue(
                        op.HasField("prim") or len(op.fused.op_list) == 1
                    )

    def test_odd_round_counts_and_reused_output_grids(self):
        # Three row and column tiles reuse both pipeline slots repeatedly.
        # Odd reduction counts also change which slot holds the final partial.
        for rounds in (3, 6):
            for tail in (False, True):
                with self.subTest(rounds=rounds, tail=tail):
                    routed, waits = compile_case(
                        rounds=rounds,
                        tail=tail,
                        rows=12,
                        columns=12,
                        reduction=12,
                        output_tiles=(3, 3),
                    )
                    self.assertEqual(len(routed), 1)
                    self.assertFalse(waits)

    def test_logical_integer_accumulators_keep_dependency(self):
        # Check the real kernel with concrete coordinates. Symbolic export of
        # this existing dependency path rejects aliased semaphore views on
        # Torch 2.12, including on the unmodified base compiler.
        for dtype in ("int24", "int32", "unknown"):
            for tail in (False, True):
                with self.subTest(dtype=dtype, tail=tail):
                    node, _, _ = make_case(logical_dtype=dtype, tail=tail)
                    plan = _gemm_plan(node)
                    specs, kernel = _gemm_scratch_and_kernel(
                        plan.gemm_kernel,
                        plan.fused_gm,
                        reduction_dim=plan.grid_dims[2],
                        num_k=plan.num_k,
                        acc_shape=plan.acc_shape,
                        in_specs=plan.in_specs,
                        out_specs=plan.out_specs,
                        fused_idx=plan.fused_idx,
                        anchor=plan.anchor,
                        accumulate_fp32=False,
                        async_pipeline=True,
                    )
                    slots = [
                        torch.ones(spec.tile_sizes) for spec in plan.in_specs
                    ]
                    outputs = [torch.zeros(plan.acc_shape)]
                    scratch = [torch.zeros(plan.acc_shape) for _ in specs]
                    previous = torch.tensor(0)
                    post = torch.tensor(0)

                    def run(round_index):
                        return kernel(
                            (0, 0, round_index),
                            slots,
                            outputs,
                            scratch,
                            [torch.tensor(1) for _ in slots],
                            [torch.tensor(1)],
                            post,
                            lambda: previous,
                        )

                    # Round 1 cannot read a first partial whose producer has
                    # not signaled. Once it has, consume exactly one credit;
                    # retirement must not consume that same credit again.
                    with self.assertRaisesRegex(
                        AssertionError, "no pending signal"
                    ):
                        run(1)
                    previous.fill_(1)
                    self.assertFalse(run(1))
                    self.assertEqual(previous.item(), 0)
                    self.assertEqual(post.item(), 1)
                    torch.testing.assert_close(
                        outputs[0], torch.full(plan.acc_shape, 4.0)
                    )

    def test_supported_float_storage(self):
        for dtype in ("bfloat16", "float32", "fp8_e4m3"):
            with self.subTest(dtype=dtype):
                routed, waits = compile_case(logical_dtype=dtype)
                self.assertEqual(len(routed), 1)
                self.assertFalse(waits)

    def test_single_round_and_synchronous_paths_are_unchanged(self):
        for options in ({"rounds": 1}, {"synchronous": True}, {"rows": 1}):
            with self.subTest(**options):
                routed, waits = compile_case(**options)
                self.assertFalse(routed)
                self.assertFalse(waits)

    def test_scratch_accumulation_path(self):
        routed, waits = compile_case(dtype=torch.bfloat16, accumulate_fp32=True)
        self.assertFalse(waits)
        # The widening cast already supplies a vector stage; no redundant
        # routing flag should be attached to that fused operation.
        self.assertFalse(routed)

    def test_leading_dequantize_already_orders_first_partial(self):
        routed, waits = compile_case(tail="dequantize", logical_dtype="int24")
        self.assertFalse(routed)
        self.assertFalse(waits)

    def test_convolution_reduction_preserves_routing_through_alias(self):
        graph = torch.fx.Graph()
        a = graph.placeholder("input")
        b = graph.placeholder("weight")
        conv = graph.call_function(torch.ops.aten.conv2d.default, (a, b))
        graph.output(conv)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        inputs = (torch.ones(1, 8, 4, 4), torch.ones(8, 8, 1, 1))
        ShapeProp(gm).propagate(*inputs)
        conv.meta["l2_tiling"] = (1, 2, 2, 2, 2)
        lowered = build_conv2d(conv)
        torch.testing.assert_close(lowered(*inputs), gm(*inputs))
        routed, waits = routing_and_waits(gen_code_bufferized(lowered, inputs))
        self.assertEqual(len(routed), 1)
        self.assertIn("conv2d", routed[0][1].target)
        self.assertFalse(waits)


if __name__ == "__main__":
    unittest.main()
