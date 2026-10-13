"""Load recorded timing inputs before mapping search."""

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

ACCELERATOR_FIELDS = {
    "kernel_launch_overhead_cycles",
    "sa_output_slack_cycles",
    "spmm_row_cycles",
    "cim",
}
CIM_FIELDS = {
    "output_cycles_per_vector",
    "accumulation_feedback_cycles",
    "weight_release_cycles",
    "weight_ready_cycles",
    "output_bank_drain_cycles",
    "output_pipeline",
}
SOC_FIELDS = {"name", "bank_switch_cycles", "memory_request_latency_cycles"}


def _keys(value, expected, label):
    actual = set(value) if isinstance(value, dict) else set()
    if not isinstance(value, dict) or actual != expected:
        raise ValueError(
            f"{label}: missing {sorted(expected - actual)}, "
            f"unknown {sorted(actual - expected)}"
        )


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate timing field: {name}")
        result[name] = value
    return result


@dataclass(frozen=True)
class TimingProfile:
    """Keep record contents in config equality and search cache keys."""

    document: str

    @cached_property
    def data(self):
        return json.loads(self.document)

    @property
    def accelerator(self):
        return self.data["accelerator"]

    @property
    def soc(self):
        return self.data["soc"]

    @classmethod
    def load(cls, path):
        if not path:
            raise ValueError(
                "A recorded timing profile is required. Set --timing_profile "
                "to a JSON file, or use 'jeffrey-reference' for a reference comparison."
            )
        if str(path) == "jeffrey-reference":
            path = Path(__file__).with_name("jeffrey-reference.json")
        try:
            data = json.loads(
                Path(path).read_text(), object_pairs_hook=_unique_object
            )
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Cannot read timing profile {path}: {error}"
            ) from error
        _keys(
            data,
            {
                "schema_version",
                "name",
                "metadata",
                "accelerator",
                "soc",
                "evidence",
            },
            "timing profile",
        )
        if (
            type(data["schema_version"]) is not int
            or data["schema_version"] != 1
        ):
            raise ValueError("Unsupported timing profile schema_version")
        if not isinstance(data["name"], str) or not data["name"]:
            raise ValueError("A timing profile needs a name")
        if not isinstance(data["metadata"], dict):
            raise ValueError("Timing metadata must be an object")
        _keys(data["accelerator"], ACCELERATOR_FIELDS, "accelerator")
        _keys(data["accelerator"]["cim"], CIM_FIELDS, "accelerator.cim")
        _keys(data["soc"], SOC_FIELDS, "soc")
        if not isinstance(data["soc"]["name"], str) or not data["soc"]["name"]:
            raise ValueError("A timing profile needs a SoC name")
        values = {
            f"accelerator.{k}": v
            for k, v in data["accelerator"].items()
            if k != "cim"
        }
        values.update(
            {
                f"accelerator.cim.{k}": v
                for k, v in data["accelerator"]["cim"].items()
            }
        )
        values.update(
            {f"soc.{k}": v for k, v in data["soc"].items() if k != "name"}
        )
        for name, value in values.items():
            if name.endswith("output_pipeline"):
                continue  # The existing OutputPipeline type validates this object below.
            if name.endswith("accumulation_feedback_cycles") and value is None:
                continue  # Explicit unknown, not a zero-cycle hardware path.
            minimum = (
                1
                if name.endswith(
                    ("output_cycles_per_vector", "accumulation_feedback_cycles")
                )
                else 0
            )
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        _keys(data["evidence"], set(values), "evidence")
        for name, evidence in data["evidence"].items():
            _keys(evidence, {"status", "source"}, f"evidence.{name}")
            if evidence["status"] not in (
                "measured",
                "derived",
                "reference",
                "assumed",
                "unknown",
            ):
                raise ValueError(f"Invalid evidence status for {name}")
            if (
                not isinstance(evidence["source"], str)
                or not evidence["source"]
            ):
                raise ValueError(f"Missing evidence source for {name}")
            if values[name] is None and evidence["status"] != "unknown":
                raise ValueError(
                    f"{name}: null requires unknown evidence status"
                )
        result = cls(json.dumps(data, sort_keys=True, indent=2) + "\n")
        try:
            pipeline = result.cim_options().output_pipeline
            if pipeline is not None:
                for stage in pipeline.stages:
                    for name in ("forward_cycles", "feedback_cycles"):
                        value = getattr(stage, name)
                        if type(value) is not int or value < 0:
                            raise ValueError(
                                f"{name} must be a nonnegative integer"
                            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid CIM timing: {error}") from error
        return result

    def cim_options(self):
        # Import after AcceleratorConfig construction to avoid package cycles.
        from voyager_compiler.codegen.transform.tiling.cim_timing import (
            TimingOptions,
        )

        values = dict(self.accelerator["cim"])
        values["memory_request_latency"] = self.soc[
            "memory_request_latency_cycles"
        ]
        return TimingOptions(**values)
