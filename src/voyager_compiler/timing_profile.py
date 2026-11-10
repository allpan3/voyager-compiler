"""Load recorded timing inputs before mapping search."""

import json
import warnings
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

ACCELERATOR_FIELDS = {
    "kernel_launch_overhead_cycles",
    "spmm_row_cycles",
}
CIM_FIELDS = {
    "accumulation_feedback_cycles",
    "output_bank_drain_cycles",
    "output_pipeline",
}
SOC_FIELDS = {"name", "bank_switch_cycles"}


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
                "to a JSON file."
            )
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
                "name",
                "metadata",
                "accelerator",
                "soc",
                "evidence",
            },
            "timing profile",
        )
        if not isinstance(data["name"], str) or not data["name"]:
            raise ValueError("A timing profile needs a name")
        if not isinstance(data["metadata"], dict):
            raise ValueError("Timing metadata must be an object")
        optional = (
            set(data["accelerator"])
            if isinstance(data["accelerator"], dict)
            else set()
        ) & {"cim", "sa_output_pipeline", "sa_input_pipeline"}
        _keys(data["accelerator"], ACCELERATOR_FIELDS | optional, "accelerator")
        sa_records = {"sa_input_pipeline", "sa_output_pipeline"}
        if optional & sa_records and not sa_records <= optional:
            raise ValueError(
                "SA timing requires both input and output pipeline records; "
                "omit both to use architectural timing"
            )
        if "cim" in optional:
            _keys(
                data["accelerator"]["cim"],
                CIM_FIELDS
                | (
                    {"banked_pipeline"}
                    if "banked_pipeline" in data["accelerator"]["cim"]
                    else set()
                ),
                "accelerator.cim",
            )
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
                for k, v in data["accelerator"].get("cim", {}).items()
            }
        )
        values.update(
            {f"soc.{k}": v for k, v in data["soc"].items() if k != "name"}
        )
        for name, value in values.items():
            if name.endswith(
                ("output_pipeline", "input_pipeline", "banked_pipeline")
            ):
                continue  # Pipeline types validate these records below.
            if value is None and name != "soc.bank_switch_cycles":
                continue  # These consumers explicitly report unknown delays.
            minimum = 1 if name.endswith("accumulation_feedback_cycles") else 0
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
            result.sa_output_pipeline
            if "sa_input_pipeline" in result.accelerator:
                values = result.accelerator["sa_input_pipeline"]
                _keys(
                    values,
                    {
                        "input_buffer_forward_cycles",
                        "weight_buffer_forward_cycles",
                        "input_buffer_read_cycles",
                        "weight_buffer_read_cycles",
                        "array_forward_cycles",
                        "accumulation_forward_cycles",
                    },
                    "accelerator.sa_input_pipeline",
                )
                if any(type(v) is not int or v < 0 for v in values.values()):
                    raise ValueError(
                        "SA input delays must be nonnegative integers"
                    )
            from voyager_compiler.codegen.transform.tiling.timing.output import (
                BankedOutputPipeline,
                OutputOptions,
            )

            banked = result.accelerator.get("cim", {}).get("banked_pipeline")
            if banked is not None:
                BankedOutputPipeline.from_dict(banked)
            pipeline = OutputOptions(
                output_pipeline=result.accelerator.get("cim", {}).get(
                    "output_pipeline"
                )
            ).output_pipeline
            if pipeline is not None:
                for stage in pipeline.stages:
                    for name in ("forward_cycles", "feedback_cycles"):
                        value = getattr(stage, name)
                        if type(value) is not int or value < 0:
                            raise ValueError(
                                f"{name} must be a nonnegative integer"
                            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid pipeline timing: {error}") from error
        return result

    @cached_property
    def sa_output_pipeline(self):
        from voyager_compiler.codegen.transform.tiling.timing.sa_pipeline import (
            SAOutputPipeline,
        )

        value = self.accelerator.get("sa_output_pipeline")
        return None if value is None else SAOutputPipeline.from_dict(value)

    def cycles(self, path):
        """Return a recorded delay, warning when only a lower bound is known."""
        value = self.data
        for key in path.split("."):
            value = value[key]
        if value is None:
            warnings.warn(
                f"{path} is uncharacterized for this build; the estimate omits "
                "this physical delay and is a lower bound for that path.",
                RuntimeWarning,
                stacklevel=2,
            )
            return 0
        return value

    def cim_options(self):
        # Import after AcceleratorConfig construction to avoid package cycles.
        from voyager_compiler.codegen.transform.tiling.cim_timing import (
            TimingOptions,
        )

        if "cim" not in self.accelerator:
            raise ValueError(
                "This timing profile has no CIM build measurements"
            )
        values = dict(self.accelerator["cim"])
        values["output_bank_drain_cycles"] = self.cycles(
            "accelerator.cim.output_bank_drain_cycles"
        )
        return TimingOptions(**values)
