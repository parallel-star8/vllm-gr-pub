# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Physical limits shared by Beam admission and frontend grouping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_gr.config.backend_plan import backend_wire
from vllm_gr.config.resolved_choices import read_resolved_choices

if TYPE_CHECKING:
    from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


@dataclass(frozen=True, slots=True)
class BeamLimits:
    """Hardware placement and the native runner's Decode row budget."""

    device_type: str | None
    max_decode_rows: int | None
    retain_candidate_widths: bool
    constraint_backend: str | None
    runner_generation: str
    local_kv_heads: int | None
    head_size: int | None

    def uses_retained_widths(self, config: Any) -> bool:
        options = config.deployment
        return (
            self.retain_candidate_widths
            and options.candidate_width is not None
            and options.candidate_width == options.beam_width
        )

    def __post_init__(self) -> None:
        if self.device_type is not None and not isinstance(self.device_type, str):
            raise ValueError("Invalid Beam admission device_type")
        if type(self.retain_candidate_widths) is not bool:
            raise ValueError("Invalid Beam admission retain_candidate_widths")
        if self.constraint_backend is not None and not isinstance(self.constraint_backend, str):
            raise ValueError("Invalid Beam admission constraint_backend")
        if self.runner_generation not in ("v1", "v2"):
            raise ValueError("Invalid Beam admission runner_generation")
        budget = self.max_decode_rows
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError("Invalid Beam admission max_decode_rows")


def local_kv_geometry(config: Any) -> tuple[int | None, int | None]:
    """Query native KV geometry after tensor-parallel placement is known."""
    try:
        model = config.model_config
        heads = model.get_num_kv_heads(getattr(config, "parallel_config", None))
        head_size = model.get_head_size()
    except (AttributeError, TypeError, ValueError):
        return None, None
    if type(heads) is not int or type(head_size) is not int:
        raise ValueError("NPU unconstrained beam_search requires positive integer KV-head geometry")
    return heads, head_size


def beam_limits(config: Any) -> BeamLimits:
    wire = backend_wire(config)
    if wire is not None:
        # Re-admission replaces the envelope. Keep the verified limits on that
        # engine, so every member of a batch need not re-hash the engine plan.
        cached = getattr(config, "_gr_beam_limits", None)
        if isinstance(cached, tuple) and cached[0] is wire:
            return cached[1]
        admitted = read_resolved_choices(config)
        assert admitted is not None
        choices = admitted.choices.to_dict()
        if "beam_limits" in choices:
            fields = choices["beam_limits"]
            if not isinstance(fields, dict) or fields.keys() != {
                "device_type",
                "max_decode_rows",
                "retain_candidate_widths",
                "constraint_backend",
                "runner_generation",
                "local_kv_heads",
                "head_size",
            }:
                raise ValueError("Invalid Beam admission limits")
            limits = BeamLimits(**fields)
            try:
                object.__setattr__(config, "_gr_beam_limits", (wire, limits))
            except (AttributeError, TypeError):
                pass
            return limits
    # Configs produced before limits were recorded carry only native placement.
    device = getattr(getattr(config, "device_config", None), "device", None)
    placement = str(device).split(":", 1)[0] if device is not None else None
    scheduler = getattr(config, "scheduler_config", None)
    budget = getattr(scheduler, "max_num_batched_tokens", None) if placement == "npu" else None
    backend = placement if placement in ("cuda", "npu") else ("cuda" if placement is None else None)
    generation = "v2" if getattr(config, "use_v2_model_runner", False) else "v1"
    heads, size = None, None
    if placement == "npu":
        from vllm_gr.config import GRConfig

        if GRConfig.from_vllm_config(config).beam_max_decode_steps > 3:
            heads, size = local_kv_geometry(config)
    return BeamLimits(placement, budget, placement == "npu", backend, generation, heads, size)


def validate_beam_batch(request: BeamBatchRequest, limits: BeamLimits) -> None:
    """Check the peak number of input rows across Decode stages."""
    budget = limits.max_decode_rows
    if budget is None or request.beam_params.max_tokens == 1:
        return
    params = request.beam_params
    # Prefill produces the first winners. Each remaining invocation consumes
    # the preceding stage's winners; the final output is never a model input.
    input_width = (
        params.beam_width
        if params.dynamic_beam_widths is None
        else max(params.dynamic_beam_widths[:-1])
    )
    physical_tokens = len(request.items) * input_width
    if physical_tokens > budget:
        device = limits.device_type.upper() if limits.device_type else "Native"
        raise ValueError(
            f"{device} Beam Decode requires {physical_tokens} physical tokens "
            f"(batch_size={len(request.items)} * peak_input_width={input_width}), "
            f"exceeding max_num_batched_tokens={budget}; increase the token budget "
            "or reduce the batch size or Decode input widths"
        )
