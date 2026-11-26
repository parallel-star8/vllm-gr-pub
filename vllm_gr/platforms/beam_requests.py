# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Host-side device policy shared by Beam frontends and EngineCore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm_gr.config import GRConfig
from vllm_gr.config.models import (
    ConstraintBackendKind,
    ConstraintMode,
    configured_constraint_mode,
    runtime_constraint_table,
)
from vllm_gr.platforms.admission import beam_limits, validate_beam_batch

if TYPE_CHECKING:
    from vllm_gr.v1.beam.batch_contracts import BeamBatchRequest


def validate_gr_device_config(config: Any, request: BeamBatchRequest | None = None) -> None:
    """Check execution support without importing a device runtime."""
    limits = beam_limits(config)
    device = limits.device_type
    if device not in (None, "cuda", "npu"):
        raise ValueError(
            "Asynchronous scheduling requires a CUDA ModelRunner or NPU ModelRunner V1"
        )
    gr_config = GRConfig.from_vllm_config(config)
    if gr_config.beam_sid_stage_vocab and device != "npu":
        raise ValueError("beam.sid_stage_vocab requires NPU ModelRunner V1")
    if device == "npu":
        from vllm_gr.platforms.ascend.admission import validate_npu_device_config

        validate_npu_device_config(limits, gr_config, request)


def validate_decode_token_budget(request: BeamBatchRequest, config: Any) -> None:
    validate_beam_batch(request, beam_limits(config))


def decode_batch_capacity(config: Any, beam_width: int) -> int | None:
    """Return the physical Decode batch cap used for frontend grouping."""
    budget = beam_limits(config).max_decode_rows
    return budget // beam_width if budget is not None else None


def validate_dynamic_width_support(config: Any) -> None:
    if beam_limits(config).constraint_backend is None:
        raise ValueError("dynamic_beam_widths and result_width require CUDA or NPU beam_search")


def validate_frontend_constraint_mode(
    config: Any, gr_config: GRConfig, catalog_strategy: str | None
) -> None:
    placement = beam_limits(config).constraint_backend
    if configured_constraint_mode(gr_config) is ConstraintMode.NONE:
        if placement is None:
            raise ValueError("unconstrained beam_search requires CUDA or NPU")
        return

    table = runtime_constraint_table(gr_config)
    backend = ConstraintBackendKind(placement or "cuda")
    if table is None or not table.enabled or table.backend is not backend:
        raise ValueError(
            f"beam_search requires an enabled canonical {backend.value.upper()} constraint table"
        )
    if catalog_strategy is not None:
        raise ValueError("catalog_strategy is not applicable to the constraint_table backend")
