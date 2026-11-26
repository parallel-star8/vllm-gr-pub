# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Worker readiness RPC and cross-worker startup agreement."""

from __future__ import annotations

import logging
from typing import Any

from vllm_gr.config.backend_plan import (
    BackendPlan,
    PlanData,
    read_backend_plan,
)
from vllm_gr.platforms.services import validate_required_services

REPORT_FORMAT_VERSION = 2

logger = logging.getLogger(__name__)


def _attention_by_kind(runner: Any) -> dict[str, list[str]]:
    groups = getattr(runner, "attn_groups", ())
    if not groups:
        return {}
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    selected: dict[str, set[str]] = {}
    for group in groups:
        for attention in group:
            kind = get_kv_cache_spec_kind(attention.kv_cache_spec).value
            selected.setdefault(kind, set()).add(attention.backend.get_name())
    return {kind: sorted(names) for kind, names in selected.items()}


def _graph_report(binding: Any, runner: Any) -> dict[str, Any] | None:
    if binding.stage is None or binding.stage.graph_status is None:
        return None
    status = binding.stage.graph_status(runner)
    captured, reasons = status["captured"], status["fallback"]
    if captured:
        outcome = "partial_fallback" if reasons else "captured"
    else:
        outcome = "fallback" if reasons else "not_captured"
    return {"outcome": outcome, "captured": captured, "fallback": reasons}


def worker_backend_readiness(worker: Any) -> dict[str, Any]:
    """Callable RPC: native executors pass their Worker as the first argument."""
    from vllm_gr.v1.worker.runner_binding import bind_backend_services

    runner = getattr(worker, "model_runner", None)
    if runner is None:
        raise RuntimeError("Worker has no model runner for backend plan validation")
    binding = bind_backend_services(runner)
    if binding is None:
        raise RuntimeError(
            "Worker did not receive the engine backend plan; use the same vllm-gr version on all ranks"
        )
    report = binding.report()
    model = getattr(getattr(runner, "vllm_config", None), "model_config", None)
    table = getattr(getattr(runner, "constraint_backend", None), "info", None)
    metadata = getattr(table, "table_metadata", None)
    report.update(
        config_digest=binding.plan.request.config_digest,
        backend_id=binding.plan.backend_id,
        compatibility=binding.plan.request.compatibility.to_dict(),
        format_version=REPORT_FORMAT_VERSION,
        attention_by_kind=_attention_by_kind(runner),
        decode_mode=getattr(binding.attention_executor, "DECODE_MODE", None),
        graph=_graph_report(binding, runner),
        max_model_len=getattr(model, "max_model_len", None),
        constraint_digest=getattr(metadata, "artifact_digest", None),
    )
    return report


def validate_worker_readiness(plan: BackendPlan, reports: Any, *, expected_workers: int) -> None:
    if type(expected_workers) is not int or expected_workers < 1:
        raise ValueError("Backend readiness requires a positive integer worker count")
    if not isinstance(reports, (list, tuple)) or len(reports) != expected_workers or not reports:
        raise RuntimeError(f"Backend readiness requires {expected_workers} worker reports")
    required = plan.required_services
    expected_identity = (
        plan.backend_id,
        plan.request.config_digest,
        plan.request.compatibility.to_dict(),
    )
    services = None
    runtime_limits = None
    native = plan.request.native.to_dict()
    admitted_length = native.get("model_config", {}).get("max_model_len")
    constraint = plan.request.config.to_dict().get("constraint") or {}
    table = constraint.get("table") or {}
    expected_digest = table.get("artifact_digest") if table.get("enabled") else None
    expected_graph = plan.choices.to_dict().get("graph")
    requested_decode = (
        plan.choices.to_dict().get("decode_attention") if "attention" in required else None
    )
    per_kind = native.get("attention_per_kind", {})
    for index, report in enumerate(reports):
        if not isinstance(report, dict):
            raise RuntimeError(f"Backend worker {index} is not ready")
        if report.get("ready") is not True:
            raise RuntimeError(
                f"Backend worker {index} is not ready: {report.get('error', 'unknown error')}"
            )
        if report.get("format_version") != REPORT_FORMAT_VERSION:
            raise RuntimeError(
                f"Backend worker {index} readiness format mismatch; use the same vllm-gr version on all ranks"
            )
        if (
            report.get("backend_id"),
            report.get("config_digest"),
            report.get("compatibility"),
        ) != expected_identity:
            raise RuntimeError(
                f"Backend worker {index} platform identity, normalized config or runtime version mismatch"
            )
        length = report.get("max_model_len")
        if admitted_length is not None and (
            type(length) is not int or not 0 < length <= admitted_length
        ):
            raise RuntimeError(
                f"Backend worker {index} max_model_len differs from its admission limit"
            )
        if admitted_length is not None and length < admitted_length:
            logger.warning(
                "Backend worker %d fitted max_model_len to %d (admitted %d)",
                index,
                length,
                admitted_length,
            )
        digest = report.get("constraint_digest")
        if expected_digest is not None and digest != expected_digest:
            raise RuntimeError(f"Backend worker {index} loaded constraint digest mismatch")
        graph = report.get("graph")
        graph_agreement = None
        if graph is not None:
            if (
                not isinstance(graph, dict)
                or not isinstance(graph.get("captured"), list)
                or not isinstance(graph.get("fallback"), dict)
            ):
                raise RuntimeError(f"Backend worker {index} returned an invalid graph report")
            if expected_graph == "eager" and graph["captured"]:
                raise RuntimeError(
                    f"Backend worker {index} captured Beam Decode graphs with eager admission"
                )
            graph_agreement = (graph.get("outcome"), graph["captured"], sorted(graph["fallback"]))
        actual_attention = report.get("attention_by_kind")
        if not isinstance(actual_attention, dict):
            raise RuntimeError(f"Backend worker {index} lacks resolved attention groups")
        for kind, selected in actual_attention.items():
            expected = per_kind.get(kind)
            if expected is not None and selected != [expected]:
                raise RuntimeError(
                    f"Backend worker {index} bound {kind} attention {selected}, expected {expected}"
                )
        decode = report.get("decode_mode")
        if requested_decode is not None and (
            decode in (None, "auto") or requested_decode not in ("auto", decode)
        ):
            raise RuntimeError(
                f"Backend worker {index} bound Decode mode {decode!r}, expected {requested_decode!r}"
            )
        limits = (length, digest, graph_agreement, actual_attention, decode)
        if runtime_limits is not None and limits != runtime_limits:
            raise RuntimeError(
                "Workers disagree on runtime limits, graph captures, attention groups or Decode mode"
            )
        runtime_limits = limits
        reported = report.get("services")
        if not isinstance(reported, dict):
            raise RuntimeError(f"Backend worker {index} has no bound services")
        validate_required_services(plan, required, reported, f"Backend worker {index}")
        current = PlanData(reported)
        if services is not None and current != services:
            raise RuntimeError("Workers disagree on bound backend service implementations")
        services = current


def _external_launcher_reports(executor: Any, expected_workers: int) -> list[Any]:
    """Collect the single local worker report from every externally launched rank."""
    from torch import distributed as dist
    from vllm.distributed.parallel_state import get_world_group

    group = get_world_group().cpu_group
    actual = dist.get_world_size(group)
    try:
        if actual != expected_workers:
            raise RuntimeError(
                f"External launcher readiness expected {expected_workers} ranks, got {actual}"
            )
        local = executor.collective_rpc(worker_backend_readiness)
        if not isinstance(local, (list, tuple)) or len(local) != 1:
            raise RuntimeError(
                "External launcher executor must return its single local worker report"
            )
        report = local[0]
    except Exception as exc:
        # Enter the collective on failed ranks as well; every peer must see the failure.
        report = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
    reports: list[Any] = [None] * actual
    dist.all_gather_object(reports, report, group=group)
    return reports


def verify_engine_backend_readiness(engine: Any) -> None:
    plan = read_backend_plan(engine.vllm_config)
    if plan is None:
        return
    parallel = getattr(engine.vllm_config, "parallel_config", None)
    # For native multiprocessing each DP EngineCore owns a TP/PP/PCP executor.
    expected = getattr(parallel, "world_size", 1)
    try:
        if (
            expected > 1
            and getattr(parallel, "distributed_executor_backend", None) == "external_launcher"
        ):
            reports = _external_launcher_reports(engine.model_executor, expected)
        else:
            reports = engine.model_executor.collective_rpc(worker_backend_readiness)
        validate_worker_readiness(plan, reports, expected_workers=expected)
    except BaseException:
        try:
            engine.model_executor.shutdown()
        except Exception:
            logger.exception("Failed to shut down executor after backend readiness failure")
        raise
    engine._gr_backend_fingerprint = plan.fingerprint
