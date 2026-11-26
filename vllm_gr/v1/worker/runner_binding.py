# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Build, adopt and release the services owned by one native runner."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from types import ModuleType
from typing import TYPE_CHECKING, Any

from vllm_gr.config.backend_plan import BackendPlan
from vllm_gr.config.resolved_choices import ResolvedChoices, read_resolved_choices
from vllm_gr.platforms.planning import worker_plan_from_choices
from vllm_gr.platforms.services import (
    BoundService,
    constraint_execution_override,
    planning_provider,
    validate_required_services,
)
from vllm_gr.v1.worker.runner_lifecycle import NativeRunnerAdapter
from vllm_gr.v1.worker.stage_binding import StageBinding

if TYPE_CHECKING:
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl


logger = logging.getLogger(__name__)
_BINDING_ATTR = "_gr_backend_binding"


def _beam_attention_layers(runner: Any) -> tuple[BeamAttentionImpl, ...]:
    """Only visit implementations owned by this model, including wrapped models."""
    model = getattr(runner, "model", None)
    # A model using native attention has no Beam implementation to bind.
    if model is None or "vllm_gr.v1.attention.backends.beam_attn" not in sys.modules:
        return ()
    get_model = getattr(runner, "get_model", None)
    if callable(get_model):
        model = get_model()
    from vllm_gr.v1.attention.backends.beam_attn import BeamAttentionImpl

    return tuple(
        impl
        for layer in model.modules()
        if isinstance(impl := getattr(layer, "impl", None), BeamAttentionImpl)
    )


def _implementation_id(service: Any) -> str | None:
    if service is None:
        return None
    if isinstance(service, BoundService):
        service = service.implementation
    if isinstance(service, ModuleType):
        return service.__name__
    declared = getattr(service, "implementation_id", None)
    if isinstance(declared, str):
        return declared
    return f"{type(service).__module__}.{type(service).__qualname__}"


def _close_services(services: list[Any]) -> None:
    seen = set()
    for service in reversed(services):
        if service is None or isinstance(service, BoundService) or id(service) in seen:
            continue
        seen.add(id(service))
        close = getattr(service, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # Continue releasing the remaining services, including when
                # cleanup follows a more useful startup exception.
                logger.exception("Failed to close backend service %s", _implementation_id(service))


@dataclass(frozen=True)
class BackendBinding:
    plan: BackendPlan
    runner_adapter: Any
    attention_executor: Any
    constraint_execution: Any
    canonical_constraint_id: str | None = None

    attention_layers: tuple[BeamAttentionImpl, ...] = ()

    stage: StageBinding | None = None

    def detach_attention(self) -> None:
        for impl in self.attention_layers:
            if vars(impl).get("execution") is self.attention_executor:
                del impl.execution

    def report(self) -> dict[str, Any]:
        return {
            "ready": True,
            "fingerprint": self.plan.fingerprint,
            "services": {
                "runner": _implementation_id(self.runner_adapter),
                **({"stage": self.stage.implementation_id} if self.stage is not None else {}),
                "attention": _implementation_id(self.attention_executor),
                "constraint": self.canonical_constraint_id
                or _implementation_id(self.constraint_execution),
            },
        }


def close_backend_binding(runner: Any) -> None:
    binding = getattr(runner, _BINDING_ATTR, None)
    setattr(runner, _BINDING_ATTR, None)
    if binding is not None:
        binding.detach_attention()
        if getattr(runner, "_gr_constraint_execution", None) is binding.constraint_execution:
            runner._gr_constraint_execution = None
            runner._gr_constraint_plan_fingerprint = None
        _close_services(
            [binding.runner_adapter, binding.attention_executor, binding.constraint_execution]
        )


def bind_backend_services(runner: Any) -> BackendBinding | None:
    """Validate the engine plan locally, then publish all services atomically."""
    from vllm_gr.platforms.loader import activate_current_backend

    config = getattr(runner, "vllm_config", None)
    expected = read_resolved_choices(config)
    if expected is None:
        return None
    existing = getattr(runner, _BINDING_ATTR, None)
    if existing is not None:
        if not isinstance(existing, BackendBinding):
            raise RuntimeError("Runner contains an invalid backend binding")
        if ResolvedChoices.from_plan(existing.plan) != expected:
            raise RuntimeError("Runner backend plan changed; reload its owned resources first")
        return existing
    provider = activate_current_backend()
    plan = worker_plan_from_choices(config, expected, provider=provider)
    selected = planning_provider(provider)
    selected.validate_worker(plan=plan, native_runner=runner)
    required = plan.required_services
    device = getattr(runner, "device", None)
    created: list[Any] = []
    binding: BackendBinding | None = None
    try:
        adapter = selected.create_model_runner_adapter(plan=plan, native_runner=runner)
        created.append(adapter)
        if adapter is None:
            raise RuntimeError("Backend did not bind the required runner adapter")
        for phase in getattr(type(runner), "_gr_runner_methods", ()):
            if not callable(getattr(adapter, phase, None)):
                raise RuntimeError(f"Backend runner adapter does not implement {phase}()")
        attention = None
        if "attention" in required:
            attention = selected.create_attention_executor(plan=plan, device=device)
            created.append(attention)
            if attention is None:
                raise RuntimeError("Backend did not bind the required attention executor")
        constraint = None
        canonical = None
        if plan.constraint_mode == "legacy":
            constraint = getattr(runner, "_gr_constraint_execution", None)
            if constraint is not None:
                if getattr(runner, "_gr_constraint_plan_fingerprint", None) != plan.fingerprint:
                    raise RuntimeError(
                        "Preloaded constraint execution belongs to another backend plan"
                    )
            else:
                constraint = constraint_execution_override(device=device)
                if constraint is None:
                    constraint = selected.create_constrained_topk_execution(
                        plan=plan, device=device
                    )
            created.append(constraint)
            if constraint is None or not constraint.is_available():
                raise RuntimeError("Backend constraint execution is unavailable")
        elif plan.constraint_mode == "canonical":
            # Canonical artifact ownership stays with the existing loader until
            # #714. Its load/warmup must already have completed successfully.
            canonical = getattr(runner, "constraint_backend", None)
            if canonical is None:
                raise RuntimeError("Required canonical constraint backend was not loaded")
        layers = _beam_attention_layers(runner) if attention is not None else ()
        for impl in layers:
            impl.validate_execution(attention)
        stage = None
        if isinstance(adapter, NativeRunnerAdapter):
            installed = getattr(type(runner), "_gr_stage_binding", None)
            expected = plan.choices.to_dict().get("stage")
            if installed is not None and expected is not None:
                if installed.implementation_id != expected:
                    raise RuntimeError("Installed stage differs from the admitted backend plan")
                adapter.bind_stage_sampler(runner, installed)
                stage = installed
        binding = BackendBinding(
            plan, adapter, attention, constraint, _implementation_id(canonical), layers, stage
        )
        validate_required_services(plan, required, binding.report()["services"], "Backend worker")
        for impl in layers:
            impl.execution = attention
        setattr(runner, _BINDING_ATTR, binding)
        return binding
    except BaseException:
        if binding is not None:
            binding.detach_attention()
        preloaded = getattr(runner, "_gr_constraint_execution", None)
        if preloaded is not None and any(service is preloaded for service in created):
            runner._gr_constraint_execution = None
            runner._gr_constraint_plan_fingerprint = None
        _close_services(created)
        raise


def bind_after_model_load(
    original: Any, *, close_runtime: Callable[[Any], None] | None = None
) -> Any:
    if getattr(original, "_gr_binds_backend_after_load", False):
        return original

    @wraps(original)
    def load(self: Any, *args: Any, **kwargs: Any) -> Any:
        existing = getattr(self, _BINDING_ATTR, None)
        if (
            existing is not None
            and existing.stage is not None
            and existing.runner_adapter.pending_stage_sample is not False
        ):
            raise RuntimeError("GR sampling must consume the previous dispatch before model reload")
        if close_runtime is not None:
            # Reject undrained native work before replacing the current generation.
            close_runtime(self)
        if existing is not None:
            # A full load constructs a new model even when its plan is unchanged.
            close_backend_binding(self)
        try:
            from vllm_gr.v1.attention.selector_patch import prepare_engine_attention

            prepare_engine_attention(getattr(self, "vllm_config", None))
            result = original(self, *args, **kwargs)
            bind_backend_services(self)
        except BaseException:
            if close_runtime is not None:
                try:
                    close_runtime(self)
                except Exception:
                    logger.exception("Failed to release native runtime after model-load failure")
            try:
                from vllm_gr.v1.worker.model_runner_patch import (
                    close_constraint_backend,
                )

                close_constraint_backend(self)
            except Exception:
                logger.exception("Failed to release backend resources after model-load failure")
            raise
        return result

    setattr(load, "_gr_binds_backend_after_load", True)
    return load
