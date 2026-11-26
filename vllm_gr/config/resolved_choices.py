# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""The engine's serializable GR choices, independent of frontend placement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .backend_plan import BackendPlan, PlanData, ServiceRequirements, backend_wire

FORMAT_VERSION = 2


@dataclass(frozen=True, slots=True)
class ResolvedChoices:
    backend_id: str
    config_digest: str
    versions: PlanData
    choices: PlanData
    admission_digest: str
    requirements: ServiceRequirements | None
    limits: PlanData

    @classmethod
    def from_plan(cls, plan: BackendPlan) -> ResolvedChoices:
        native = plan.request.native.to_dict()
        model = native.get("model_config", {})
        cache = native.get("cache_config", {})
        return cls(
            plan.backend_id,
            plan.request.config_digest,
            plan.request.compatibility,
            plan.choices,
            plan.fingerprint,
            plan.requirements,
            PlanData(
                {
                    "max_model_len": model.get("max_model_len"),
                    "block_size": cache.get("block_size"),
                    "use_v2_model_runner": native.get("use_v2_model_runner", False),
                    "kv_cache_dtype": cache.get("cache_dtype"),
                }
            ),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "backend_id": self.backend_id,
            "config_digest": self.config_digest,
            "versions": self.versions.to_dict(),
            "choices": self.choices.to_dict(),
            "admission_digest": self.admission_digest,
            "requirements": self.requirements.to_dict() if self.requirements is not None else None,
            "limits": self.limits.to_dict(),
        }

    @classmethod
    def from_wire(cls, value: Any) -> ResolvedChoices:
        if (
            not isinstance(value, Mapping)
            or type(value.get("format_version")) is not int
            or value["format_version"] != FORMAT_VERSION
        ):
            raise ValueError(
                "Unsupported GR choices format; use the same vllm-gr version on all ranks"
            )
        if set(value) != {
            "format_version",
            "backend_id",
            "config_digest",
            "versions",
            "choices",
            "admission_digest",
            "requirements",
            "limits",
        }:
            raise ValueError("Invalid GR choices fields")
        backend, digest = value["backend_id"], value["config_digest"]
        if (
            not isinstance(backend, str)
            or not backend
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("Invalid GR choices identity or config digest")
        admission_digest = value["admission_digest"]
        if not isinstance(admission_digest, str) or len(admission_digest) != 64:
            raise ValueError("Invalid GR admission digest")
        requirements = value["requirements"]
        limits = value["limits"]
        if not isinstance(limits, Mapping) or set(limits) != {
            "max_model_len",
            "block_size",
            "kv_cache_dtype",
            "use_v2_model_runner",
        }:
            raise ValueError("Invalid GR admission limits")
        return cls(
            backend,
            digest,
            PlanData(value["versions"]),
            PlanData(value["choices"]),
            admission_digest,
            ServiceRequirements.from_dict(requirements) if requirements is not None else None,
            PlanData(value["limits"]),
        )


def read_resolved_choices(config: Any) -> ResolvedChoices | None:
    wire = backend_wire(config)
    if wire is None:
        return None
    if isinstance(wire, Mapping) and "format_version" in wire:
        return ResolvedChoices.from_wire(wire)
    # The compatibility provider API can still supply a local BackendPlan envelope.
    return ResolvedChoices.from_plan(BackendPlan.from_wire(wire))
