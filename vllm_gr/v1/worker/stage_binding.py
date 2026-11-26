# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Stage operations selected when the native runner binds its engine plan."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageBinding:
    """Stateless operations for one runner implementation.

    Execute and retire receive worker-owned scheduler data. A None execute
    result requires a subsequent sample on the same runner. Retirement never
    invokes the model. Cancel is idempotent and capture uses the execute
    implementation's graph policy. Session state belongs to the native runner.
    """

    implementation_id: str
    execute: Callable[..., Any]
    sample: Callable[[Any, Any], Any]
    retire: Callable[[Any, Any], Any]
    cancel_finished: Callable[[Any, Collection[str]], None]
    capture_startup: Callable[[Any, bool], bool]
    graph_status: Callable[[Any], dict[str, Any]] | None = None


def bound_stage(worker: Any) -> StageBinding | None:
    binding = getattr(worker, "_gr_backend_binding", None)
    return getattr(binding, "stage", None)


def require_stage(worker: Any) -> StageBinding:
    """A GR control frame requires a stage owned by this loaded model."""
    stage = bound_stage(worker)
    if stage is None:
        raise RuntimeError("GR stage received before the runner bound its backend plan")
    return stage
