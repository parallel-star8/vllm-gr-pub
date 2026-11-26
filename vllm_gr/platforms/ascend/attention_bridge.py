# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project
"""Preserve an explicit Beam selection in the Ascend platform selector."""

from __future__ import annotations

from functools import wraps
from typing import Any


def install(platform: type[Any]) -> int:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # Bind the original descriptor to the caller's subclass on delegation.
    original = getattr(platform, "get_attn_backend_cls").__func__
    if vars(original).get("_gr_ascend_custom") is True:
        return 0

    @wraps(original)
    def select(
        cls: type, selected_backend: Any, attn_selector_config: Any, num_heads: int | None = None
    ) -> str:
        if selected_backend is AttentionBackendEnum.CUSTOM:
            return selected_backend.get_path()
        return original(cls, selected_backend, attn_selector_config, num_heads=num_heads)

    setattr(select, "_gr_ascend_custom", True)
    setattr(platform, "get_attn_backend_cls", classmethod(select))
    return 1
