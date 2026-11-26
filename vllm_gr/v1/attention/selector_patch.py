# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-GR project

"""Hardware-neutral CUSTOM attention selection for GR runtime initialization."""

from threading import Lock
from typing import Any

from vllm_gr.config import GRAttentionBackend, GRConfig

_BEAM_BACKEND_PATH = "vllm_gr.v1.attention.backends.beam_attn.BeamAttentionBackend"
_REGISTRATION_LOCK = Lock()


def _get_current_gr_attention_backend() -> GRAttentionBackend | None:
    from vllm.config import get_current_vllm_config_or_none

    return GRConfig.from_vllm_config(get_current_vllm_config_or_none()).attention_backend


def _register_beam_backend() -> None:
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    custom = AttentionBackendEnum.CUSTOM
    with _REGISTRATION_LOCK:
        if custom.is_overridden():
            owner = custom.get_path()
            if owner != _BEAM_BACKEND_PATH:
                raise RuntimeError(
                    f"GR Beam requires CUSTOM attention, already registered to {owner}. "
                    "Load these attention backends in separate processes."
                )
        else:
            register_backend(custom, _BEAM_BACKEND_PATH)


def prepare_engine_attention(config: Any) -> None:
    """Claim the process-local slot before a Beam engine constructs its layers."""
    if GRConfig.from_vllm_config(config).attention_backend is not GRAttentionBackend.CUSTOM:
        return
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if (
        getattr(getattr(config, "attention_config", None), "backend", None)
        is not AttentionBackendEnum.CUSTOM
    ):
        raise ValueError("GR CUSTOM attention must be selected before native attention selection")
    _register_beam_backend()
    apply_custom_attention_backend_patch()


def clear_attention_backend_cache() -> None:
    from vllm.v1.attention.selector import _cached_get_attn_backend

    _cached_get_attn_backend.cache_clear()


def apply_custom_attention_backend_patch(force: bool = False) -> int:
    """Install the bridge on the selected vendor platform when needed."""
    from vllm.platforms import current_platform

    from vllm_gr.platforms.identity import platform_identity

    del force
    if not platform_identity(current_platform).has_lineage("vllm_ascend.platform.NPUPlatform"):
        return 0
    from vllm_gr.platforms.ascend.attention_bridge import install

    return install(type(current_platform))
