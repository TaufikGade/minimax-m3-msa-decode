# SPDX-License-Identifier: Apache-2.0
"""Evidence-bounded backend policy for MiniMax M3 sparse decode.

This module deliberately stays independent of the pinned vLLM snapshot. It
encodes only crossover decisions demonstrated by this repository's B300
measurements; unsupported or unmeasured cases fall back to Triton.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DecodeBackend(str, Enum):
    TRITON = "triton"
    CUTLASS = "cutlass"


class DispatchReason(str, Enum):
    STABLE_CUTLASS_WIN = "stable_cutlass_win"
    BELOW_MEASURED_CROSSOVER = "below_measured_crossover"
    UNSUPPORTED_SCALE_MODE = "unsupported_scale_mode"
    UNMEASURED_QUERY_LENGTH = "unmeasured_query_length"
    UNMEASURED_HEAD_GEOMETRY = "unmeasured_head_geometry"


@dataclass(frozen=True)
class DispatchDecision:
    backend: DecodeBackend
    reason: DispatchReason
    cutlass_min_batch: int | None


# Conservative thresholds from ten independent-process CUDA Graph runs on the
# same B300. Both geometries have GQA ratio 16, so the absolute head geometry
# must remain part of the key.
_SCALAR_DECODE_THRESHOLDS = {
    (64, 4): 16,  # TP1-like
    (16, 1): 64,  # TP4-like
}


def select_decode_backend(
    *,
    batch_size: int,
    decode_query_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    scale_mode: str,
) -> DispatchDecision:
    """Select a backend without extrapolating beyond measured shapes.

    The CUTLASS comparison currently supports scalar FP8 KV scales only. The
    crossover evidence covers one-token decode and the exact TP1-like and
    TP4-like geometries listed above.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if decode_query_len < 1:
        raise ValueError("decode_query_len must be positive")
    if num_q_heads < 1 or num_kv_heads < 1:
        raise ValueError("head counts must be positive")

    if scale_mode != "scalar":
        return DispatchDecision(
            DecodeBackend.TRITON,
            DispatchReason.UNSUPPORTED_SCALE_MODE,
            None,
        )

    if decode_query_len != 1:
        return DispatchDecision(
            DecodeBackend.TRITON,
            DispatchReason.UNMEASURED_QUERY_LENGTH,
            None,
        )

    threshold = _SCALAR_DECODE_THRESHOLDS.get((num_q_heads, num_kv_heads))
    if threshold is None:
        return DispatchDecision(
            DecodeBackend.TRITON,
            DispatchReason.UNMEASURED_HEAD_GEOMETRY,
            None,
        )

    if batch_size < threshold:
        return DispatchDecision(
            DecodeBackend.TRITON,
            DispatchReason.BELOW_MEASURED_CROSSOVER,
            threshold,
        )

    return DispatchDecision(
        DecodeBackend.CUTLASS,
        DispatchReason.STABLE_CUTLASS_WIN,
        threshold,
    )


def should_prepare_cutlass_metadata(
    *,
    batch_size: int,
    decode_query_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    scale_mode: str,
) -> bool:
    """Return whether the evidence-bounded policy selects CUTLASS."""
    decision = select_decode_backend(
        batch_size=batch_size,
        decode_query_len=decode_query_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        scale_mode=scale_mode,
    )
    return decision.backend is DecodeBackend.CUTLASS
