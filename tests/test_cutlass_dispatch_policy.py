# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))

from cutlass_dispatch_policy import (  # noqa: E402
    DecodeBackend,
    DispatchReason,
    select_decode_backend,
    should_prepare_cutlass_metadata,
)


@pytest.mark.parametrize(
    ("num_q_heads", "num_kv_heads", "batch", "expected"),
    [
        (64, 4, 1, DecodeBackend.TRITON),
        (64, 4, 8, DecodeBackend.TRITON),
        (64, 4, 15, DecodeBackend.TRITON),
        (64, 4, 16, DecodeBackend.CUTLASS),
        (64, 4, 32, DecodeBackend.CUTLASS),
        (16, 1, 16, DecodeBackend.TRITON),
        (16, 1, 32, DecodeBackend.TRITON),
        (16, 1, 63, DecodeBackend.TRITON),
        (16, 1, 64, DecodeBackend.CUTLASS),
    ],
)
def test_scalar_thresholds_are_geometry_aware(
    num_q_heads: int,
    num_kv_heads: int,
    batch: int,
    expected: DecodeBackend,
) -> None:
    decision = select_decode_backend(
        batch_size=batch,
        decode_query_len=1,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        scale_mode="scalar",
    )
    assert decision.backend is expected


def test_per_token_scale_stays_on_triton() -> None:
    decision = select_decode_backend(
        batch_size=128,
        decode_query_len=1,
        num_q_heads=64,
        num_kv_heads=4,
        scale_mode="per_token_head",
    )
    assert decision.backend is DecodeBackend.TRITON
    assert decision.reason is DispatchReason.UNSUPPORTED_SCALE_MODE
    assert decision.cutlass_min_batch is None


@pytest.mark.parametrize(
    ("decode_query_len", "num_q_heads", "num_kv_heads", "reason"),
    [
        (2, 64, 4, DispatchReason.UNMEASURED_QUERY_LENGTH),
        (1, 32, 2, DispatchReason.UNMEASURED_HEAD_GEOMETRY),
    ],
)
def test_unmeasured_shapes_do_not_extrapolate(
    decode_query_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    reason: DispatchReason,
) -> None:
    decision = select_decode_backend(
        batch_size=128,
        decode_query_len=decode_query_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        scale_mode="scalar",
    )
    assert decision.backend is DecodeBackend.TRITON
    assert decision.reason is reason


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("batch_size", 0),
        ("decode_query_len", 0),
        ("num_q_heads", 0),
        ("num_kv_heads", 0),
    ],
)
def test_invalid_dimensions_are_rejected(field: str, value: int) -> None:
    kwargs = {
        "batch_size": 16,
        "decode_query_len": 1,
        "num_q_heads": 64,
        "num_kv_heads": 4,
        "scale_mode": "scalar",
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        select_decode_backend(**kwargs)


def test_metadata_helper_matches_backend_decision() -> None:
    assert should_prepare_cutlass_metadata(
        batch_size=16,
        decode_query_len=1,
        num_q_heads=64,
        num_kv_heads=4,
        scale_mode="scalar",
    )
    assert not should_prepare_cutlass_metadata(
        batch_size=16,
        decode_query_len=1,
        num_q_heads=16,
        num_kv_heads=1,
        scale_mode="scalar",
    )
