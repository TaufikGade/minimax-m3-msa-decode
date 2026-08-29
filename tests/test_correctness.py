# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from msa_harness import (  # noqa: E402
    allocate_workspace,
    launch_triton_baseline,
    make_case,
    reference_decode,
)


@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("fp8", [False, True])
def test_triton_matches_fp32_reference(batch: int, fp8: bool) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    case = make_case(batch, fp8=fp8)
    workspace = allocate_workspace(case)
    actual = launch_triton_baseline(case, workspace).clone()
    expected = reference_decode(case)

    assert torch.isfinite(actual).all()
    tolerance = 2e-2 if fp8 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    )
    assert cosine.item() > 0.999
