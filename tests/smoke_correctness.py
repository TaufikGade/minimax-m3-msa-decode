# SPDX-License-Identifier: Apache-2.0
"""Dependency-free GPU correctness smoke test."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from msa_harness import (  # noqa: E402
    allocate_workspace,
    launch_triton_baseline,
    make_case,
    reference_decode,
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    for batch in (1, 4):
        for fp8 in (False, True):
            case = make_case(batch, fp8=fp8)
            workspace = allocate_workspace(case)
            actual = launch_triton_baseline(case, workspace).clone()
            expected = reference_decode(case)
            if not torch.isfinite(actual).all():
                raise AssertionError(f"non-finite output: batch={batch}, fp8={fp8}")
            tolerance = 2e-2 if fp8 else 1e-2
            torch.testing.assert_close(
                actual, expected, rtol=tolerance, atol=tolerance
            )
            cosine = torch.nn.functional.cosine_similarity(
                actual.float().flatten(), expected.float().flatten(), dim=0
            ).item()
            if cosine <= 0.999:
                raise AssertionError(
                    f"cosine={cosine}: batch={batch}, fp8={fp8}"
                )
            max_abs = (actual.float() - expected.float()).abs().max().item()
            print(
                f"PASS batch={batch} fp8={fp8} "
                f"max_abs={max_abs:.6g} cosine={cosine:.8f}"
            )


if __name__ == "__main__":
    main()
