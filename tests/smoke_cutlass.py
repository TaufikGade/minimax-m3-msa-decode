# SPDX-License-Identifier: Apache-2.0
"""JIT and correctness smoke for the standalone CUTLASS decode path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

from cutlass_harness import make_cutlass_case, launch_cutlass_full  # noqa: E402
from msa_harness import launch_triton_baseline, reference_decode  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--effective-kv-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    max_abs = (actual.float() - expected.float()).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    return max_abs, cosine


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    case = make_cutlass_case(
        args.batch,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        effective_kv_len=args.effective_kv_len,
        seed=args.seed,
    )
    expected = reference_decode(case.baseline)

    triton_out = launch_triton_baseline(
        case.baseline, case.baseline_workspace
    ).clone()
    cutlass_out = launch_cutlass_full(case).clone()
    torch.cuda.synchronize()

    torch.testing.assert_close(triton_out, expected, rtol=2e-2, atol=2e-2)
    # Q is FP8 only on the CUTLASS path, so allow its additional quantization.
    torch.testing.assert_close(cutlass_out, expected, rtol=5e-2, atol=5e-2)
    triton_max_abs, triton_cosine = metrics(triton_out, expected)
    cutlass_max_abs, cutlass_cosine = metrics(cutlass_out, expected)
    if cutlass_cosine <= 0.995:
        raise AssertionError(f"CUTLASS cosine similarity too low: {cutlass_cosine}")

    print(
        "PASS "
        f"batch={args.batch} heads={args.num_heads}/{args.num_kv_heads} "
        f"effective_kv_len={args.effective_kv_len} "
        f"q_scale={case.query_scale:.9g} "
        f"triton_max_abs={triton_max_abs:.6g} "
        f"triton_cosine={triton_cosine:.8f} "
        f"cutlass_max_abs={cutlass_max_abs:.6g} "
        f"cutlass_cosine={cutlass_cosine:.8f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
