# SPDX-License-Identifier: Apache-2.0
"""Launch exactly one sparse-decode kernel inside an NCU capture range."""

from __future__ import annotations

import argparse
import json

import torch

from msa_harness import (
    allocate_workspace,
    default_num_chunks,
    launch_triton_baseline,
    launch_triton_merge,
    launch_triton_partial,
    make_case,
    reference_decode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warm up and validate one decode shape, then launch only the selected "
            "kernel between cudaProfilerStart/Stop calls."
        )
    )
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--kernel", choices=("partial", "merge"), required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--profile-launches", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=64)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--no-pdl", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch < 1:
        raise ValueError("batch must be positive")
    if args.warmup < 1:
        raise ValueError("warmup must be positive")
    if args.profile_launches < 1:
        raise ValueError("profile-launches must be positive")
    if args.num_heads < 1 or args.num_kv_heads < 1:
        raise ValueError("head counts must be positive")
    if args.num_heads % args.num_kv_heads:
        raise ValueError("num-heads must be divisible by num-kv-heads")


def check_correctness(case, workspace) -> dict[str, float]:
    expected = reference_decode(case)
    actual = launch_triton_baseline(case, workspace).clone()
    torch.cuda.synchronize()

    if not torch.isfinite(actual).all():
        raise AssertionError("baseline produced NaN or Inf")
    tolerance = 2e-2 if case.kv_cache.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ) else 1e-2
    torch.testing.assert_close(
        actual, expected, rtol=tolerance, atol=tolerance
    )
    difference = actual.float() - expected.float()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().flatten(), expected.float().flatten(), dim=0
    ).item()
    return {
        "max_abs_error": difference.abs().max().item(),
        "cosine_similarity": cosine,
    }


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    use_pdl = not args.no_pdl
    case = make_case(
        args.batch,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        fp8=not args.bf16,
        seed=args.seed,
    )
    selected_chunks = args.chunks
    if selected_chunks is None:
        selected_chunks = default_num_chunks(case)
    workspace = allocate_workspace(case, selected_chunks)

    correctness = check_correctness(case, workspace)

    if args.kernel == "partial":
        target = lambda: launch_triton_partial(  # noqa: E731
            case, workspace, use_pdl=use_pdl
        )
    else:
        # Populate valid partial/LSE inputs once. Repeated merge launches only
        # overwrite the output and leave those inputs unchanged.
        launch_triton_partial(case, workspace, use_pdl=use_pdl)
        target = lambda: launch_triton_merge(  # noqa: E731
            case, workspace, use_pdl=use_pdl
        )

    for _ in range(args.warmup):
        target()
    torch.cuda.synchronize()

    manifest = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "batch": args.batch,
        "num_heads": args.num_heads,
        "num_kv_heads": args.num_kv_heads,
        "kv_dtype": str(case.kv_cache.dtype),
        "scale_mode": "scalar" if not args.bf16 else "none",
        "page_layout": "random",
        "chunks": workspace.num_chunks,
        "kernel": args.kernel,
        "warmup": args.warmup,
        "profile_launches": args.profile_launches,
        "seed": args.seed,
        "use_pdl": use_pdl,
        **correctness,
    }
    print(json.dumps(manifest, sort_keys=True), flush=True)

    torch.cuda.cudart().cudaProfilerStart()
    torch.cuda.nvtx.range_push(f"msa_decode_{args.kernel}")
    try:
        for _ in range(args.profile_launches):
            target()
        torch.cuda.synchronize()
    finally:
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
