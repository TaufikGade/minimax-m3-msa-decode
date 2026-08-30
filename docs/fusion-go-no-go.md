# Split-K fusion go/no-go memo

## Decision

**No-go for a functional fused kernel at the current evidence level.**

**Go only for a bounded resource-feasibility study of the scalar-scale path.**
The study may specify the cross-CTA reduction protocol and compile a
non-functional/resource-only skeleton, but it must not proceed to a complete
attention implementation until the occupancy and spill gates below pass.

The per-token-scale path is a separate no-go for direct same-program fusion:
its partial specialization already uses 255 registers per thread. It may only
join the functional prototype after a skeleton demonstrates zero local-memory
spills and preserves the measured batch-16 wave count.

## Evidence matrix

| Question | Scalar TP1 | Scalar TP4 | Per-token TP1/TP4 | Decision impact |
|---|---|---|---|---|
| Workspace reaches DRAM? | Yes: merge estimate matches 0.26/1.02 MiB order | Yes: 0.07/1.02 MiB order | Yes; scale adds only 3--4% DRAM bytes | Pass |
| Recoverable Graph cost? | `full-partial` 2.02 us at b1, 4.08 us at b16 | 1.97--2.08 us at b1/8/16 | eliminate-merge bound 15--29% | Pass as an upper bound |
| Upper bound exceeds 10%? | 19.63% b1; 14.33% b16 | 19.58%, 16.97%, 14.73% | 15.48--28.73% | Pass |
| Comfortable registers? | No proof: 195 b1, 168 b16 | Same 195/168 | No: 255 | Fail/unknown |
| Preserve CTA count? | No design demonstrated | Critical: only 16 CTAs at b1 | No design demonstrated | Fail/unknown |
| Preserve occupancy/waves? | No design demonstrated | No design demonstrated | Especially risky despite lower smem | Fail/unknown |
| Existing alternative at larger batch? | CUTLASS wins from b16 | CUTLASS wins from b64 | CUTLASS control only covers scalar scale | Narrows fusion scope |

The performance prerequisites are present, but the resource and parallelism
prerequisites are not. Therefore the data support design work, not a functional
kernel.

## What the merge timing means

Standalone merge or instrumented kernel time must not be used as the expected
fusion gain. Nsight Compute reports 6.7--9.7 us for the profiled merge kernels,
and isolated Graph merge is about 6.8 us, but the paired incremental cost in a
full Graph is only 2.0--4.1 us.

The paired `full - partial` value is the credible end-to-end upper bound because
it removes the common Graph replay and scheduling floor. Even that bound is
not fully recoverable: a correct output still needs cross-chunk max/sum/output
reduction and the final output write.

The earlier observation of an approximately 11.5 us merge fraction should
therefore be treated as instrumentation/standalone latency, not a serving-path
speedup prediction.

## Why a direct fusion is structurally difficult

The current partial grid assigns independent CTAs to split chunks, while merge
combines their online-softmax state. A single CUDA kernel cannot simply append
the merge code to each producer CTA because the reduction crosses CTA
boundaries.

The obvious design families all require evidence before implementation:

1. **One CTA owns all chunks.** This removes the cross-CTA barrier but reduces
   the partial grid by the chunk factor. It is rejected for the feasibility
   study because batch-1 TP1 already launches only 64 CTAs on 148 SMs, and TP4
   launches only 16.
2. **Cooperative/persistent grid synchronization.** This can preserve logical
   producers but constrains the grid to simultaneously resident CTAs. Current
   73.7 KiB shared memory and 168--195 registers already limit residency, so a
   residency proof is required before considering it.
3. **Cluster or distributed shared-memory reduction.** This could keep several
   producer CTAs together, but the required cluster shape, chunk mapping, and
   Triton/SM103 lowering have not been demonstrated. It remains a paper design.
4. **Global workspace plus last-producer reduction.** This can remove the
   second launch but retains most workspace traffic and adds counters/fences.
   It is only worthwhile if a microbenchmark proves that launch recovery alone
   exceeds the synchronization cost.

No family currently proves both unchanged CTA parallelism and unchanged
occupancy. A functional implementation would be premature.

## Scope after the CUTLASS crossover

Fusion is only relevant where Triton remains the selected backend:

- TP1-like 64/4 heads: principally batches below 16;
- TP4-like 16/1 heads: principally batches below 64;
- per-token FP8 scale: all thresholds remain unproven for CUTLASS and must not
  inherit the scalar dispatch policy.

This makes batch-1 TP4 the mandatory worst-case acceptance shape: it has only
16 partial CTAs, so any design that reduces producer CTA count is immediately
disqualified even if larger batches improve.

## Resource-only skeleton gates

A bounded skeleton is allowed only to answer resource questions. It should
contain the proposed state, synchronization objects, and launch geometry, but
not a complete attention solution. Continue to a functional prototype only if
all gates pass for scalar and per-token builds:

1. **No spills:** generated code and NCU show no local-memory load/store traffic.
2. **Registers:** per-token does not exceed the architectural 255-register
   shape or trigger compiler spilling; scalar does not materially reduce
   resident CTAs relative to 195/168 registers.
3. **Shared memory:** scalar stays within the current 73.7 KiB envelope and
   per-token within its 42.0 KiB envelope, unless occupancy is independently
   shown unchanged.
4. **Parallelism:** producer CTA count is not lower than the current partial
   grid for batch-1 TP1 and TP4.
5. **Reduction tail:** if fewer CTAs execute the final reduction than the
   current `(total_q, num_heads)` merge grid, the added tail must remain below
   the 5% full-Graph regression threshold on batch-1 TP4.
6. **Waves:** batch-16 waves/SM are at least the measured scalar 0.58 and
   per-token 0.86 values.
7. **No global-residency assumption:** launch correctness must not depend on
   more CTAs being simultaneously resident than resource limits allow.

Failure of any gate ends the fusion study with a negative result.

## Functional prototype acceptance criteria

If the skeleton passes, a later functional prototype must satisfy all of the
following in the same Slurm allocation as its baseline:

- correctness against the FP32 gathered-KV reference for scalar and varying
  per-token physical-page scales;
- TP1 and TP4, batches 1/8/16, upstream default chunk policy;
- eager and CUDA Graph measurements with 10 independent processes, 100
  warmups, and 500 iterations;
- stable full-Graph improvement
  `Delta > max(5%, 2 * run-to-run CV)`;
- no regression above 5% on batch-1 TP4;
- NCU confirmation of zero spills and preserved CTA/wave behavior.

If the prototype cannot exceed 5% full-Graph improvement, the final conclusion
should be negative: the 14--29% eliminate-merge bound is not practically
recoverable without sacrificing resource shape or parallelism, and CUTLASS
already covers the larger-batch regime.

## Recommended next action

Do not modify the production attention implementation. First write a one-page
design note choosing exactly one cross-CTA protocol and calculate its CTA,
cluster/cooperative-residency, shared-memory, and register-state requirements
for these four mandatory shapes:

- TP1 scalar, batch 1 / chunks 16;
- TP1 per-token, batch 16 / chunks 4;
- TP4 scalar, batch 1 / chunks 16;
- TP4 per-token, batch 16 / chunks 16.

Only after that arithmetic fits the resource gates should a resource-only
skeleton be compiled.

The first protocol analysis is recorded in
`docs/fusion-last-producer-design.md`.
