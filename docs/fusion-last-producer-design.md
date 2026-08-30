# Last-producer single-launch design note

## Status

This is a paper/resource design, not a kernel implementation.

The selected protocol preserves the current split-K producer grid and lets the
last producer CTA for each `(query token, KV head)` perform the cross-chunk
online-softmax reduction. It is the least invasive single-launch candidate,
but it retains the global partial/LSE workspace. It must therefore be described
as launch fusion, not workspace-eliminating fusion.

## Protocol

For every `(query token, KV head)` group:

1. Every existing chunk CTA computes and stores its partial output and LSE to
   the current global workspace.
2. After a block barrier, one designated lane performs a device-scope release
   fence and atomically increments a group completion counter.
3. The old counter value identifies whether this CTA is the last producer.
   Non-last CTAs return; they never wait or spin.
4. The last producer obtains acquire visibility, reloads all chunk partials for
   its 16-query-head GQA group, performs the online-softmax merge, writes final
   output, and resets the group counter to zero.
5. Kernel completion orders the reset before the next CUDA Graph replay using
   the same stable counter address.

The no-spin property is mandatory. CUDA does not guarantee that arbitrary
thread blocks are scheduled while resident blocks wait for them. The completion
counter follows NVIDIA's documented last-block reduction pattern: partial data
must become visible before the completion signal, using device-scope ordering
and matching visibility semantics. See the
[CUDA Programming Guide last-block reduction example](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html).

Triton 3.6 lowering must be inspected before implementation. This note does
not assume that a default `atomic_add` alone supplies the required release and
acquire semantics.

## Shape arithmetic

The current producer grid is
`total_q * chunks * num_kv_heads`. The current merge grid is
`total_q * num_query_heads`. The proposed last-producer count is
`total_q * num_kv_heads`.

Workspace bytes are:

```text
partial = chunks * total_q * query_heads * 128 * 2
LSE     = chunks * total_q * query_heads * 4
counter = total_q * kv_heads * 4
```

| Mandatory shape | Producer CTAs | Current merge CTAs | Last reducers | Partial + LSE | Counters | Reg/thread | Dynamic smem | Waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP1 scalar, b1/c16, 64/4 | 64 | 64 | 4 | 266,240 B | 16 B | 195 | 73.7 KiB | 0.22 |
| TP1 per-token, b16/c4, 64/4 | 256 | 1,024 | 64 | 1,064,960 B | 256 B | 255 | 42.0 KiB | 0.86 |
| TP4 scalar, b1/c16, 16/1 | 16 | 16 | 1 | 66,560 B | 4 B | 195 | 73.7 KiB | 0.05 |
| TP4 per-token, b16/c16, 16/1 | 256 | 256 | 16 | 1,064,960 B | 64 B | 255 | 42.0 KiB | 0.86 |

The protocol preserves all producer CTAs, but final-reduction CTA parallelism
falls by the GQA ratio of 16. The last CTA can reduce the 16 query heads as a
block-shaped operation rather than serial scalar heads, yet TP4 batch 1 still
has only one reducer CTA. This is the primary latency risk.

## Resource lifetime

The producer already carries a `[16, 128]` FP32 output accumulator. A viable
lowering must end that accumulator's live range after storing the partial, then
reuse registers/shared memory for the merge loop. The merge should stream over
chunks instead of materializing `[chunks, 16, 128]` in registers.

Expected peak storage is therefore the maximum of producer and reducer state,
not their sum, but this is a compiler hypothesis rather than evidence. It is
plausible for scalar mode and unproven for per-token mode. At 255 registers,
per-token has no margin for counter, predicate, address, or merge-state
liveness mistakes.

The reducer can reuse the producer CTA's dynamic shared-memory allocation only
after all threads complete the producer phase. No shared memory is visible
across CTAs; all cross-CTA state remains in global memory.

## Traffic and benefit bound

This protocol does **not** remove the partial write or merge read. It adds a
small counter/fence transaction and removes only the second kernel launch while
moving merge work onto the last producer's tail.

Its achievable gain is therefore strictly below the measured paired
eliminate-merge bounds:

- scalar TP1: below 19.63% at batch 1 and 14.33% at batch 16;
- scalar TP4: below 19.58%, 16.97%, and 14.73% at batches 1/8/16;
- per-token controls: below the measured 15--29% loose bounds.

Because the mathematical reduction, final output store, and workspace traffic
remain, the realistic recovery may be only the Graph node/launch portion. The
protocol fails immediately if the one/few reducer CTAs add more tail latency
than that launch recovery.

## Paper go/no-go

| Gate | Result | Reason |
|---|---|---|
| Preserve producer CTA count | Pass | Grid is unchanged |
| Avoid global producer waiting | Pass | Non-last CTAs exit |
| Eliminate workspace traffic | Fail | Full partial/LSE workspace remains |
| Remove second launch | Potential pass | Merge executes inside last producer |
| Preserve reduction parallelism | Fail/unknown | 16x fewer reducer CTAs |
| Scalar register feasibility | Unknown | Must compile below spill/residency gate |
| Per-token register feasibility | High risk | Starts at 255 registers/thread |
| Graph-stable state | Conditional | Counter reset and stable address must be proven |

The design is eligible only for a scalar resource-only skeleton. It is not
eligible for a functional prototype, and per-token compilation is diagnostic
only. If scalar compilation increases residency limits or a reduction-tail
microbenchmark regresses TP4 batch 1 by more than 5%, reject this protocol and
stop; do not extend it into the attention kernel.

## Skeleton questions

A resource-only skeleton, if authorized later, must answer exactly these
questions before any attention code is copied:

1. Does the device-scope fence/atomic sequence lower with the intended memory
   semantics on SM103?
2. What are registers, local-memory traffic, dynamic shared memory, and
   resident CTAs for scalar and per-token specializations?
3. Can a streamed 16-head merge loop reuse producer accumulator storage?
4. What is the isolated last-reducer tail for 1, 4, 16, and 64 reducer CTAs?
5. Does resetting counters inside the last CTA remain correct across repeated
   CUDA Graph replays?

Until all five are measured, the project remains no-go for a functional fused
kernel.
