# Last-producer resource skeleton findings

## Scope and provenance

- measurement code commit: `e1682106c3de`
- Slurm job: `13131`
- GPU: NVIDIA B300 SXM6 AC
- GPU UUID: `GPU-3924bd78-8b2f-8e85-3f20-8b0687d0bb0a`
- Triton: 3.6.0
- correctness stress: 1000 CUDA Graph replays per shape
- timing: 100 warmups and 500 Graph replays

The skeleton contains synthetic partial/LSE production, the proposed
device-scope completion counter, a streamed 16-head online merge in the last
producer CTA, output write, and in-kernel counter reset. It contains no
attention dot products and is not connected to the production path.

## Synchronization result

All four shapes passed correctness after eager execution and 1000 consecutive
Graph replays. Every counter returned to zero. PTX contains the intended
GPU-scope acquire-release completion atomic and release reset:

```text
atom.global.gpu.acq_rel.add.u32
atom.global.gpu.release.exch.b32
```

This establishes that Triton 3.6 can express and lower the required memory
ordering on SM103 for the synthetic protocol.

## Resource result

| Shape | Full registers | Producer registers | Local spill words/thread | Dynamic smem |
|---|---:|---:|---:|---:|
| TP1 scalar b1/c16 | 32 | 20 | 0 | 4 B |
| TP1 per-token b16/c4 | 32 | 24 | 0 | 4 B |
| TP4 scalar b1/c16 | 32 | 20 | 0 | 4 B |
| TP4 per-token b16/c16 | 32 | 24 | 0 | 4 B |

These numbers validate only the synthetic state machine. They do not prove
that the real 195/168/255-register partial kernel can absorb the reducer state;
the attention dot-product pipeline is intentionally absent.

## Paired reduction-tail result

Each full skeleton Graph was paired with the same synthetic producer compiled
without atomic or reducer code.

| Shape | Producer only | Last-producer full | Added tail | Tail share of full |
|---|---:|---:|---:|---:|
| TP1 scalar b1/c16, 4 reducers | 6.496 us | 16.192 us | 9.696 us | 59.88% |
| TP1 per-token b16/c4, 64 reducers | 6.304 us | 9.568 us | 3.264 us | 34.11% |
| TP4 scalar b1/c16, 1 reducer | 6.736 us | 13.888 us | 7.152 us | 51.50% |
| TP4 per-token b16/c16, 16 reducers | 6.592 us | 15.968 us | 9.376 us | 58.72% |

The small-batch scalar tails are substantially larger than the real two-kernel
paired `full - partial` budgets:

- TP1 batch 1: 9.696 us synthetic tail versus 2.018 us recoverable bound;
- TP4 batch 1: 7.152 us synthetic tail versus 1.970 us recoverable bound.

The synthetic and attention kernels are not added or subtracted as absolute
latencies. The comparison is a gate: collapsing 16 merge CTAs per GQA group to
one last reducer already costs several times the entire measured recovery
budget before attention register pressure is introduced.

## Decision

**Reject the last-producer protocol and stop the fusion implementation.**

It passes memory ordering, Graph reset, correctness, and synthetic spill
checks. It fails the mandatory reduction-tail gate because final-reduction CTA
parallelism falls by 16x. Integrating the real attention pipeline cannot remove
that structural tail and would additionally expose the 195--255 register risk.

The production Triton kernel remains unchanged. Larger scalar batches should
use the measured CUTLASS crossover policy instead of pursuing this fusion.

## Raw results

- `results/raw/last-producer-e1682106c3de-tp1-scalar-b1-c16-13131_0.json`
- `results/raw/last-producer-e1682106c3de-tp1-pertoken-b16-c4-13131_1.json`
- `results/raw/last-producer-e1682106c3de-tp4-scalar-b1-c16-13131_2.json`
- `results/raw/last-producer-e1682106c3de-tp4-pertoken-b16-c16-13131_3.json`
