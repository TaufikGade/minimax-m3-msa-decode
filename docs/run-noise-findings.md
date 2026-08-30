# B300 Run-to-Run Noise Findings

## Scope and provenance

- Slurm job: 11862
- Measurement code commit: `2461be43e232312163018c4cad9183d8c7acf096`
- Baseline kernel tag: `baseline-b300-split-sweep-v1` at `617aa68`
- GPU: NVIDIA B300 SXM6 AC
- GPU UUID: `GPU-3924bd78-8b2f-8e85-3f20-8b0687d0bb0a`
- Driver: 580.126.09
- KV cache: FP8 E4M3 with scalar scales
- Shape: 64 query heads / 4 KV heads, sequence length 2048
- Mode: CUDA Graph replay
- Independent runs: 10 per batch
- Per run: 100 warm-up launches and 500 measured launches

Each run used a fresh Python process and therefore rebuilt its inputs,
workspace, and CUDA Graph. Odd trials measured batches in the order 1/4/8/16;
even trials used 16/8/4/1 to reduce correlation between batch and thermal state.

## Results

The coefficient of variation is calculated across the ten per-run medians
using the sample standard deviation:

\[
CV = \frac{s(\text{run medians})}{\operatorname{mean}(\text{run medians})}.
\]

| Batch | Chunks | Mean median | Sample std | CV | Required speedup | Min–max median |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16 | 10.547 us | 0.079 us | 0.746% | 5.0% | 10.464–10.688 us |
| 4 | 16 | 14.664 us | 0.043 us | 0.297% | 5.0% | 14.560–14.704 us |
| 8 | 8 | 20.725 us | 0.058 us | 0.282% | 5.0% | 20.640–20.800 us |
| 16 | 4 | 28.973 us | 0.048 us | 0.166% | 5.0% | 28.896–29.056 us |

For every batch, `2CV` is below the predeclared 5% floor. A candidate is
therefore called a stable improvement only when its same-job speedup exceeds
5% and the direction is consistent across independent, interleaved trials.

The machine-readable summary is
`results/raw/run-noise-2461be43e232-11862.csv`. Its SHA-256 is
`36e0cddadb1efba2e4a0d566663a047eb53b2a6210f086704f1bcdcec059e1dd`.
The ten input CSV files are under
`results/raw/noise/msa-2461be43e232-11862/`.

## Comparability limitation

The NCU jobs used GPU UUID `GPU-778768b4-6c9e-e483-890e-0812760948ae`, while
this timing job used UUID `GPU-3924bd78-8b2f-8e85-3f20-8b0687d0bb0a`.
The older `baseline-smoke.csv` also reports materially higher absolute Graph
latencies but does not record a GPU UUID. These results must not be combined
into a cross-machine speedup claim.

NCU remains valid for explaining kernel behavior, but every future performance
claim must measure baseline and candidate in the same Slurm allocation, on the
same GPU, using interleaved or randomized order. The values above establish a
noise floor and a current baseline range; they are not a permanent absolute
latency reference for a different B300.
