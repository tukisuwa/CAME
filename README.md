<h1 align="center">CAME Optimizer</h1>
<h3 align="center">ACL 2023 Outstanding Paper Award<br/>Confidence-guided Adaptive Memory Efficient Optimization</h3>


This is an official implementation of **CAME** optimizer in the "[Confidence-guided Adaptive Memory Efficient Optimization](https://arxiv.org/abs/2307.02047)". Please cite the paper and star this repo if you find CAME useful. Thanks!

[Paper](https://arxiv.org/abs/2307.02047) | [Twitter](https://twitter.com/ZangweiZheng/status/1680227732788236289) | [Blog](https://zhengzangw.github.io/blogs/came) | [Pypi Package](https://pypi.org/project/came-pytorch/) | [zhihu](https://zhuanlan.zhihu.com/p/643816029)
## Method

In this work, we studied a confidence-guided strategy to reduce the instability of existing memory efficient optimizers. Based on this strategy, we proposed CAME to simultaneously achieve two goals: fast convergence as in traditional adaptive methods, and low memory usage as in memory-efficient methods.

The pseudo code is presented in the figure with difference with Adafactor in blue fonts.

<p align="center">
<img src="assets/came_code.png" alt="CAME optimizer pseudo code" width="50%" />
</p>
<!-- ![CAME_code](assets/came_code.png) -->

## Install 
```
pip install came-pytorch
```

## CUDA 8-bit

This fork includes an optional CUDA extension and several CUDA-oriented modes:
- `CAMECUDA`: recommended **speed-first** optimizer for CUDA training.
  - keeps optimizer state in floating point instead of 8-bit
  - uses the fused fp-state CUDA path for contiguous CUDA 2D tensors
  - typically improves step speed versus baseline `CAME`, at the cost of slightly higher VRAM
- `CAME8bit`: recommended single-entrypoint 8-bit optimizer for real models.
  - CUDA 2D params -> quantized `CAME8bit2D` fast path when available
  - contiguous CUDA 1D params -> generic non-factored CUDA fallback by default
  - small-prefix contiguous CUDA ND factored params -> guarded extension-backed ND path
  - everything else (other LayerNorm / bias cases, Conv, Embedding, CPU params, irregular layouts) -> `CAME8bitFull`
- `CAME8bitMemory`: recommended **memory-first** experimental 8-bit optimizer.
  - keeps the compact 8-bit state layout from `CAME8bitFull`
  - reuses small per-device shared CUDA scratch buffers for common 2D/1D CUDA cases
  - usually lowers persistent optimizer VRAM versus `CAME8bit`, but is slower than `CAMECUDA`
- `CAME8bitFull`: reference all-state 8-bit optimizer.
  - works on CPU or CUDA, and supports arbitrary parameter shapes
  - quantizes `exp_avg` and the second-moment / residual statistics blockwise
  - designed as the correctness baseline for future CUDA/Triton optimization
  - CUDA fallback paths reuse persistent `fp32` buffers and blockwise workspace for both factored and non-factored tensors
  - factored CUDA fallback packs row/col `sq` + `res` state into batched blockwise qstate buffers
  - CUDA fallback blockwise transforms can run through the extension's generic quantize / dequantize kernels
  - non-factored CUDA fallback can now run the full optimizer step through a dedicated extension path
  - 2D factored CUDA fallback now uses the dedicated extension path by default; set `prefer_factored_cuda_ext_path=False` to force the legacy path
  - guarded ND factored CUDA fallback now uses a measured heuristic: always for `prefix_batch <= 4`, and for `prefix_batch <= 8` when the per-slice matrix is at least `48x48`
  - benchmark-only `*_small_nd` modes can still override that small-prefix ND cutoff with `--benchmark-nd-batched-matrix-min-numel` for isolated or mixed threshold studies before changing production heuristics again
  - large-prefix ND chunking remains a separate path for per-slice matrices at least `128x128`
- `CAME8bit2D`: specialized optimizer for CUDA **2D parameters only**
  - all persistent optimizer state is blockwise-quantized
  - `exp_avg` uses the tiled CUDA `int8` format with `absmax` per 16x16 tile
  - factored statistics are updated from quantized state directly inside the CUDA extension
  - `fp32` buffers remain only for factors and reduction scratch

Supported scope:
- sparse gradients are not supported
- `CAME8bit2D` fast path requires `block_size == 256`
- 8-bit state layout is fixed after the first step, so parameter resizing is not supported
- `CAME8bit2D` still uses `fp32` scratch buffers for reductions / factors / kernel temporaries
- CUDA Graph capture is currently intended for the CUDA 2D fast path with contiguous params / grads

Specs / notes:
- `doc/CAME8bit2D_SPEC.md`
- `doc/OPTIMIZATION_NOTES.md`
- `doc/CUDA_8BIT_ROADMAP.md`
- `doc/CUDA_MODE_SPLIT_PROPOSAL.md`
- `doc/CUDA_MODE_IMPLEMENTATION_PLAN.md`
- `doc/FACT_ND_CHUNK_FUSED_PLAN.md`

### Build from source (recommended for Windows)

Prereqs:
- PyTorch with CUDA (e.g. `torch==...+cu12x`)
- NVIDIA CUDA Toolkit installed (same **major** as `torch.version.cuda`; minor mismatch is usually OK)
- Windows: Visual Studio 2022 Build Tools (MSVC C++ toolchain)

If you don't build the extension, you can still use `CAME` (pure PyTorch). The 8-bit variants require the extension.

**Windows (PowerShell)**
```bat
cd path\to\CAME
cmd /d /s /c "<path-to-vcvars64.bat> && set DISTUTILS_USE_SDK=1 && python setup.py build_ext --inplace"
```

Common `vcvars64.bat` locations:
- VS 2022 BuildTools: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`
- VS 2022 Community: `C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat`

Alternatively, open **"x64 Native Tools Command Prompt for VS 2022"** and run:
```bat
cd path\to\CAME
python setup.py build_ext --inplace
```

**Linux**
```bash
cd path/to/CAME
python -m pip install -U pip setuptools wheel
python setup.py build_ext --inplace
```

Tips:
- Set `TORCH_CUDA_ARCH_LIST` to reduce build time (example: `8.6;8.9`).
- If DLL load fails on Windows, ensure the CUDA Toolkit `bin` directory is on the DLL search path (this repo also tries to add common CUDA `bin` locations automatically).

## Usage

```python
from came_pytorch import CAME
optimizer = CAME(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    betas=(0.9, 0.999, 0.9999),
    eps=(1e-30, 1e-16)
)
```

### Example Selection Trend

One local `sd-scripts` SDXL LoRA run with the same 300-step setup produced this trend:

| Optimizer | Time | VRAM |
|-----------|------|------|
| `CAME` | `12:01` | `7.2 GB` |
| `CAMECUDA` | `08:32` | `7.4 GB` |
| `CAME8bit` | `08:47` | `7.3 GB` |
| `CAME8bitMemory` | `09:48` | `7.0 GB` |

Read this as:
- choose `CAMECUDA` when step speed matters most
- choose `CAME8bit` for a balanced speed / memory tradeoff
- choose `CAME8bitMemory` when persistent optimizer VRAM matters most

These numbers are workload- and environment-dependent, but they reflect the intended mode split of this fork.

### Usage (CUDA fp-state)

Speed-first CUDA mode without 8-bit optimizer state compression.
Use this when throughput is the main goal:

```python
from came_pytorch import CAMECUDA

optimizer = CAMECUDA(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
)
```

### Usage (8-bit)

Single-optimizer entrypoint (recommended):
```python
from came_pytorch import CAME8bit
optimizer = CAME8bit(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

Memory-first 8-bit mode.
Use this when persistent optimizer VRAM matters more than raw step speed:
```python
from came_pytorch import CAME8bitMemory
optimizer = CAME8bitMemory(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

To force the full-state reference path even on CUDA 2D weights:
```python
from came_pytorch import CAME8bit
optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    prefer_cuda_fast_path=False,
)
```

Reference full-state 8-bit optimizer:
```python
from came_pytorch import CAME8bitFull
optimizer = CAME8bitFull(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

To force the legacy 2D factored CUDA fallback path:
```python
from came_pytorch import CAME8bitFull
optimizer = CAME8bitFull(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    prefer_factored_cuda_ext_path=False,
)
```

2D-only fast path (will raise for non-2D / non-CUDA params):
```python
from came_pytorch import CAME8bit2D
optimizer = CAME8bit2D(model.parameters(), lr=2e-4, weight_decay=1e-2)
```

CUDA Graph-friendly fast path setup:
```python
from came_pytorch import CAME8bit

optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    cuda_graph_compatible=True,
)

# After grads exist and before capture:
optimizer.prime_for_cuda_graph()
```

Tracked runtime opt-in candidates:
```python
from came_pytorch import CAME8bit

optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    cuda_1d_runtime_mode=CAME8bit.CUDA_1D_RUNTIME_MODE_REPEATED_FP16_PAIRMIN,
)

optimizer = CAME8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=1e-2,
    cuda_nd_runtime_mode=CAME8bit.CUDA_ND_RUNTIME_MODE_CHUNK32_LARGE_DIRECT_ROW_SUM,
)
```
These two runtime profiles are the current opt-in candidates being tracked by
benchmark policy. Lower-level CUDA tuning kwargs remain available for benchmark
or diagnostic work, but these profiles are the intended user-facing entrypoints
for the repeated-1D and repeated mixed-ND experiments.

For the public `CAME8bit` surface, treat these as the stable user-facing knobs:
- `cuda_1d_runtime_mode`
- `cuda_nd_runtime_mode`
- `prefer_cuda_fast_path` / `prefer_cuda_2d_fast_path`
- `prefer_factored_cuda_ext_path`
- `cuda_graph_compatible`

Direct low-level CUDA tuning kwargs such as `prefer_cuda_1d_batched_fast_path`,
`cuda_1d_batched_min_bucket_size`, `cuda_factored_nd_chunk_size_override`, or
`cuda_nonfactored_use_fp16_update` should now be treated as experimental
benchmark / diagnostic controls. `CAME8bit` emits a `RuntimeWarning` when those
are set directly so production-facing code can prefer the runtime profiles.

## Validation

The repository includes a small regression suite for the optimizer behavior:

```bash
python -m pytest tests -q
```

Coverage includes:
- `CAME8bit` fallback coverage for the full-state 8-bit path
- `CAME8bit` CUDA 1D fallback / opt-in batching coverage
- `CAME8bitFull` all-state 8-bit regression tests for factored / non-factored params
- factored / non-factored CUDA fallback coverage for the workspace-backed `CAME8bitFull` path
- CUDA 2D fast-path smoke tests and quantized-state layout checks
- one-step numerical proximity between `CAME8bit2D` and reference `CAME`
- a short optimization loop to catch obvious instability regressions

## Benchmarking

The repository also includes a benchmark harness for the 8-bit optimizer modes:

```bash
python benchmarks/benchmark_came_8bit.py --preset 4096x4096 --device cuda --output-json build/bench_4096.json
```
By default, the synthetic benchmark now runs only `baseline` and
`runtime_candidate` modes. `benchmark_only` modes require explicit selection
with `--mode-category benchmark_only` or explicit `--mode ...`.

To focus on the currently tracked runtime opt-ins instead of the full benchmark
surface, use:

```bash
python benchmarks/benchmark_came_8bit.py \
  --preset mixed_transformer_nd_repeated \
  --device cuda \
  --dtype fp32 \
  --mode-category runtime_candidate
```

To inspect the shared mode catalog and see which modes are baseline,
runtime-candidate, or benchmark-only:

```bash
python benchmarks/benchmark_came_8bit.py --list-modes
python benchmarks/benchmark_came_model.py --list-modes
```

For the matching model-level runtime-candidate artifact:

```bash
python benchmarks/benchmark_came_model.py \
  --preset transformer_1d_heavy \
  --preset transformer_vector_heavy \
  --preset transformer_vector_heavy_repeated_nd \
  --device cuda \
  --dtype fp32 \
  --mode-category runtime_candidate \
  --benchmark-reruns 3 \
  --output-json build/model_benchmark_runtime_candidates.json
```

For a small model-level benchmark that measures both total training-step time and
the optimizer-step slice inside that step:

```bash
python benchmarks/benchmark_came_model.py \
  --preset transformer_nd_tiny \
  --device cuda \
  --dtype fp32 \
  --mode 8bit \
  --mode 8bit_forced_full_factored_ext \
  --mode 8bit_forced_full_no_factored_ext \
  --output-json build/model_bench_tiny.json
```

The model benchmark uses a transformer-like block with LayerNorm, attention-style
projections, a gated MLP, and large-prefix ND parameters so mixed-path optimizer
coverage and end-to-end optimizer share are visible together.
With `--profile-kernels --profile-repeats 5`, it also emits the same
`fact_nd_chunk_self_device_time_us` median/min/max/span fields used by the
synthetic benchmark policy.
The self-hosted GPU benchmark workflow also runs this model benchmark and checks
it against `benchmarks/model_regression_policy.json`.
The fallback-oriented `transformer_nd_fallback` case is also part of that model
regression coverage, so the widened small-prefix ND heuristic stays visible in a
realistic fallback-heavy block.
The model benchmark now also includes `transformer_nd_borderline`, which keeps
multiple `prefix_batch == 8` ND tensors clustered around the promoted `48x48`
cutoff so future heuristic regressions are visible at training-step level.
The same workflow now also records an eager `fp32` sweep that compares `8bit`
against `8bit_no_2d_fastpath`, so the 2D fast-path improvement stays covered by
CI regression policy as well.
Use `--preset transformer_nd_fallback` when you want a model-level case that
keeps small-slice / irregular ND tensors on fallback paths to confirm whether
those remain the next bottleneck.
Use `--preset transformer_nd_subborderline` when you want a benchmark-only
model case for `40x56`-class prefix-8 ND tensors below the current cutoff; the
current `2240` candidate improves step time materially but still drifts too far
from `full8bit` to promote.
Use `--preset transformer_1d_heavy` when you want to isolate how much the
legacy single-vector 1D route or opt-in 1D batching changes model-level total step time.
Use `--preset transformer_heterogeneous_1d_heavy` when you want a model-level
case where 2D weights still dominate numel but heterogeneous 1D tensors
dominate tensor count, so default `8bit` versus opt-in legacy 1D modes can be
read through both `paths` coverage and optimizer-step share.
Use `--preset transformer_vector_heavy` when you want the same 1D comparison in
a more mixed block that still includes 2D fast-path weights and chunked ND work.
Use `--preset transformer_vector_heavy_repeated_nd` when you want the same mixed
block but with repeated `(16,128,128)` and `(32,256,256)` chunked-ND tensors so
shape-selective chunked-ND comparisons are less sensitive to a single tensor.
For those promotion probes, prefer `--benchmark-reruns 3`: model benchmark
output now annotates a concrete `promotion_*` winner using `total_step` as the
primary metric and `optimizer_step` as a tiebreak inside a `2%` total-step band.
The current runtime translation of that benchmark-only winner is
`8bit_runtime_chunk32_direct_row_sum_256_only`, which uses `chunk_size == 32`
for large-prefix ND and only enables chunked-ND `direct_row_sum` for
`256x256`-class matrices and above. In runtime code that maps to
`CAME8bit(..., cuda_nd_runtime_mode=CAME8bit.CUDA_ND_RUNTIME_MODE_CHUNK32_LARGE_DIRECT_ROW_SUM)`.
The self-hosted GPU workflow now also tracks that runtime opt-in on
`transformer_vector_heavy_repeated_nd` with `--benchmark-reruns 3`, comparing it
directly against `8bit_fact_nd_chunk32` under the same total-step-primary
promotion rule used by the benchmark output.
At this point the main development theme is no longer "add another benchmark
probe". The stable split is:
- production baseline: 2D fast path, current small-prefix ND heuristic, and
  required large-prefix ND chunked ext
- runtime opt-in candidates: repeated-vector 1D and repeated mixed-ND runtime
  modes
- benchmark-only diagnostics: grouped 1D, further ND threshold widening, and
  scratch-precision chunked-ND studies
The self-hosted GPU workflow now also keeps `transformer_vector_heavy` in the
model benchmark artifacts so the dedicated 1D CUDA path and the ND chunk metric
both stay visible on a more mixed model case. Its direct `8bit` versus
`8bit_single_1d_fastpath` readout is now observational on that preset rather
than baseline-gating.
The workflow also keeps a `transformer_1d_heavy` regression pair so the simpler
1D-heavy case still guards the legacy single-vector 1D route directly.
The workflow now also keeps a `transformer_heterogeneous_1d_heavy` regression
pair so the default generic 1D fallback and the legacy single-vector route are
compared on a tensor-count-heavy mixed case as well.
The benchmark-only exact-size multitensor candidate to watch now is
`8bit_multitensor_fp16_update_pairmin_1d_fastpath`: it combines fp16 update
scratch with a `min_bucket_size == 2` threshold so repeated vector pairs can hit
the low-copy multitensor path. It is still off by default, but the self-hosted
GPU workflow now tracks it against `8bit_single_1d_fastpath` on both
`transformer_1d_heavy` and `transformer_vector_heavy`.
That candidate also has a runtime-backed benchmark mode now:
`8bit_runtime_fp16_update_pairmin_1d_fastpath`. It is just `CAME8bit` with
`cuda_1d_runtime_mode=CAME8bit.CUDA_1D_RUNTIME_MODE_REPEATED_FP16_PAIRMIN`.
That profile enables the repeated-vector multitensor fp16-update path and also
turns on `cuda_nonfactored_use_fp16_update` so any remaining single-vector
fallback gets the cheaper fp16 update scratch. For heterogeneous 1D-heavy
cases, there is also a narrower probe mode, `8bit_nonfactored_fp16_update`,
that only switches the generic non-factored CUDA fallback to fp16 update
scratch without changing dispatch.
The self-hosted GPU workflow now tracks that narrower fallback probe on
`transformer_heterogeneous_1d_heavy` as well, alongside `8bit` and
`8bit_single_1d_fastpath`, so heterogeneous tensor-count regressions stay
visible separately from the repeated-vector story.
Longer dtype sweeps keep that conclusion narrow: on this RTX 4070 Ti SUPER,
`8bit_nonfactored_fp16_update` is a stable win against baseline `8bit` for the
fp32 heterogeneous model sweep, but bf16/fp16 ordering still flips across
reruns and mixed vector-heavy cases. So `cuda_nonfactored_use_fp16_update`
should still be treated as an opt-in candidate or benchmark probe, not a new
runtime default.
A benchmark-only heuristic mode, `8bit_auto_nonfactored_fp16_update_fp32_only`,
now exists to model the most conservative promotion path: it enables
`cuda_nonfactored_use_fp16_update` only when the parameter dtype is fp32 and
otherwise behaves like plain `8bit`. That is useful for measuring a possible
fp32-only default, but the current results are still noisy enough that it
remains an evaluation mode rather than a policy target.
The main open optimization theme is therefore no longer another immediate ND
threshold change. `fact_nd_chunk` and the promoted `48x48` small-prefix ND rule
are already measurable and regression-guarded, while the next likely leverage is
broader 1D / small non-factored coverage on mixed and model workloads.

Useful options:
- `--preset mixed_small` for a mixed parameter-group benchmark
- `--preset mixed_nd` for a higher-rank mixed benchmark with conv-style tensors
- `--preset mixed_nd_small` for a higher-rank mixed benchmark where small prefix batches can use the ND factored ext path
- `--preset mixed_nd_large` for a higher-rank mixed benchmark where large prefix batches can exercise the ND chunked ext path on eligible tensors
- `--preset mixed_nd_fallback` for a higher-rank mixed benchmark that intentionally keeps small-slice / irregular ND tensors on fallback paths
- `--preset mixed_nd_borderline` for a higher-rank mixed benchmark focused on `prefix_batch == 8` tensors around the current `48x48` small-prefix ND ext cutoff
- `--preset mixed_nd_subborderline` for a benchmark-only higher-rank probe focused on `prefix_batch == 8` tensors just below the current `48x48` cutoff, useful when studying whether `40x56`-class shapes should be promoted next
- `--preset transformer_nd_subborderline` in `benchmark_came_model.py` for the same question at model level; current results favor keeping the `2240` candidate benchmark-only
- `--preset mixed_nonfactored_nd` for an experimental higher-rank probe that flattens contiguous ND tensors through the non-factored path to test whether broader generic fallback work is worth pursuing
- `--preset mixed_vectorlike_nd` for an experimental higher-rank probe restricted to rank>2 tensors with a single non-singleton dimension, to test the narrowest semantics-preserving non-factored candidates first
- `--preset mixed_degenerate_factored_nd` for an experimental higher-rank probe where one of the last two dimensions is `1`; in fp32 this is the narrowest case where the factored approximation becomes exact, so it is the right place to test whether the current 8-bit layout also preserves that equivalence
- `--preset mixed_1d_heavy` for a mixed benchmark with many large 1D tensors so explicit single-vector or batched 1D modes become measurable
- `--preset mixed_heterogeneous_1d_heavy` for a mixed benchmark where 2D weights still dominate numel but heterogeneous 1D tensors dominate tensor count, intended for grouped 1D batching studies
- `--preset mixed_transformer_nd` for a hybrid transformer-like benchmark that mixes large 2D projections, chunk-friendly ND tensors, and bias-style 1D params
- `--preset transformer_nd_borderline` in `benchmark_came_model.py` for a model-level case centered on the promoted `48x48` small-prefix ND cutoff
- `--preset transformer_1d_heavy` in `benchmark_came_model.py` for the simpler direct guard on legacy single-vector 1D behavior
- `--preset transformer_heterogeneous_1d_heavy` in `benchmark_came_model.py` for a model-level case where heterogeneous 1D tensors dominate tensor count and make optimizer-share differences easier to see
- `--preset transformer_vector_heavy` in `benchmark_came_model.py` for a model-level case that mixes large vectors with 2D fast-path weights and chunked ND tensors
- `--nd-prefix-sweep 2,4,8,16,32 --nd-matrix-shape 256,256` to generate isolated ND crossover cases for dispatcher tuning
- `--mode 8bit --mode 8bit_forced_full --mode full8bit --mode came` to compare the main entrypoints
- `--mode 8bit_single_1d_fastpath` to opt into the legacy single-vector CUDA 1D route for direct comparison against the default generic fallback
- `--mode 8bit_batched_1d_fastpath` to try the experimental exact-size repeated-vector 1D batching path
- `--mode 8bit_grouped_1d_fastpath` to try a benchmark-only grouped 1D batching prototype that pads nearby vector sizes into shared buckets
- `--mode 8bit_no_1d_fastpath` as an explicit label for the default generic 1D fallback behavior, or `--mode 8bit_no_2d_fastpath` to isolate the 2D fast path
- `--mode 8bit_flattened_nd_nonfactored` or `--mode 8bit_forced_full_flattened_nd_nonfactored` for a benchmark-only experimental probe that flattens contiguous rank>2 tensors into the non-factored path; current results are speed-favorable but not accuracy-safe enough to promote
- `--mode 8bit_vectorlike_nd_nonfactored` or `--mode 8bit_forced_full_vectorlike_nd_nonfactored` for a narrower benchmark-only probe that only flattens rank>2 tensors with a single non-singleton dimension; this reduces delta versus the broader probe, but it is still not yet safe enough to promote
- `--mode 8bit_degenerate_factored_nd_nonfactored` or `--mode 8bit_forced_full_degenerate_factored_nd_nonfactored` for the exact-candidate probe where one of the last two dimensions is `1`; current results show that the fp32 exactness does not survive the existing flattened 8-bit state layout
- `--mode 8bit_degenerate_factored_nd_slicewise_nonfactored` or `--mode 8bit_forced_full_degenerate_factored_nd_slicewise_nonfactored` for a stricter benchmark-only probe that keeps slice-local non-factored qstate for the same degenerate shapes; current results show essentially the same delta as the flattened variant and much worse step time, so the mismatch is not just a flattened-state artifact
- `--mode full8bit --mode full8bit_no_factored_ext` to compare the default 2D factored CUDA fallback against the legacy path directly
- `--mode full8bit_factored_ext --mode full8bit_no_factored_ext` or `--mode 8bit_forced_full_factored_ext --mode 8bit_forced_full_no_factored_ext` with `--nd-prefix-sweep ...` to measure ND dispatcher crossover points directly
- `--mode full8bit_factored_ext_small_nd` or `--mode 8bit_forced_full_factored_ext_small_nd` with `--benchmark-nd-batched-matrix-min-numel ...` to run benchmark-only small-prefix ND threshold studies without changing production behavior
- `--mode 8bit_forced_full --mode 8bit_forced_full_no_factored_ext` to compare the same fallback choice through the single-entrypoint `CAME8bit`
- `--cuda-graphs` to benchmark CUDA Graph replay for the 2D fast path
- `--profile-kernels` to collect a one-step CUDA profiler summary
- `--output-csv ...` to generate a spreadsheet-friendly report

Reported metrics include:
- average optimizer step time
- peak temporary CUDA memory during the timed region
- fast-path coverage for `CAME8bit`
- path-level coverage by numel and tensor count
- ND crossover metadata such as prefix batches and matrix shapes per path category in JSON / CSV output
- benchmark-only ND cutoff metadata for `*_small_nd` modes when `--benchmark-nd-batched-matrix-min-numel` is used
- final parameter delta vs `CAME8bitFull`

The model-level benchmark additionally reports:
- average total training-step time
- average optimizer-step time inside that training step
- optimizer-step share of the total timed step
- final loss from the timed run

The summary table now includes a `paths` column with compact coverage breakdowns
such as `2d:99.7%/4,fact:0.2%/1,nf:0.1%/2`, which means:
- numel share first
- tensor count second
- categories split into CUDA 2D fast path, CUDA 1D fast path, higher-rank CUDA factored ext path, higher-rank CUDA factored fallback, 2D CUDA factored fallback, CUDA non-factored fallback, CPU full8bit, and reference

For regression checks, save a JSON report and run the policy checker:

```bash
python benchmarks/benchmark_came_8bit.py \
  --preset smoke \
  --preset mixed_small \
  --preset mixed_nd_large \
  --preset mixed_transformer_nd \
  --preset 4096x4096 \
  --device cuda \
  --dtype fp32 \
  --dtype bf16 \
  --dtype fp16 \
  --mode full8bit \
  --mode full8bit_no_factored_ext \
  --mode 8bit_forced_full \
  --mode 8bit_forced_full_no_factored_ext \
  --warmup-steps 5 \
  --benchmark-steps 20 \
  --compare-steps 5 \
  --output-json build/benchmark_regression_eager.json

python benchmarks/benchmark_came_8bit.py \
  --preset 4096x4096 \
  --preset 8192x4096 \
  --device cuda \
  --dtype fp16 \
  --mode 8bit \
  --warmup-steps 5 \
  --benchmark-steps 20 \
  --compare-steps 5 \
  --output-json build/benchmark_regression_fastpath_eager.json

python benchmarks/benchmark_came_8bit.py \
  --preset 4096x4096 \
  --preset 8192x4096 \
  --preset mixed_transformer_nd \
  --device cuda \
  --dtype fp32 \
  --mode 8bit \
  --mode 8bit_no_2d_fastpath \
  --warmup-steps 5 \
  --benchmark-steps 20 \
  --compare-steps 5 \
  --output-json build/benchmark_regression_fastpath_fp32.json

python benchmarks/benchmark_came_8bit.py \
  --preset mixed_transformer_nd \
  --device cuda \
  --dtype fp32 \
  --mode 8bit_forced_full \
  --warmup-steps 5 \
  --benchmark-steps 20 \
  --compare-steps 5 \
  --profile-kernels \
  --profile-repeats 5 \
  --kernel-limit 20 \
  --output-json build/benchmark_regression_fact_nd_chunk_profile.json

python benchmarks/benchmark_came_8bit.py \
  --preset mixed_nd_large \
  --preset mixed_transformer_nd \
  --preset mixed_nd_fallback \
  --preset mixed_nd_borderline \
  --device cuda \
  --dtype fp32 \
  --mode full8bit_factored_ext \
  --mode full8bit_no_factored_ext \
  --mode 8bit_forced_full_factored_ext \
  --mode 8bit_forced_full_no_factored_ext \
  --warmup-steps 5 \
  --benchmark-steps 20 \
  --compare-steps 5 \
  --output-json build/benchmark_regression_factored_ext_fp32.json

The resulting `fact_nd_chunk_self_device_time_us` metric is the median of the
repeated `came_cuda.fact_nd_chunk` profiler samples, and the artifact also
stores the raw run list plus min/max values. When the repeated samples show an
unusually wide spread, the benchmark summary also emits a warning.

python benchmarks/benchmark_came_8bit.py \
  --preset 4096x4096 \
  --preset 8192x4096 \
  --device cuda \
  --dtype fp16 \
  --mode 8bit \
  --warmup-steps 5 \
  --benchmark-steps 20 \
  --compare-steps 5 \
  --cuda-graphs \
  --output-json build/benchmark_regression_graph.json

python benchmarks/check_benchmark_regressions.py \
  --input-json build/benchmark_regression_eager.json \
  --input-json build/benchmark_regression_factored_ext_fp32.json \
  --input-json build/benchmark_regression_fastpath_eager.json \
  --input-json build/benchmark_regression_fastpath_fp32.json \
  --input-json build/benchmark_regression_fact_nd_chunk_profile.json \
  --input-json build/benchmark_regression_graph.json

python benchmarks/benchmark_came_model.py \
  --preset transformer_nd_tiny \
  --preset transformer_nd_fallback \
  --preset transformer_nd_borderline \
  --device cuda \
  --dtype fp32 \
  --mode 8bit_forced_full_factored_ext \
  --warmup-steps 5 \
  --benchmark-steps 20 \
  --compare-steps 5 \
  --profile-kernels \
  --profile-repeats 5 \
  --kernel-limit 5 \
  --output-json build/model_benchmark_fact_nd_chunk_profile.json

python benchmarks/check_benchmark_regressions.py \
  --input-json build/model_benchmark_regression.json \
  --input-json build/model_benchmark_fact_nd_chunk_profile.json \
  --input-json build/model_benchmark_vector_heavy_regression.json \
  --input-json build/model_benchmark_vector_heavy_repeated_nd_regression.json \
  --input-json build/model_benchmark_vector_heavy_fact_nd_chunk_profile.json \
  --input-json build/model_benchmark_1d_heavy_regression.json \
  --input-json build/model_benchmark_heterogeneous_1d_heavy_regression.json \
  --policy benchmarks/model_regression_policy.json
```

The regression checker now also prints a `runtime_candidate_summary` section so
tracked opt-in modes can be reviewed separately from the full rule table.

Artifact / policy map:
- `build/benchmark_regression_*.json` and `build/model_benchmark_regression.json`
  Baseline regression artifacts. These are the main policy-gated inputs that
  must stay healthy on every CI run.
- `build/model_benchmark_vector_heavy_regression.json`,
  `build/model_benchmark_1d_heavy_regression.json`,
  `build/model_benchmark_heterogeneous_1d_heavy_regression.json`, and
  `build/model_benchmark_vector_heavy_repeated_nd_regression.json`
  Focused policy inputs for the current 1D / mixed-ND decision points.
- `build/benchmark_runtime_candidates.json` and
  `build/model_benchmark_runtime_candidates.json`
  Observation-oriented runtime-candidate artifacts. These are intended for
  repeated history collection and manual comparison, not as the sole promotion
  gate.
- `build/benchmark_runtime_candidates_diff.{txt,json}` and
  `build/model_benchmark_runtime_candidates_diff.{txt,json}`
  Workflow-generated post-processing reports that compare the current
  runtime-candidate artifacts against the latest successful prior workflow
  artifact on the comparison branch when available. The same text diff is also
  published into the workflow job summary for quick review, together with the
  recommended artifact review order for promotion checks.
- `build/runtime_candidate_window_review_1d.{txt,json}` and
  `build/runtime_candidate_window_review_mixed_nd.{txt,json}`
  Workflow-generated 5-run review summaries for the two tracked runtime
  candidates when at least 4 prior successful runs are available on the same
  branch. These are also echoed into the workflow job summary before the
  one-run diff sections.
- `build/runtime_candidate_promotion_readiness.{txt,json}`
  Workflow-generated aggregate summary for the tracked runtime candidates. This
  is the quickest way to see which tracked opt-ins currently clear the minimum
  5-run promotion bar before opening the per-candidate window reviews.
- `build/benchmark_benchmark_only_<suite>.{json,csv}` and
  `build/model_benchmark_benchmark_only_<suite>.{json,csv}`
  Manual-only workflow-dispatch artifacts for experimental `benchmark_only`
  sweeps. These live under a separate `came-benchmark-only-sweeps-<suite>`
  artifact and are not part of the default regression or promotion gate.
- `benchmarks/regression_policy.json`
  Synthetic / optimizer-level baseline policy.
- `benchmarks/model_regression_policy.json`
  Model-level policy, including the currently tracked runtime mixed-ND and
  repeated-1D comparisons. On `transformer_vector_heavy`, the direct
  `8bit` versus `8bit_single_1d_fastpath` comparison remains visible in the
  benchmark artifact but is no longer a baseline-gating rule.

Default-promotion bar for a tracked runtime candidate:
- require at least 5 consecutive successful `cuda-benchmark-regression`
  workflow runs with both `benchmark_runtime_candidates.*` and
  `model_benchmark_runtime_candidates.*` present
- treat those 5 runs as the minimum observation window only when they were
  produced with the current `--benchmark-reruns 3` setup
- baseline policy artifacts must stay green for the entire window; a candidate
  never promotes off a run set that already regressed the default baseline
- on its focused model promotion preset, the candidate should win at least 4 of
  the 5 runs and must not fall outside the total-step tie band on any run in
  that window
- repeated-1D promotion claims stay scoped to repeated-vector 1D workloads, and
  repeated mixed-ND claims stay scoped to repeated mixed-ND workloads; do not
  convert a narrow win into a broader heuristic without direct coverage
- promotion remains a human decision after reviewing the policy artifacts plus
  the runtime-candidate diff history; the history is evidence, not an
  auto-promote trigger

How to review that 5-run window:
- first confirm the policy-gated baseline layer stayed healthy on all 5 runs:
  `build/benchmark_regression_*.json`, `build/model_benchmark_regression.json`,
  and the focused policy artifacts for the same run
- for the repeated-1D candidate
  (`8bit_runtime_fp16_update_pairmin_1d_fastpath`), treat
  `build/model_benchmark_1d_heavy_regression.json` as the focused model
  promotion artifact
- for the repeated mixed-ND candidate
  (`8bit_runtime_chunk32_direct_row_sum_256_only`), treat
  `build/model_benchmark_vector_heavy_repeated_nd_regression.json` as the
  focused model promotion artifact
- use `build/model_benchmark_runtime_candidates_diff.txt` and
  `build/benchmark_runtime_candidates_diff.txt` across those 5 runs to check
  whether winner flips or tie-band exits appeared in the observation layer
- when a run needs deeper inspection, open the matching
  `build/model_benchmark_runtime_candidates.json` entry and verify
  `promotion_is_winner`, `promotion_decision_basis`,
  `promotion_candidate_rank`, and
  `promotion_within_primary_tie_band` directly for the candidate mode
- treat `build/model_benchmark_heterogeneous_1d_heavy_regression.json` as a
  guardrail artifact only; it can block over-broad claims, but it is not the
  focused promotion artifact for the repeated-1D candidate
- if the 5-run window clears the bar, promotion review should still cite both
  the focused policy artifact and the runtime-candidate diff history together,
  not either one alone

To aggregate that review across a window directly:

```bash
python benchmarks/review_runtime_candidate_window.py \
  --candidate-mode 8bit_runtime_fp16_update_pairmin_1d_fastpath \
  --run-dir artifacts/run_001 \
  --run-dir artifacts/run_002 \
  --run-dir artifacts/run_003 \
  --run-dir artifacts/run_004 \
  --run-dir artifacts/run_005
```

That review script assumes the provided directories are already consecutive
successful `cuda-benchmark-regression` runs. It infers the focused artifact for
the tracked runtime candidates, prints the per-run focused promotion state, and
summarizes the observation layer across `benchmark_runtime_candidates.json` and
`model_benchmark_runtime_candidates.json`. It also emits a machine-readable
report with `--output-json`.

To aggregate the tracked runtime candidates into one readiness summary:

```bash
python benchmarks/review_runtime_candidate_readiness.py \
  --run-dir artifacts/run_001 \
  --run-dir artifacts/run_002 \
  --run-dir artifacts/run_003 \
  --run-dir artifacts/run_004 \
  --run-dir artifacts/run_005
```

That aggregate review reuses the tracked-candidate defaults from
`review_runtime_candidate_window.py` and reports which candidates, if any,
currently meet the minimum promotion bar across the provided run window.

To diff two runtime-candidate artifacts directly:

```bash
python benchmarks/compare_benchmark_artifacts.py \
  --baseline-json build/model_benchmark_runtime_candidates_prev.json \
  --candidate-json build/model_benchmark_runtime_candidates.json \
  --only-changed-metrics \
  --sort-by worsening \
  --max-entries 8
```
That report defaults to `runtime_candidate` modes and prints only the key timing
and drift metrics needed for promotion review. The header also includes
`worsening / improving / unchanged` counts across the full filtered set, even
when the printed body is capped with `--max-entries`, plus compact counts for
promotion flips and active-path changes. If any winner flips are present, those
cases are re-listed immediately under the header before the capped body, with
rank and tie-band changes included in the one-line summary. Rank-only movement
without a winner flip is intentionally left in the regular diff body for now so
the summary stays focused on promotion-relevant changes rather than noise. The
header still counts tie-band entries and exits separately, so near-promotion
movement is visible even when `winner_flips=0`. Those tie-band-only cases are
intentionally not re-listed in the focus section yet; at the current candidate
count, the header count plus the regular diff body is enough.

For ND dispatcher tuning, an isolated prefix-batch sweep looks like:

```bash
python benchmarks/benchmark_came_8bit.py \
  --nd-prefix-sweep 2,4,8,16,32 \
  --nd-matrix-shape 256,256 \
  --device cuda \
  --dtype fp32 \
  --mode full8bit_factored_ext \
  --mode full8bit_no_factored_ext \
  --output-json build/nd_prefix_sweep.json
```

The repository also includes a self-hosted GPU GitHub Actions workflow at
`.github/workflows/cuda-benchmark-regression.yml` that runs the same sweep, plus
`.github/workflows/cpu-tests.yml` for CPU-only regression coverage on pull requests.
The GPU workflow also uploads benchmark JSON / CSV files as artifacts.
When launched with `workflow_dispatch`, that GPU workflow also exposes a
`run_benchmark_only_sweeps` input that runs a separate manual-only job for the
experimental `benchmark_only` mode category. It also exposes a
`benchmark_only_suite` choice with:
- `all` for the broad experimental sweep
- `grouped_1d` for grouped/vector-batching studies; this suite now emits
  separate exact-size and variable-length artifacts instead of mixing them
- `nd_probes` for ND threshold / flattened-ND style probes

## Hyper-parameter Tuning

* Pre-training: Based on our experiments on BERT-Large, GPT-2, and T5, it's suitable to choose a learning rate for CAME 0.5-0.9x lr for AdamW.
* Set $\beta_1$ and $\beta_2$ to the same values used in AdamW. Choose $\beta_3$ to be larger than $\beta_2$. For example, consider choosing $\beta_3$ between $[0.9995, 0.99995]$ if setting $\beta_1, \beta_2=0.9, 0.999$, and choosing $\beta_3$ between $[0.99, 0.999]$ if setting $\beta_1, \beta_2=0.9, 0.95$. Due to computational resource constraints, we did not explore more combinations of three betas. Different training tasks may require different combinations of optimal performance.
* If you have any feedback or comments regarding hyper-parameter tuning, please do not hesitate to provide them to us!

## Experiments

Apart from the BERT and T5 experiments shown in the paper, we conduct more and record the results here.

### Fine-tuning Llama-7B

|                |    MMLU     |  WikiText  |  HellaSwag  |  TruthfulQA (MC)  |    BoolQ    |    COPA     |     WSC     |     WIC     |
| -------------- |:-----------:|:----------:|:-----------:|:-----------------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Alpaca-7B      |    40.21    |    6.74    |    59.76    |     **38.89**     |  **79.57**  |  **88.00**  |    46.15    |    49.84    |
| Alpaca-7B-CAME |  **40.59**  |  **6.38**  |  **59.80**  |       38.61       |    79.08    |  **88.00**  |  **49.04**  |  **50.78**  |

We fine-tuned Llama-7B with [stanford-alpaca](https://github.com/tatsu-lab/stanford_alpaca) (52k instruction-tuning dataset). To replicate our result, first register the CAME optimizer to the transformer package. Then in Alpaca training script, change the default optimizer from "adamw" to "came".

Alpaca-7B and Alpaca-7B-CAME are evaluated using [Instruct-eval](https://github.com/declare-lab/instruct-eval) and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

### Pre-training Llama on C4

<p align="center">
<img src="assets/llama_came.png" alt="CAME optimizer of Llama Pre-training" width="50%" />
</p>
<!-- ![CAME_code](assets/llama_came.png) -->

The pre-training of Llama-1B is based on [C-Optim](https://github.com/kyleliang919/C-Optim). The hyperparameters of CAME are configured with betas (0.9, 0.95, 0.995), and AdamW uses betas (0.9, 0.95).

### Pre-training GPT-2

![CAME_gpt2](assets/gpt-2_came.png)

The pre-training of GPT-2 (Medium, 345M) is based on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM). To replicate our result, add the CAME optimizer in [`megatron/optimizer/__init__.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/optimizer/__init__.py) and set the *args.optimizer* to "came".

## Memory Usage Comparison
To ensure a fair comparison, we set the batch size to 1 for the pre-training of GPT-2 (Medium) to examine the memory footprint of CAME and AdamW.

|              | AdamW | CAME     | 
|--------------|-------|----------|
| Memory (GiB) | 8.77  | **7.44** | 

## Citation

```bibtex
@inproceedings{luo2023came,
  title={CAME: Confidence-guided Adaptive Memory Efficient Optimization},
  author={Luo, Yang and Ren, Xiaozhe and Zheng, Zangwei and Jiang, Zhuo and Jiang, Xin and You, Yang},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={4442--4453},
  year={2023}
}
```
