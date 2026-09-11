# GPU-native coil benchmarks

The benchmark matrix has three fixed problem classes:

- **minimal**: the documented minimal stage-two problem;
- **engineering**: the stage-two objective with distance and curvature terms;
- **stress**: a high-resolution scaling and memory case.

Record a CPU baseline from the repository root:

    OMP_NUM_THREADS=1 .venv/bin/python benchmarks/gpu/benchmark_objective.py \
      --problem minimal \
      --output benchmarks/gpu/results/minimal-cpu.json

Use at least seven repetitions for recorded results. Compilation and
accelerator measurements will be added separately so cold-start and steady
state timings cannot be confused. Result JSON files are machine-readable and
include the exact Git revision, dimensions, dependency versions, active
objective terms, thread count, raw timing samples, and optional L-BFGS-B solve.

## NVIDIA GPU profiling with Colab

Open the checked-in notebook directly in Colab:

[Run the NVIDIA profiler in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_profile.ipynb)

Select a GPU runtime before executing the first cell. The notebook refuses a
CPU fallback, runs numerical parity tests, jointly sweeps and confirms
candidate tile sizes and reverse-pass implementations, reports compilation
separately from steady-state execution, profiles the winner with JAX/Perfetto,
saves a device-memory profile, and downloads all artifacts as a zip file. The
synchronized CPU comparison uses one OpenMP thread and explicitly pins
SIMSOPT's JAX-based geometry operations to a CPU device.

The standalone autotuner evaluates a Cartesian product of tile sizes and the
ordinary-autodiff and analytic-custom-VJP modes. Values larger than a problem
dimension are clipped to that dimension, which includes a one-tile candidate
for the minimal problem. Each candidate must pass CPU objective and gradient
parity before it can be ranked. The fastest three from the screening pass, the
best ordinary-autodiff candidate, and the original 128-by-256 autodiff baseline
are timed again; the confirmed median selects the winner.

The analytic custom VJP is the default for composed reverse-mode objectives.
The ordinary `biot_savart_field` function remains available when a forward-mode
JVP is required. Tile defaults remain conservative and should be autotuned for
each workload and accelerator.

    OMP_NUM_THREADS=1 python benchmarks/gpu/sweep_gpu_tiles.py \
      --problem minimal \
      --output benchmarks/gpu/results/minimal-tile-sweep.json

The same harness can be run on any CUDA machine:

    python benchmarks/gpu/profile_gpu_objective.py \
      --problem minimal \
      --trace-dir benchmarks/gpu/results/minimal-trace \
      --memory-profile benchmarks/gpu/results/minimal-memory.prof \
      --output benchmarks/gpu/results/minimal-gpu.json

## Production-scale core-objective benchmark

The dedicated production workflow uses the `stress` dimensions: six base
coils, order 12, 180 quadrature points per curve, and a 128-by-128 half-period
surface. This measures the GPU-native quadratic-flux and length objective at
production scale. Coil-distance, surface-distance, and curvature terms remain
deferred and are identified as such in the generated summary.

[Run the production-scale profiler in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_production_profile.ipynb)

The production runner sweeps 25 custom-VJP tile configurations, confirms the
three fastest, times the one-thread CPU baseline for seven repetitions, profiles
the winner, and writes a gate summary alongside the trace and memory profile.
The 3x speed gate is reported without suppressing artifact export when it is not
met.

    python benchmarks/gpu/production_benchmark.py \
      --problem stress \
      --output-dir benchmarks/gpu/results/production
