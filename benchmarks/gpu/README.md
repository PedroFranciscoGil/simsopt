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
CPU fallback, runs numerical parity tests, sweeps and confirms candidate tile
sizes, reports compilation separately from steady-state execution, profiles
the winning tile pair with JAX/Perfetto, saves a device-memory profile, and
downloads all artifacts as a zip file. The synchronized CPU comparison uses
one OpenMP thread and explicitly pins SIMSOPT's JAX-based geometry operations
to a CPU device.

The standalone autotuner evaluates a Cartesian product of tile sizes. Values
larger than a problem dimension are clipped to that dimension, which includes
a one-tile candidate for the minimal problem. Each candidate must pass CPU
objective and gradient parity before it can be ranked. The fastest three from
the screening pass and the current 128-by-256 default are timed again; the
confirmed median selects the winner.

    OMP_NUM_THREADS=1 python benchmarks/gpu/sweep_gpu_tiles.py \
      --problem minimal \
      --output benchmarks/gpu/results/minimal-tile-sweep.json

The same harness can be run on any CUDA machine:

    python benchmarks/gpu/profile_gpu_objective.py \
      --problem minimal \
      --trace-dir benchmarks/gpu/results/minimal-trace \
      --memory-profile benchmarks/gpu/results/minimal-memory.prof \
      --output benchmarks/gpu/results/minimal-gpu.json
