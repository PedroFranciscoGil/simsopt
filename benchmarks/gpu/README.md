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

## Production-scale full-engineering benchmark

The dedicated production workflow uses the `stress` dimensions: six base
coils, order 12, 180 quadrature points per curve, and a 128-by-128 half-period
surface. This measures quadratic flux, length, coil--coil distance,
coil--surface distance, curvature, and mean-squared curvature in one
differentiated GPU objective. No term in the pinned stage-two benchmark
objective is deferred. Pass `--objective-scope core` or
`--objective-scope local-engineering` to the lower-level sweep or profiler to
reproduce the earlier incremental measurements.

[Run the production-scale profiler in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_production_profile.ipynb)

The production runner sweeps 25 custom-VJP tile configurations, confirms the
three fastest, times the one-thread CPU baseline for seven repetitions, profiles
the winner, and writes a gate summary alongside the trace and memory profile.
The 3x speed gate is reported without suppressing artifact export when it is not
met.

    python benchmarks/gpu/production_benchmark.py \
      --problem stress \
      --output-dir benchmarks/gpu/results/production

## SciPy L-BFGS-B trajectory comparison

After the complete objective passes its production gates, compare the unchanged
SciPy solver on the CPU `Optimizable` graph and the compiled GPU bridge. The
bridge uses SIMSOPT's current-first degree-of-freedom order, keeps fixed currents
out of the optimization vector, and transfers only the flat vector, scalar, and
gradient. The workflow records every evaluation and accepted iterate, then
checks the two final designs through the CPU engineering-metric oracle.

Every optimization artifact stores a complete final snapshot for each backend.
The snapshot includes the mean, RMS, and maximum of
`abs(B dot n) / norm(B)` over the target surface, as well as per-base-coil
lengths, maximum curvatures, mean-squared curvatures, minimum coil--coil and
coil--surface distances, configured limits, signed feasibility margins, and
positive constraint violations. Both CPU and GPU designs are evaluated by the
same SIMSOPT CPU oracle so backend comparisons do not mix metric definitions.
Recorded runs with `--output` also export each final surface as `.vts` with
signed and absolute `(B dot n) / norm(B)` point data and all symmetry-expanded
physical coils as `.vtu`. By default they are placed beside the JSON with its
stem as a prefix; `--visualization-dir` selects a dedicated directory and the
short `cpu_final_*` and `gpu_final_*` names.

[Run the trajectory comparison in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_scipy_trajectory.ipynb)

    OMP_NUM_THREADS=1 python benchmarks/gpu/compare_scipy_trajectories.py \
      --problem stress \
      --maxiter 25 \
      --target-tile-size 1024 \
      --source-tile-size 4320 \
      --output benchmarks/gpu/results/stress-scipy-trajectory.json

## Scaled convergence experiment

The follow-up keeps the physical objective unchanged while mapping one current
coordinate unit to 100,000 amperes. Both CPU and GPU gradients receive the
corresponding exact diagonal chain-rule factor. This addresses the scale
separation between raw current and curve coordinates before any native solver
comparison.

[Run the scaled convergence experiment in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_scaled_convergence.ipynb)

    OMP_NUM_THREADS=1 python benchmarks/gpu/compare_scipy_trajectories.py \
      --problem stress \
      --maxiter 100 \
      --current-scale 100000 \
      --target-tile-size 1024 \
      --source-tile-size 4320 \
      --output benchmarks/gpu/results/stress-scaled-convergence.json

## Extended convergence and ParaView export

The next workflow raises the cap to 300 iterations. Numerical backend parity is
judged over the first 25 accepted iterates, while final acceptance separately
requires stationarity, normalized-field quality, constraint quality, evaluation
budget, and speed. Full-trajectory differences remain in the JSON as diagnostic
data. The downloaded archive also contains CPU and GPU `.vts` surfaces and
`.vtu` coils.

[Run extended convergence in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_extended_convergence.ipynb)

    OMP_NUM_THREADS=1 python benchmarks/gpu/compare_scipy_trajectories.py \
      --problem stress \
      --maxiter 300 \
      --trajectory-parity-iterations 25 \
      --current-scale 100000 \
      --target-tile-size 1024 \
      --source-tile-size 4320 \
      --visualization-dir benchmarks/gpu/results/extended-visualization \
      --output benchmarks/gpu/results/stress-extended-convergence.json
