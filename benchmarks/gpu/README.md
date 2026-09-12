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

Generate the convergence and surface-field figures used by the project report
directly from the downloaded archive with:

```sh
python benchmarks/gpu/analyze_extended_convergence.py \
  simsopt-extended-convergence.zip docs/gpu_native/figures
```

The analyzer validates the schema and required members, reads the raw-appended
VTK surface arrays without modifying the archive, and writes
`extended_convergence.png` and `extended_surface_field.png`.

## Absolute feasibility and penalty continuation

The next workflow separates physical acceptability from CPU/GPU relative
agreement. Comparison schema 4 adds an absolute gate for the positive deficits
of both final designs. The default allowances are 0.1 mm for either minimum-
distance constraint and `1e-3` for the maximum-curvature and maximum-mean-
squared-curvature excesses. These are explicit command-line policy values, not
hidden changes to the physical objective.

First, a short Cartesian sweep compares L-BFGS-B history sizes (`maxcor`) and
line-search step limits (`maxls`). Only candidates passing initial and early-
trajectory backend parity are eligible. The deterministic ranking minimizes,
in order, the number of failed absolute constraints, worst and total normalized
violation, final gradient norm, normalized-normal-field RMS, and evaluation
count. Each candidate JSON and the sweep summary retain the CPU and GPU final
metrics.

Then the continuation runner multiplies all four engineering inequality
penalties together while leaving flux, length, thresholds, coordinates, and
problem resolution unchanged. The default stages use multipliers 1, 10, and
100 for 100, 100, and 200 iterations. After each nonfinal stage, the better
CPU/GPU endpoint under the same feasibility-first ranking becomes one common
physical starting vector for both backends. This avoids giving either backend
a different continuation path. Only the final stage writes VTS/VTU files.

[Run the feasibility study in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_feasibility_continuation.ipynb)

```sh
OMP_NUM_THREADS=1 python benchmarks/gpu/sweep_solver_feasibility.py \
  --problem stress --maxiter 25 \
  --maxcor-values 10,100,300 --maxls-values 20,50 \
  --current-scale 100000 \
  --output-dir benchmarks/gpu/results/solver-sweep

OMP_NUM_THREADS=1 python benchmarks/gpu/run_penalty_continuation.py \
  --problem stress --penalty-multipliers 1,10,100 \
  --stage-maxiters 100,100,200 --maxcor 100 --maxls 50 \
  --current-scale 100000 \
  --output-dir benchmarks/gpu/results/penalty-continuation
```

Use the actual sweep winner in the second command; `100/50` above only
illustrates argument placement. The aggregate result is
`penalty-continuation-summary.json`. A false feasibility or stationarity gate
is retained as evidence and should not be converted into a passing result by
loosening tolerances after inspection.

Generate the solver-sweep, staged-continuation, and final surface-field figures
from the downloaded archive with:

```sh
python benchmarks/gpu/analyze_feasibility_study.py \
  simsopt-feasibility-study.zip docs/gpu_native/figures
```

The analyzer validates both workflow summaries, every stage's schema and
penalty multiplier, and all final VTK payloads. It also emits a concise JSON
summary containing the archive digest, final CPU/GPU objective, normalized
normal-field and constraint metrics, aggregate timing, and surface-map
comparison.

## Equality-zero augmented Lagrangian

The next implementation replaces escalation of one fixed weighted sum with the
augmented-Lagrangian convention used by the supplied `auglag_qa.py` reference:

```text
L_A(x, lambda, mu) = f(x) - lambda^T c(x)
                     + 0.5 sum_i mu_i c_i(x)^2
lambda <- lambda - mu * c
```

Here `f` is quadratic flux plus the small linear length regularizer. The four
entries of `c` are SIMSOPT's nonnegative coil--coil distance, coil--surface
distance, pointwise-curvature, and mean-squared-curvature penalty objectives.
They are equality-to-zero constraints: each is exactly zero when its underlying
engineering inequality is satisfied. This deliberately follows the supplied
workflow rather than a Powell--Hestenes--Rockafellar projected-inequality
formulation. Initial multipliers are zero, making CPU/GPU comparisons
deterministic, and multiplier and penalty vectors are dynamic inputs to one
compiled JAX executable.

The benchmark runs independent CPU-oracle and GPU-native AL trajectories from
the same physical vector. Its schema-6 JSON stores every outer state, inner
iterations and evaluations, constraint values, multipliers, penalties, final
physical variables, and complete final field and coil metrics for both
backends. The final CPU/GPU surfaces and coils are exported as VTS/VTU, with
signed and absolute `(B dot n) / abs(B)` point arrays on each surface. It also
records `nvidia-smi` accelerator identity, UUID, driver version, and memory
when available.

[Run the augmented-Lagrangian benchmark in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_augmented_lagrangian.ipynb)

```sh
OMP_NUM_THREADS=1 python benchmarks/gpu/benchmark_augmented_lagrangian.py \
  --problem stress --max-outer-iterations 8 --max-inner-iterations 50 \
  --mu-init 10 --tau 10 --mu-max 1e12 --current-scale 100000 \
  --target-tile-size 1024 --source-tile-size 4320 \
  --output benchmarks/gpu/results/stress-augmented-lagrangian.json
```

The zero-penalty constraint tolerance and the physical feasibility tolerances
are intentionally separate. Passing `c_i <= 1e-8` does not replace the recorded
distance/curvature checks, and neither a failed stationary gate nor a failed
physical gate prevents artifact export.

Validate a returned archive and generate the convergence, per-outer-step
performance, and final normalized-normal-field figures with:

```sh
python benchmarks/gpu/analyze_augmented_lagrangian.py \
  simsopt-augmented-lagrangian.zip docs/gpu_native/figures
```

The analyzer rejects incomplete or mismatched schemas, checks the full outer
histories and VTK payloads, and writes
`augmented_lagrangian_analysis_summary.json` with the archive digest, acceptance
gates, final CPU/GPU metrics, optimization states, stage speedups, and surface
comparison statistics.

## Augmented-Lagrangian conditioning study

Constraint scaling is explicit in schema 6. For positive scales `s_i`, the AL
uses `c_hat_i = c_i / s_i`, including the multiplier update, while stopping,
the active violation mask, and all physical feasibility gates continue to use
raw `c_i` and direct geometric metrics. Both representations are retained in
every outer-loop record and final CPU/GPU summary.

The automated study screens four predeclared candidates on the smaller
`engineering` problem: the original schedule, a longer inner solve, gentler
penalty growth, and measured per-family scaling with gentler growth. Candidates
are ranked lexicographically by direct physical feasibility first, then
stationarity, normalized field quality, and evaluation count. Only the winner
is rerun on `stress`, where the normal-field VTS surfaces and VTU coil sets are
exported for both CPU and GPU endpoints.

[Run the conditioning study in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_augmented_lagrangian_conditioning.ipynb)

```sh
OMP_NUM_THREADS=1 python \
  benchmarks/gpu/sweep_augmented_lagrangian_conditioning.py \
  --screen-problem engineering --final-problem stress \
  --maxcor 100 --maxls 50 --current-scale 100000 \
  --target-tile-size 1024 --source-tile-size 4320 \
  --output-dir benchmarks/gpu/results/al-conditioning
```

The compact `conditioning-study-summary.json` records candidate ranking and
the final acceptance gates. The four screening JSON files and production
winner JSON preserve complete CPU/GPU histories and final objective,
`(B dot n) / abs(B)`, coil-constraint, raw/scaled AL, timing, and provenance
metrics. No conditioning choice is accepted until the returned NVIDIA archive
passes parity, artifact, and direct physical-feasibility validation.
