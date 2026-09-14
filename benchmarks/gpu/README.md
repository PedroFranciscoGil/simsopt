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
the same physical vector. Its schema-7 JSON stores every outer state, inner
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

Validate the returned archive and generate the screening, production
convergence, and surface-field figures with:

```sh
python benchmarks/gpu/analyze_augmented_lagrangian_conditioning.py \
  simsopt-al-conditioning.zip docs/gpu_native/figures
```

The analyzer independently reproduces the candidate ranking, validates every
schema-6/7 history, cross-checks the selected production configuration, verifies
the signed/absolute surface fields, and emits
`al_conditioning_analysis_summary.json` with all decision metrics.

## Safeguarded AL qualification

The next workflow addresses the failure observed in the conditioning study.
It adds a strict inner-stationarity safeguard: when L-BFGS-B returns without
meeting its requested infinity-norm gradient tolerance, that stage cannot
update multipliers or penalties and the outer solve stops. It also tests the
zero-preserving residual-like mapping
`q(c) = sqrt(c + epsilon**2) - epsilon`. Optional automatic scales are computed
from the initial base gradient and constraint Jacobian in optimizer coordinates;
they are never below one, so calibration can attenuate but never amplify a
constraint penalty-gradient contribution.

[Run safeguarded qualification in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_augmented_lagrangian_safeguards.ipynb)

```sh
OMP_NUM_THREADS=1 python \
  benchmarks/gpu/sweep_augmented_lagrangian_safeguards.py \
  --screen-problem engineering --final-problem stress \
  --max-outer-iterations 10 --max-inner-iterations 200 \
  --current-scale 100000 --target-tile-size 1024 \
  --source-tile-size 4320 \
  --output-dir benchmarks/gpu/results/al-safeguards
```

Every screen exports CPU/GPU surface VTS files with signed and absolute
`(B dot n) / abs(B)` and coil VTU files. Production is run only when a screen
passes backend, float64 parity, physical feasibility, stationarity, final field
quality, and final constraint quality. Diagnostic ranking is reported but is
not permitted to turn a failed candidate into a winner; “no qualified winner”
is an expected valid result.

Visualization paths in the result JSON are artifact names relative to the
workflow output directory. Notebook validation resolves only the `surface_vts`
and `coils_vtu` fields against that directory; the accompanying point-data
lists describe array names and are not filesystem paths.

After downloading the archive, reproduce its qualification decisions, validate
all VTK arrays, and generate the screen, calibration, convergence, and surface
figures with:

```sh
python benchmarks/gpu/analyze_augmented_lagrangian_safeguards.py \
  simsopt-al-safeguards.zip docs/gpu_native/figures
```

## Local engineering-residual parity

The next boundary replaces the four already-aggregated squared engineering
penalties with a fixed vector of physically normalized, unsquared local hinge
residuals. Coil--coil and coil--surface entries retain one nearest-sampled-point
violation per relevant coil point, curvature remains pointwise on each base
curve, and mean-squared curvature contributes one value per base curve. The
zero set includes exactly the configured physical feasibility allowance.

This is deliberately a qualification benchmark, not an optimization run. It
therefore does not produce final-design VTS/VTU files or claim solution quality;
those mandatory artifacts and final CPU/GPU field/constraint metrics resume
when the qualified residual vector is connected to the safeguarded optimizer.

[Run local-residual parity in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_local_residual_parity.ipynb)

The notebook runs the engineering case in float64 on an NVIDIA GPU. It checks
production-threshold values and then uses a deterministic, slightly
off-symmetry probe to avoid non-unique derivatives at exact nearest-neighbor or
curvature ties. Three CPU centered finite differences are compared with GPU
JVPs. Download the archive even when a gate fails.

The equivalent command is:

```sh
OMP_NUM_THREADS=1 python \
  benchmarks/gpu/benchmark_local_residual_parity.py \
  --problem engineering --directions 3 --finite-difference-step 1e-7 \
  --current-scale 100000 --target-tile-size 1024 \
  --source-tile-size 4320 \
  --output local-residual-parity.json
```

Validate either the JSON file or downloaded ZIP and generate the activation,
value-parity, Jacobian-parity, and timing figure with:

```sh
python benchmarks/gpu/analyze_local_residual_parity.py \
  simsopt-local-residual-parity.zip docs/gpu_native/figures
```

## Local-residual augmented-Lagrangian optimization

The qualified 8,884-entry engineering residual vector is now connected to the
safeguarded AL optimizer. CPU and GPU runs lower the same JAX program onto
explicit CPU and GPU devices, so this experiment isolates accelerator execution
from differences between derivative implementations. The earlier independent
NumPy/SIMSOPT value and directional-Jacobian qualification remains the oracle
for the residual implementation itself. Reverse mode forms the required
Jacobian-transpose products without constructing the dense residual Jacobian.

The default `family_l2` policy applies one attenuation-only scale per family:
`max(sqrt(family count), initial family L2 norm)`. The AL workflow then
uses the zero-preserving transform `sqrt(r**2 + epsilon**2) - epsilon` and can
reduce family scales after accepted outer stages without recompiling. The
Colab qualification uses a gentler 0.5 scale reduction, a `sqrt(count)` floor,
and an absolute-or-relative inner first-order safeguard. Raw residuals remain
available for strict convergence diagnostics, and outer histories summarize
large vectors while retaining named final family statistics.

Schema 3 treats the AL endpoint as a feasibility warm start rather than the
final design. It runs the full direct-penalty objective through separately
CPU- and GPU-pinned JAX/SciPy bridges. The penalty hinge boundaries are placed
at the allowed 10% engineering envelope, and every accepted L-BFGS-B iterate is
checked by a compiled local-residual oracle. The exported state is the
minimum-quadratic-flux target-feasible checkpoint. This protects the result
from an infeasible terminal step and makes the required quadratic-flux target
of `1e-5` explicit.

Schema 4 adds a matched, fully device-resident L-BFGS experiment. CPU SciPy,
GPU SciPy, and device JAX solvers receive the same physical warm start and use
the same direct objective, target envelope, and accepted-checkpoint policy.
The JAX solver keeps the two-loop recursion, fixed-size history, Armijo line
search, target checks, and checkpoint selection inside one compiled executable;
it performs zero per-iteration host callbacks. Once a target-feasible point is
known, the host solvers stop after 25 accepted checkpoints without a relative
flux improvement of at least `1e-3`, or after 15 consecutive infeasible
checkpoints. Following the replay study below, the device solver uses a
separately configurable patience of 15. The full iteration allowance remains
available when no target-feasible point has yet been found.

The A100 schema-4 qualification passes the scientific target policy for all
three designs. Target-aware SciPy stops at iteration 66 and preserves the
schema-3 checkpoint exactly, reducing CPU/GPU refinement time to 58.43/0.694 s.
Device L-BFGS selects checkpoint 64 after 89 iterations and reaches quadratic
flux `7.568e-7` in 0.747 s. It is 1.14x faster per evaluation and 1.77x faster
than GPU SciPy when compilation is included, but 7.6% slower as a warm solve
because its Armijo trajectory requires more evaluations. Its physical targets
all pass; its scalar objective differs from CPU SciPy by 150%, documenting a
different valid nonconvex trade-off rather than trajectory equivalence.

[Run local-residual AL optimization in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_local_residual_augmented_lagrangian.ipynb)

```sh
OMP_NUM_THREADS=1 python \
  benchmarks/gpu/benchmark_local_residual_augmented_lagrangian.py \
  --problem engineering --max-outer-iterations 8 \
  --max-inner-iterations 300 --residual-scaling-policy family_l2 \
  --target-relative-tolerance 0.10 \
  --quadratic-flux-target 1e-5 \
  --max-refinement-iterations 600 --max-refinement-evaluations 1500 \
  --target-checkpoint-patience 25 --target-infeasible-patience 15 \
  --target-minimum-relative-improvement 1e-3 \
  --device-lbfgs-history-size 20 \
  --device-lbfgs-checkpoint-patience 15 \
  --device-lbfgs-line-search-iterations 30 \
  --refinement-constraint-weight-multiplier 1 \
  --inner-stationarity-relative-tolerance 0.01 \
  --constraint-transform smooth_abs --constraint-transform-epsilon 0.1 \
  --minimum-residual-scaling-policy sqrt_count \
  --constraint-scale-reduction-factor 0.5 \
  --current-scale 100000 --target-tile-size 1024 \
  --source-tile-size 4320 \
  --output local-residual-augmented-lagrangian.json
```

The result always records final CPU/GPU SciPy and device-GPU objective, normalized
`abs(B dot n) / abs(B)`, and coil-constraint metrics. Scientific validation
requires all three designs to satisfy one-sided engineering targets within 10% and
to have quadratic flux at or below `1e-5` with the same 10% allowance.
Because the ideal normalized normal field is zero, a percentage-to-target test
is undefined for it; the benchmark stores mean, RMS, and maximum absolute
values. CPU/GPU agreement for all retained quantities remains an independent
technical gate, so different nonconvex paths cannot redefine scientific target
satisfaction. Gradient magnitude, strict
local-residual feasibility, and tight physical allowances remain explicit
diagnostics and do not veto an otherwise physically validated design. With an
output path, the workflow also exports all three surfaces as VTS—including signed
and absolute normalized
normal field—and all three symmetry-expanded coil sets as VTU. Failed scientific
gates do not suppress these diagnostic artifacts.

Validate a returned archive and generate convergence, performance, and surface
figures with the following command. Schema-1 outputs retain the
`local_residual_al_*` prefix; schema-2 smoothed-continuation outputs use
`smoothed_local_residual_al_*`; and schema-3 target-refinement outputs use
`flux_target_refinement_*`; schema-4 device comparisons use
`device_lbfgs_comparison_*`, so new measurements do not overwrite historical
figures. Schema 3 adds an accepted-checkpoint flux trajectory with the target
and exported state marked explicitly. Schema 4 expands the trajectory and
surface comparison to the device-resident solver.

```sh
python benchmarks/gpu/analyze_local_residual_augmented_lagrangian.py \
  simsopt-device-lbfgs.zip docs/gpu_native/figures
```

## Device-LBFGS warm-start replay sweep

The replay benchmark loads the committed 207-variable, scientifically
qualified AL warm start rather than rerunning the expensive AL phase. It
compares device-resident L-BFGS history sizes 10, 20, 50, and 100 at checkpoint
patience 15 and 25. Every candidate retains its final objective, quadratic
flux, normalized normal field, coil constraints, compact optimization history,
compile time, and warm solve time. The SIMSOPT CPU oracle validates every final
state. Ranking first prefers target-valid candidates whose scalar objective is
within 10% of the CPU SciPy reference, then minimizes quadratic flux.

[Run the device-LBFGS replay sweep in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_device_lbfgs_replay.ipynb)

```sh
OMP_NUM_THREADS=1 python benchmarks/gpu/benchmark_device_lbfgs_replay.py \
  --history-sizes 10 20 50 100 \
  --checkpoint-patiences 15 25 \
  --infeasible-patience 15 --max-iterations 150 \
  --output device-lbfgs-replay-sweep.json
```

Validate the returned archive and generate the parameter-grid and trajectory
figures with:

```sh
python benchmarks/gpu/analyze_device_lbfgs_replay.py \
  simsopt-device-lbfgs-replay.zip docs/gpu_native/figures
```

The A100 replay archive with SHA-256
`e5d0b77b2ac744ef671ab3b9d739c9e15d3088d99113024711791225c5fd3d24`
qualified all eight candidates under the 10% physical-target policy. History
20 with patience 15 was the minimum-flux candidate (`7.56845e-7`) and completed
in 0.675 s with 81 evaluations. It reproduces the patience-25 endpoint with 11
fewer evaluations. History 50 with patience 15 was the speed-first alternative
at 0.559 s and 60 evaluations, with quadratic flux `8.00589e-7`. The aggregate
scalar-objective comparison remains a trajectory diagnostic: it does not
override scientific validation based on the explicit flux and coil targets.

## End-to-end device-resident augmented Lagrangian

The end-to-end qualification compares the established CPU implementation—a
Python augmented-Lagrangian outer loop with SciPy L-BFGS-B inner solves and a
CPU-pinned JAX local-residual objective—against a GPU implementation whose AL
outer loop, L-BFGS two-loop recursion, Armijo line searches, multiplier and
penalty updates, residual-scale continuation, and stationarity safeguard all
execute inside one compiled program. The comparison starts from the same
circular engineering coils and performs no post-AL refinement.

[Run the end-to-end CPU/GPU-native AL comparison in Colab](https://colab.research.google.com/github/PedroFranciscoGil/simsopt/blob/gpu-native-objective/benchmarks/gpu/colab_end_to_end_device_augmented_lagrangian.ipynb)

```sh
OMP_NUM_THREADS=1 python \
  benchmarks/gpu/benchmark_end_to_end_device_augmented_lagrangian.py \
  --problem engineering --max-outer-iterations 8 \
  --max-inner-iterations 300 --mu-init 10 --tau 2 --mu-max 1e12 \
  --history-size 20 --cpu-maxcor 100 \
  --max-line-search-iterations 50 --gpu-warm-repeats 3 \
  --target-relative-tolerance 0.10 --quadratic-flux-target 1e-5 \
  --current-scale 100000 --target-tile-size 1024 \
  --source-tile-size 4320 \
  --output end-to-end-device-augmented-lagrangian.json
```

The output records CPU compile and optimization time, GPU compilation and
three synchronized execution samples, cold end-to-end and warm speedups,
objective/gradient evaluation counts, complete outer histories, final
objective, quadratic flux, normalized `B dot n / abs(B)`, coil constraints,
and both endpoints' physical variables. CPU and GPU surfaces are exported as
VTS with the normal-field arrays, and both coil sets are exported as VTU.

Analyze the returned archive with:

```sh
python benchmarks/gpu/analyze_end_to_end_device_augmented_lagrangian.py \
  simsopt-end-to-end-device-al.zip docs/gpu_native/figures
```
