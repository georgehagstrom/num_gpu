# NUMGPU.jl — the NUM model in Julia, for CPUs and GPUs

A port of [NUMmodel](https://github.com/Kenhasteandersen/NUMmodel) (Fortran core + Matlab drivers),
following upstream **Develop at 5644bf5** (2026-09-25; v1.0 plus the mass-conserving TM conversion,
the new NPP definition and recalibration, the gammaDOC and chemostat fixes, and input.yaml).
The biology of each box runs as one kernel thread (KernelAbstractions), so the same code runs
multithreaded on CPUs and on CUDA GPUs. Results agree with the upstream library to round-off
(see *Verification*).

## Feature map

| Upstream (Develop 5644bf5) | Port |
|---|---|
| Group types: generalists, simple generalists, diatoms, simple diatoms, active/passive copepods, POM | `src/setup.jl` (initializers), `src/kernel.jl` (rates and derivatives) |
| Setups: `setupGeneralistsOnly`, `…SimpleOnly`, `…POM`, `…SimplePOM`, `setupDiatomsOnly`, `setupDiatoms_simpleOnly`, `setupGeneralistsDiatoms`, `…Diatoms_simple`, `setupGeneralistsSimpleCopepod`, `setupGeneric`, `setupNUMmodel`, `setupNUMmodelSimple`, `setupGenDiatCope` | `setup_generalists_only()`, … `setup_gen_diat_cope()` (Fortran setup + Matlab wrapper: initial conditions, POM sinking, HTL) |
| `setHTL`, `setMortHTL`, `setSinking`, `setSinkingPOM` | `set_htl!`, `set_mort_htl!`, `set_sinking!`, `set_sinking_pom!` |
| `calcDerivatives`, `simulateEuler` | `calc_derivatives`, `simulate_euler!` (all boxes at once) |
| `getFunctions`, `getRates`, `getLost`, `getBalance`, `simulateEulerFunctions` | `get_functions`, `get_rates`, `get_balance`, `simulate_euler_functions!` (`src/diagnostics.jl`) |
| `simulateGlobal` + `parametersGlobal` (MITgcm 2.8°, ECCO, UVic paths; parday or computed light; restart from `sim`; `bCalcAnnualAverages`) | `load_global_model(tm_paths(name), s; p=GlobalParams(...))`, `simulate_global`, `state_from_sim` |
| `simulateWatercolumn` + `parametersWatercolumn` | `watercolumn_model`, `simulate_watercolumn` |
| `simulateChemostat` (ode23s), `simulateChemostatEuler`, `parametersChemostat` (constant, seasonal, water-column forcing) | `simulate_chemostat` (own `ode23s`), `simulate_chemostat_euler`, `parameters_chemostat` |
| `function_convert_TM_positive(A, dv)` (Develop) | `GlobalParams(TMconversion=:develop)` (default); `:volume` also keeps uniform fields (not upstream), `:v1` is the v1.0 version, `:none` the raw matrices |
| `daily_insolation` (present-day orbit) | `daily_insolation` |
| `sim` structs for the Matlab plot scripts | `save_sim_mat` (global), `save_mat` (water column, chemostat); `scripts/loadJuliaSim.m` |

Not ported: the ~70 Matlab analysis/plotting functions (`plot*`, `panel*`, `calc*`). They read
the `sim` structs the port writes.

## Use

```julia
using NUMGPU
s = setup_num_model()                               # any of the 13 setups
g = load_global_model(tm_paths("MITgcm_2.8deg"), s; p=GlobalParams(tEnd=365))
res = simulate_global(g, s)                         # CPU, multithreaded (julia -t N)

using CUDA, Adapt                                   # GPU: move setup and model, then run
res = simulate_global(adapt(CuArray, g), adapt(CuArray, kernel_setup(s)); U=adapt(CuArray, g.U0))
```

Scripts: `scripts/run_julia_global.jl`, `scripts/spinup_global.jl`, `scripts/gpu_check.jl`.
The upstream checkout (input file, transport matrices) is found at `NUMmodel/` next to this
package, or at `ENV["NUM_ROOT"]`. For annual averages on the GPU, pass the host setup:
`simulate_global(gd, kd; U, hostsetup=s)` with `GlobalParams(bCalcAnnualAverages=true)`.
Naming follows upstream where it maps directly (e.g. `tEnd`, `bCalcAnnualAverages`,
`BC_POMclosed`), so Matlab users recognise the parameters; functions are snake_case.

## Verification

* `julia --project=test test/runtests.jl`: every setup against the upstream Fortran library
  (`NUMmodel/lib/libNUMmodel_matlab.so`, built with `cmake -S . -B build && cmake --build build &&
  cmake --install build` in NUMmodel): setup constants, derivatives on random and trajectory
  states, Euler steps, all functions/rates/balances, simulateEulerFunctions, chemostat Euler
  (1154 checks, round-off agreement), and the GPU-array path (JLArrays).
* `test/global_steps.jl`: the global driver against a dump of Matlab `simulateGlobal`
  (operators bit-identical, first steps to 2e-15).
* `test/driver_refs.jl`: chemostat (ode23s and Euler), water columns, insolation and global
  annual averages against Matlab reference runs (`scripts/make_driver_refs.m`).

Upstream bugs found during the port are reproduced (so results match) and listed in
`docs/upstream_issues.md`.
