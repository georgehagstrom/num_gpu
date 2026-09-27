# Running NUM: the Matlab frontend and the Julia port

There are two programs:

- **Upstream NUMmodel** (`NUMmodel/`, Ken H. Andersen et al.): a Matlab frontend that calls a
  compiled Fortran library for the biology of each grid box. This is how the NUM group works.
- **This port** (`julia/NUMGPU`): the same model in Julia, for many CPU threads or a GPU. It has
  no Matlab frontend, but the steps map one to one, and its results can be loaded into Matlab
  and plotted with the upstream plot functions.

A good split: small experiments (chemostat, water columns, short global runs) in whichever you
know; long or large global runs (spin-ups, ECCO, ensembles) with the port.

## Side by side

| Task | Matlab (in `NUMmodel/matlab`) | Julia (`julia --project=julia/NUMGPU -t auto`, then `using NUMGPU`) |
|---|---|---|
| Full model setup | `p = setupNUMmodel();` | `s = setup_num_model()` |
| Other setups | `setupGeneralistsOnly(10)`, `setupGeneric([0.1 10])`, ... | `setup_generalists_only(10)`, `setup_generic([0.1, 10.0])`, ... (all 13, snake_case) |
| HTL mortality | `setHTL(mortHTL, mHTL, bQuad, bDecl, bCopOnly)` | `set_htl!(s, mHTL, mortHTL, quad, decl, copOnly)` — **note: first two swapped** |
| POM sinking | `setSinkingPOM(p, 20)` | `set_sinking_pom!(s, fill(20.0, length(s.ixPOM)))` |
| Chemostat | `p = parametersChemostat(p); sim = simulateChemostat(p, L, T);` | `p = parameters_chemostat(s); sim = simulate_chemostat(s, p; L, T)` |
| Water column | `p = parametersWatercolumn(p, 1); sim = simulateWatercolumn(p, lat, lon);` | `w = watercolumn_model(tm_paths("MITgcm_2.8deg"), s; lat, lon); sim = simulate_watercolumn(w, s)` |
| Global parameters | `p = parametersGlobal(p); p.tEnd = 365; p.BC_POMclosed = true;` | `p = GlobalParams(tEnd=365.0, BC_POMclosed=true)` |
| Global run | `sim = simulateGlobal(p);` | `g = load_global_model(tm_paths("MITgcm_2.8deg"), s; p); res = simulate_global(g, s)` |
| Annual averages | `simulateGlobal(p, bCalcAnnualAverages=true)` | `GlobalParams(..., bCalcAnnualAverages=true)`; results in `res.annual` |
| Parallel | `parpool(8); setupNUMmodel(bParallel=true)` | start Julia with `-t 8` (threads); or the GPU, below |
| GPU | — | `using CUDA, Adapt`; `simulate_global(adapt(CuArray, g), adapt(CuArray, kernel_setup(s)); U=adapt(CuArray, g.U0))` |
| ECCO 1° grid | `parametersGlobal(p, 2)` | `tm_paths("MITgcm_ECCO")` |
| Restart from a run | `simulateGlobal(p, sim)` | `simulate_global(g, s; U=state_from_sim(g, s, sim))`, or `NUM_RESTART=<file>` in the spin-up script |
| Save results | `sim` is in the workspace | `save_sim_mat("run_sim.mat", g, res, tm_paths("MITgcm_2.8deg").dir)` (global); `save_mat("wc.mat", sim)` (water column, chemostat) |
| Plot | `plotGlobal(sim)`, `plotWatercolumn(sim)`, ... | load into Matlab (below), or `notebooks/run_figures.qmd` for spin-ups |
| Model parameters (input file) | edit `NUMmodel/input/input.yaml` | the same file is read at setup; pass a modified copy: `setup_num_model("my/input.yaml")` |

Units are the same everywhere: µg C/l for biomass and DOC, µg N/l for N, days, µmol photons
m⁻² s⁻¹ for light.

## Global parameters (`GlobalParams`, defaults = upstream Develop `parametersGlobal`)

| Field | Default | Matlab `p` field | Meaning |
|---|---|---|---|
| `tEnd` | 365 | `p.tEnd` | simulated days |
| `tSave` | 365/12 | `p.tSave` | save interval (days) |
| `dt` | 0.1 | `p.dt` | Euler step of the biology (days); 0.05 gives converged NPP (README) |
| `dtTransport` | 0.5 | `p.dtTransport` | transport step (days); upstream assumes 0.5 |
| `BC_POMclosed` | false | `p.BC_POMclosed` | true: POM stays in the bottom cell |
| `BCmixing`, `BCvalue` | [1,0,1]/365, initial bottom values | same | nutrient exchange at the sea floor |
| `kw` | 0.1 | `p.kw` | light attenuation by water (1/m) |
| `bUse_parday_light` | true | same | false: clear-sky insolation instead of the parday climatology |
| `bCalcAnnualAverages` | false | option of `simulateGlobal` | annual NPP, HTL production, pico/nano/micro |
| `TMconversion` | `:develop` | (fixed upstream) | removal of negative transport-matrix entries |

## Example scripts (`examples/`)

Run from the repository root. Times: laptop, 6 CPU threads, including Julia's compilation.

| Script | What it does | Time |
|---|---|---|
| `01_chemostat.jl` | generalists-only chemostat for a year; prints the size spectrum; writes `chemostat.mat` | ~20 s |
| `02_watercolumn.jl` | full model, one water column (60°N, 15°W), one year; writes `watercolumn.mat` | ~1 min |
| `03_global_cpu.jl` | full model, global 2.8°, 30 days, annual averages; writes a Matlab `sim` struct | ~2 min |
| `04_global_gpu.jl` | the same for a year on an NVIDIA GPU (`--project=julia/gpu`) | ~1-1.5 min on an A100 (setup and compilation; the year itself ~8 s) |
| `05_customize.jl` | changing setups, HTL, sinking and the global options | seconds |

```bash
julia --project=julia/NUMGPU examples/01_chemostat.jl
julia --project=julia/NUMGPU -t auto examples/03_global_cpu.jl
julia --project=julia/gpu examples/04_global_gpu.jl
```

For long runs use `scripts/spinup_global.jl` (options as environment variables, listed in its
header), locally or on a cluster (`docker/APPTAINER.md`).

## Running the examples and scripts in the container

The image (`ghcr.io/georgehagstrom/num-gpu:e0f0916`) contains Julia, the packages, the port and
`scripts/`, but no data. Mount (`-v`, `--bind`) the repository and an output folder, and point
`NUM_ROOT` at the mounted `NUMmodel/` (input file and transport matrices). Tested:

```bash
# Docker (laptop or workstation), CPU; run from the repository root
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e JULIA_DEPOT_PATH=/tmp/.julia:/opt/julia-depot \
    -e JULIA_NUM_THREADS=8 -e NUM_ROOT=/repo/NUMmodel \
    -v $PWD:/repo:ro -v $PWD/out:/out -w /out \
    ghcr.io/georgehagstrom/num-gpu:e0f0916 julia /repo/examples/03_global_cpu.jl
# add --gpus all (and use examples/04_global_gpu.jl) on a machine with an NVIDIA GPU

# Apptainer (cluster); see docker/APPTAINER.md for the one-time GPU warm-up
apptainer exec --nv --bind $PWD:/repo --bind $SCRATCH/num:/workspace --env NUM_ROOT=/repo/NUMmodel \
    num-gpu.sif bash -c 'cd /workspace && julia /repo/examples/04_global_gpu.jl'
```
`--user` and `JULIA_DEPOT_PATH` make the output files yours and give Julia a writable cache
directory; without them Docker writes as root. The scripts work the same way
(`julia /opt/num_model_gpu/scripts/spinup_global.jl ...`, or the mounted copy in `/repo/scripts`).

## Loading Julia results into Matlab

```matlab
cd NUMmodel/matlab                        % upstream functions must be on the path
addpath('../../scripts')                  % loadJuliaSim.m
sim = loadJuliaSim('/path/to/global_30d_sim.mat');   % attaches p (setupNUMmodel, 2.8 deg) and Ntot
plotGlobal(sim)
```
`loadJuliaSim` assumes the full `setupNUMmodel` on MITgcm 2.8°. Save more than one time
(`tSave` smaller than `tEnd`): upstream `plotGlobal` needs at least two saved times. Water-column
and chemostat files from `save_mat` load with `S = load('watercolumn.mat'); sim = S.sim;`.

## Working in the Julia REPL

- The first call of each function compiles it (seconds to a minute); keep the session open and
  later calls are fast. Start Julia with threads: `julia --project=julia/NUMGPU -t auto`.
- Results are Julia `Dict`s or `NamedTuple`s with the same field names as Matlab's `sim`
  (`sim["N"]`, `res.U`, `res.t`); arrays are 1-based and column-major as in Matlab.
- Help on any function: `?simulate_global`. The upstream-to-port function map is in
  `julia/NUMGPU/README.md`.
