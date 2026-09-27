# NUM on GPUs: a Julia port of the NUM plankton model

A port of [NUMmodel](https://github.com/Kenhasteandersen/NUMmodel) (Ken H. Andersen and
co-workers; Fortran core + Matlab drivers) to Julia. The same code runs multithreaded on CPUs and
on NVIDIA GPUs: the biology of every grid box is one kernel thread, and the transport-matrix
step is a sparse matrix product on the device.

The port follows upstream branch **`Develop` at commit 5644bf5** (2026-09-25), which is included
as the git submodule `NUMmodel/`. It covers all group types (generalists, simple generalists,
diatoms, simple diatoms, active and passive copepods, POM), the 13 upstream setups, the
diagnostics (`getFunctions`, `getRates`, `getLost`, `getBalance`) and the drivers: global
(transport matrices), water column and chemostat. Details and the mapping to upstream functions:
[julia/NUMGPU/README.md](julia/NUMGPU/README.md).

This port was written with substantial assistance from an AI model (Anthropic's Claude). Every
component was checked against the upstream Fortran library and Matlab drivers; see
*Verification*. Issues found in upstream while porting, and their status in `Develop`, are in
[docs/upstream_issues.md](docs/upstream_issues.md).

## Performance (measured, full `setupNUMmodel`, 54 state variables per box, Euler step 0.1 d)

| Grid | Boxes | GPU | Time per model year |
|---|---|---|---|
| MITgcm 2.8° | 52,749 | A100 SXM 80 GB | 8.4 s (100 years: 15 min) |
| MITgcm 2.8° | 52,749 | H100 SXM | about 5.6 s |
| MITgcm ECCO 1° | 682,604 | A100 SXM 80 GB | 100 s, plus about 5 min setup; 26 GB GPU memory |

The model needs double precision: use GPUs with full FP64 throughput (A100, H100, V100, GH200).

## Getting started

Requirements: Julia 1.11 (e.g. via [juliaup](https://github.com/JuliaLang/juliaup)); for the tests
also gfortran and cmake; for GPU runs a recent NVIDIA driver (the GPU environment pins the CUDA 12.6
runtime; see [docker/APPTAINER.md](docker/APPTAINER.md) for older drivers).

```bash
git clone --recursive <this repository> num_model_gpu && cd num_model_gpu
julia --project=julia/NUMGPU -e 'using Pkg; Pkg.instantiate()'     # CPU
julia --project=julia/gpu    -e 'using Pkg; Pkg.instantiate()'     # CPU + CUDA (run on a GPU node)
```
The package finds the upstream checkout (input file, transport matrices) at `NUMmodel/`, or at
`ENV["NUM_ROOT"]` if set.

### Transport matrices
They are not part of either repository (several GB). Download the MITgcm configurations of the
Transport Matrix Method from Samar Khatiwala
(<https://sites.google.com/view/samarkhatiwala-research-tmm>) and place them so that:
```
NUMmodel/TMs/MITgcm_2.8deg/grid.mat, config_data.mat
NUMmodel/TMs/MITgcm_2.8deg/Matrix5/Data/boxes.mat
NUMmodel/TMs/MITgcm_2.8deg/Matrix5/TMs/matrix_nocorrection_01.mat ... _12.mat
NUMmodel/TMs/MITgcm_2.8deg/BiogeochemData/Theta_bc.mat
NUMmodel/TMs/MITgcm_ECCO/...          (same layout, Matrix1 instead of Matrix5)
```
`N0.mat`, `Si0.mat`, `parday.mat` and `NUMmodel_init.mat` (initial conditions and light) come with
the NUMmodel repository. All results quoted here used the official MITgcm 2.8° and ECCO sets.

### Run
New users: **[docs/GUIDE.md](docs/GUIDE.md)** compares the upstream Matlab frontend with the Julia
workflow step by step, and `examples/` has short, runnable scripts (chemostat, water column,
global on CPU and GPU, customizing the model).
```julia
# julia --project=julia/NUMGPU -t auto
using NUMGPU
s = setup_num_model()                                           # full setup (Develop defaults)
g = load_global_model(tm_paths("MITgcm_2.8deg"), s; p=GlobalParams(tEnd=365.0))
res = simulate_global(g, s)                                     # CPU, multithreaded

# julia --project=julia/gpu
using NUMGPU, CUDA, Adapt
res = simulate_global(adapt(CuArray, g), adapt(CuArray, kernel_setup(s)); U=adapt(CuArray, g.U0))
```
Scripts (all take `NUM_TMDIR` for the matrix directory):
- `scripts/spinup_global.jl` — multi-year run with annual diagnostics and restart files
  (options in the header: closed POM bottom, Euler step, TM conversion, ECCO).
- `scripts/run_julia_global.jl` — one model year, writes a Matlab-compatible `sim` struct.
- `scripts/gpu_check.jl` — GPU against CPU checks for every setup, plus benchmarks.

On a cluster with Apptainer/SLURM, see [docker/APPTAINER.md](docker/APPTAINER.md) (public image
`ghcr.io/georgehagstrom/num-gpu:e0f0916`). RunPod notes: [docker/RUNPOD.md](docker/RUNPOD.md).

### Figures
`notebooks/run_figures.qmd` (Quarto, R/ggplot) makes maps and time series of a spin-up output;
`notebooks/make_century_notebook.py` makes the same as a Jupyter notebook. Notebooks marked
HISTORICAL are records of NUMmodel v1.0 runs.

## Verification

| Check | How to run | Needs |
|---|---|---|
| TM positivity conversions | `julia --project=julia/NUMGPU/test julia/NUMGPU/test/tm_conversion.jl` | nothing |
| All 13 setups against the Fortran library: derivatives, Euler steps, all diagnostics (1154 checks, round-off agreement); GPU-array path (JLArrays) | `julia --project=julia/NUMGPU/test julia/NUMGPU/test/runtests.jl` | Fortran library built (below); the transport part of the array test needs the 2.8° matrices and is skipped otherwise |
| Global driver against a dump of Matlab `simulateGlobal` (operators bit-identical, first steps to 1e-14) | `julia --project=julia/NUMGPU/test julia/NUMGPU/test/global_steps.jl <dump.mat>` | Matlab; dump from `scripts/dump_global_reference.py` |
| Chemostat, water column, insolation and a 30-day global run against Matlab reference runs | `julia --project=julia/NUMGPU/test julia/NUMGPU/test/driver_refs.jl <refs.mat> [<global.mat>]` | Matlab; refs from `scripts/make_driver_refs.m` |

Build the Fortran reference library (the NUMmodel submodule no longer ships binaries):
```bash
cd NUMmodel && cmake -S . -B build && cmake --build build && cmake --install build
```

## Things to know
- **Time step.** Global NPP depends on the Euler step through upstream's nutrient guard (uptake
  from a nearly empty pool is capped at pool/dt). At 2.8°, year 10: dt 0.1 d gives 140, 0.05 d
  161, 0.025 d 162 PgC per year. 0.1 d is the upstream default; 0.05 d is converged to about 1%
  at 1.6x the cost (`GlobalParams(dt=0.05)`, or `NUM_DT=0.05` in the scripts).
- **Spin-up.** From World Ocean Atlas nitrate the model drifts for decades (NPP 144 to 122 PgC per
  year over 100 years at 2.8°, closed POM bottom) while total N changes by less than 0.3%.
- **Round-off sensitivity.** Nutrient-depleted surface boxes amplify differences of 1e-15 to
  O(1) within days, in this port as in upstream. Compare runs statistically, not box by box.
- **Transport-matrix conversion.** `GlobalParams(TMconversion=...)`: `:develop` (default,
  upstream Develop, conserves mass), `:volume` (also keeps uniform fields uniform; not
  upstream), `:v1` (NUMmodel v1.0, loses about 0.5% of N per year), `:none`.

## License
GPL-3.0 (see [LICENSE](LICENSE)), as upstream NUMmodel, of which this is a derivative.
