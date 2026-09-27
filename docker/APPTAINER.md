# Running the image on an HPC cluster (Apptainer / Singularity + SLURM)

The image `ghcr.io/georgehagstrom/num-gpu:e0f0916` (public) contains Julia 1.11, CUDA.jl and the
precompiled port. It is built on RunPod's generic Ubuntu 22.04 base image; the RunPod parts (its
`/start.sh`, SSH, Jupyter) are simply not used under Apptainer. Data are not in the image.

## Requirements
- An NVIDIA GPU with good FP64 throughput (A100, H100, V100, GH200). The model runs in double
  precision; L40S, RTX and A10-class cards are 30-60x slower in FP64.
- A recent NVIDIA driver (check with `nvidia-smi`). The image pins the CUDA 12.6 runtime
  (`julia/gpu/LocalPreferences.toml`); driver 560 or newer is safe, and older CUDA 12.x drivers
  may work through CUDA's minor-version compatibility. If CUDA.jl reports that the driver is too
  old, change the pin to a CUDA version the driver supports (e.g. `version = "12.2"` for driver
  535) and rebuild the image, or use the no-container route below.

## Get the image and the data
```bash
apptainer pull num-gpu.sif docker://ghcr.io/georgehagstrom/num-gpu:e0f0916
mkdir -p $SCRATCH/num            # writable directory, bound to /workspace in the container
# transport matrices (see the top-level README): $SCRATCH/num/MITgcm_2.8deg, $SCRATCH/num/MITgcm_ECCO
```

## Three things to get right
1. **Use `apptainer exec --nv`**, not `run`: `run` starts RunPod's start script; `--nv` exposes the GPU.
2. **Bind a writable directory to `/workspace`**. The image keeps Julia's writable depot there
   (`JULIA_DEPOT_PATH=/workspace/.julia:/opt/julia-depot`) and expects the data in
   `/workspace/MITgcm_2.8deg` (`NUM_TMDIR`). Without it, Julia cannot write its caches.
3. **Warm up once on a GPU node.** The image is built without a GPU, so on the first GPU start
   CUDA.jl has to select and compile its runtime (about 2-3 minutes; the error message
   "could not find an appropriate CUDA runtime" on the first call is expected). The result is
   cached in `/workspace/.julia` and later jobs start quickly. Do this before submitting arrays:
```bash
apptainer exec --nv --bind $SCRATCH/num:/workspace num-gpu.sif bash -c '
  julia -e "using CUDA; CUDA.set_runtime_version!(v\"12.6\")"
  julia -e "using CUDA; CUDA.versioninfo()"'
```

## Checks and a run
```bash
apptainer exec --nv --bind $SCRATCH/num:/workspace num-gpu.sif bash -c '
  cd /opt/num_model_gpu && julia scripts/gpu_check.jl /workspace/gpu_check_out'   # ends with ALL CHECKS PASSED
```

Example SLURM job (100-year spin-up at 2.8°, about 15 minutes on an A100):
```bash
#!/bin/bash
#SBATCH --job-name=num-spinup
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
cd $SCRATCH/num
apptainer exec --nv --bind $SCRATCH/num:/workspace $HOME/num-gpu.sif bash -c '
  cd /workspace && NUM_NPP_ALL=1 julia /opt/num_model_gpu/scripts/spinup_global.jl 100 spinup_100yr.mat'
```
For ECCO (1°, about 100 s per model year on an A100 plus ~5 min setup, 26 GB GPU memory):
`NUM_TMDIR=/workspace/MITgcm_ECCO NUM_MONTHLY=0 ...`, and `--mem=128G` or more (the monthly
matrices are loaded and raised to a power on the host). Restart files are written every
`NUM_CHECKPOINT_YEARS` (default 10) years; continue with `NUM_RESTART=<file>`.

## Without a container
On a cluster with Julia (or `juliaup`), clone the repository and instantiate on a **GPU node**:
```bash
git clone --recursive <repo> && cd <repo>
julia --project=julia/gpu -e 'using Pkg; Pkg.instantiate(); using CUDA; CUDA.versioninfo()'
```
Precompiling where a GPU is present avoids the runtime-selection problem entirely. To use the
cluster's CUDA module instead of CUDA.jl's downloaded runtime:
`julia --project=julia/gpu -e 'using CUDA; CUDA.set_runtime_version!(local_toolkit=true)'`
(after `module load cuda`), then restart Julia.
