# Running the NUM GPU checks on RunPod (optional; for HPC clusters see APPTAINER.md)

Needs: a RunPod account with prepaid credit, the image pushed to a registry
(`ghcr.io/georgehagstrom/num-gpu:<tag>`, public), and the data tarball from `scripts/pack_data.sh`.
Budget: one A100 (80 GB) session of 1–2 h, about $1.20–1.60/h on-demand.

## 1. Template (once)
Console → Templates → New Template
- Container image: `ghcr.io/georgehagstrom/num-gpu:<tag>` (tag = git commit, see `docker/push.sh`)
- Container disk: 20 GB; Volume disk: 10 GB, mounted at `/workspace`
- Expose TCP port 22 (SSH) and HTTP port 8888 (Jupyter), as in RunPod's base-image templates
- Private image: RunPod needs a GitHub **classic** personal access token with only the
  `read:packages` scope (github.com → Settings → Developer settings → Personal access tokens →
  Tokens (classic)). Add it in RunPod under Settings → Container Registry Auth
  (username `georgehagstrom`, password = the token) and select it in the template.

## 2. Pod
Pods → Deploy → pick **A100 80GB** (or H100) → your template → On-Demand → Deploy.
Avoid L4/L40S/RTX cards for timing: their FP64 is 1/32–1/64 of FP32 (fine for a smoke test only).

## 3. Data
On your machine: `scripts/pack_data.sh`, then `runpodctl send num_tm_2p8.tar` (prints a code).
In the pod's web terminal: `cd /workspace && runpodctl receive <code> && tar -xf num_tm_2p8.tar`
(or `scp` over the pod's SSH port). This gives `/workspace/MITgcm_2.8deg`, the default `NUM_TMDIR`.

## 4. Run
```
num-help                       # shows the commands and checks the data path
nvidia-smi
cd /opt/num_model_gpu
julia scripts/gpu_check.jl /workspace/gpu_check_out 2>&1 | tee /workspace/gpu_check.txt
julia scripts/run_julia_global.jl 365 /workspace/julia_gpu_1yr.mat   # optional: a full year
```
The first CUDA call downloads CUDA libraries (~1–2 min). `gpu_check.jl` ends with
`ALL CHECKS PASSED` or lists failures.

## 5. Bring results back, then stop the pod
`runpodctl send /workspace/gpu_check_out/gpu_check.log` (and the `.mat` files if wanted).
**Stop or terminate the pod when done** — billing continues while it runs; a stopped pod
still bills its volume disk (~$0.20/GB/month idle).
