# NUM model GPU port: Julia 1.11 + CUDA.jl environment, precompiled.
# Base: RunPod's minimal Ubuntu 22.04 image (keeps its /start.sh, which starts SSH/Jupyter on
# RunPod). No system CUDA is needed: CUDA.jl downloads its own CUDA libraries on first use; the
# host only needs an NVIDIA driver. Works elsewhere with `docker run --gpus all` or Apptainer.
#
# Build (from the repo root):  docker build --platform linux/amd64 -f docker/Dockerfile -t num-gpu .
# Data is NOT in the image: mount/copy the transport-matrix folder to $NUM_TMDIR.
FROM runpod/base:0.7.0-jammy

ARG JULIA_VERSION=1.11.5
ARG PRECOMPILE_TASKS=4
RUN curl -fsSL https://julialang-s3.julialang.org/bin/linux/x64/${JULIA_VERSION%.*}/julia-${JULIA_VERSION}-linux-x86_64.tar.gz \
      | tar -xz -C /opt && ln -s /opt/julia-${JULIA_VERSION}/bin/julia /usr/local/bin/julia

WORKDIR /opt/num_model_gpu
COPY julia/ julia/
COPY scripts/ scripts/
COPY NUMmodel/input/ NUMmodel/input/
COPY docker/entry_help.sh /usr/local/bin/num-help

# julia/gpu/LocalPreferences.toml pins the CUDA runtime (12.6): the image is built without an
# NVIDIA driver, and without the pin CUDA.jl's cached runtime selection is empty on the pod
# ("could not find an appropriate CUDA runtime"). Pods need a host driver for CUDA >= 12.6.
# Packages and compiled code go to a shared depot baked into the image; at run time the user
# depot (~/.julia, writable) comes first, so CUDA's runtime download and any new packages go
# there while the baked caches are reused.
ENV JULIA_DEPOT_PATH=/opt/julia-depot
# Compile package images for generic x86-64 plus common microarchitectures (the target list of
# the official Julia binaries); otherwise the caches only match the build machine's CPU and the
# pod recompiles everything (~3.5 min on the first GPU run).
ENV JULIA_CPU_TARGET="generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)"
RUN JULIA_NUM_PRECOMPILE_TASKS=${PRECOMPILE_TASKS} julia --project=julia/gpu -e \
      'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA, NUMGPU, MAT, Adapt; println("ok")' \
 && JULIA_NUM_PRECOMPILE_TASKS=${PRECOMPILE_TASKS} julia --project=julia/NUMGPU -e \
      'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
 && chmod -R a+rX /opt/julia-depot

# The writable depot is on the pod's persistent volume, so the one-time CUDA runtime compile on the
# first GPU start (see LocalPreferences note above) is kept across pod restarts.
ENV JULIA_DEPOT_PATH=/workspace/.julia:/opt/julia-depot \
    JULIA_PROJECT=/opt/num_model_gpu/julia/gpu \
    NUM_TMDIR=/workspace/MITgcm_2.8deg \
    NUM_DEVICE=gpu
# CMD stays the base image's /start.sh (SSH, Jupyter, then sleeps) so the pod stays up.
