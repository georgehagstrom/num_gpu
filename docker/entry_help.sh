#!/bin/bash
# Prints how to run the NUM GPU checks inside the container.
cat <<'TXT'
NUM model GPU port (code in /opt/num_model_gpu, data expected in $NUM_TMDIR)
  nvidia-smi                                            # GPU visible?
  cd /opt/num_model_gpu
  julia scripts/gpu_check.jl /workspace/gpu_check_out   # checks + benchmarks (~5-10 min)
  (first GPU start: CUDA.jl compiles its runtime once, ~2-3 min; see docker/APPTAINER.md)
  julia scripts/run_julia_global.jl 365 /workspace/julia_gpu_1yr.mat   # one model year
TXT
echo "NUM_TMDIR=$NUM_TMDIR  (exists: $( [ -f "$NUM_TMDIR/grid.mat" ] && echo yes || echo NO ))"
