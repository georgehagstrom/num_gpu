# The same global run on an NVIDIA GPU. The only differences to the CPU example: load CUDA and
# move the model to the device with adapt(CuArray, ...). Use a GPU with good FP64 (A100, H100).
#
# Run from the repository root:   julia --project=julia/gpu examples/04_global_gpu.jl
using NUMGPU, CUDA, Adapt

s = setup_num_model()
p = GlobalParams(tEnd=365.0)
paths = tm_paths("MITgcm_2.8deg")
g = load_global_model(paths, s; p)                         # built on the host

gd = adapt(CuArray, g)                                     # transport matrices, forcing -> GPU
kd = adapt(CuArray, kernel_setup(s))                       # biology constants -> GPU
res = simulate_global(gd, kd; U=copy(gd.U0), verbose=false)   # add hostsetup=s for annual averages
println(CUDA.name(CUDA.device()), ": ", p.tEnd, " days in ", round(res.wall[end]; digits=1),
        " s (the first run includes compilation)")
save_sim_mat("global_gpu_sim.mat", g, res, paths.dir)      # results come back to the host
