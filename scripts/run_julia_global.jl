# One model year of the vectorized port on MITgcm_2.8deg (same setup as run_reference_global.m).
# Usage: julia --project=julia/NUMGPU scripts/run_julia_global.jl [tEnd_days] [outfile (default: ./julia_global_2p8_1yr.mat)]
#   NUM_DEVICE=gpu  run on CUDA (use --project=julia/gpu)
#   NUM_TMDIR=...   transport-matrix directory (default NUMmodel/TMs/MITgcm_2.8deg)
# Writes <outfile> (box-order double U) and <outfile>_sim.mat (Matlab sim struct).
using NUMGPU, MAT, Adapt
const ROOT = NUMGPU.numroot()
const GPU = get(ENV, "NUM_DEVICE", "cpu") == "gpu"
if GPU
    using CUDA
end
tEnd = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 365.0
out = length(ARGS) >= 2 ? ARGS[2] : "julia_global_2p8_1yr.mat"      # current directory
tmdir = get(ENV, "NUM_TMDIR", joinpath(ROOT, "TMs", "MITgcm_2.8deg"))

s = setup_num_model(joinpath(ROOT, "input", "input.yaml"))
t0 = time()
g = load_global_model(tmdir, s; p=GlobalParams(tEnd=tEnd))
kb = kernel_setup(s)
sd, gd = GPU ? (adapt(CuArray, kb), adapt(CuArray, g)) : (kb, g)
println("setup + loading: $(round(time() - t0; digits=1)) s; device: ", GPU ? CUDA.name(CUDA.device()) : "CPU"); flush(stdout)
t1 = time()
res = simulate_global(gd, sd; U=copy(gd.U0))
wall = time() - t1
println("WALL CLOCK $(tEnd) days: $(round(wall; digits=1)) s ($(round(wall / 60; digits=1)) min), ",
        GPU ? "GPU" : "threads=$(Threads.nthreads())")
# per-model-year wall clock (year 1 includes compilation of the time-stepping code)
yend = [findlast(<=(365.0 * y + 1e-9), res.t) for y in 1:floor(Int, tEnd / 365)]
for (y, k) in enumerate(yend)
    println("  model year $y: $(round(res.wall[k] - (y == 1 ? 0.0 : res.wall[yend[y-1]]); digits=1)) s")
end
matwrite(out, Dict("U" => res.U, "t" => res.t, "Tmonth" => res.Tmonth, "wall_s" => wall, "wall_at_save" => res.wall,
                   "ixBox" => g.ixBox, "iyBox" => g.iyBox, "izBox" => g.izBox); compress=false)
save_sim_mat(replace(out, r"\.mat$" => "_sim.mat"), g, res, tmdir; nNutrients=s.nNutrients)
