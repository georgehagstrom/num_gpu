# Global run on the CPU (multithreaded): 30 days at 2.8 degrees with annual-average diagnostics.
# Matlab equivalent:
#   p = setupNUMmodel(bParallel=true); p = parametersGlobal(p); p.tEnd = 30; p.tSave = 10;
#   sim = simulateGlobal(p, bCalcAnnualAverages=true); plotGlobal(sim)
#
# Needs the MITgcm_2.8deg transport matrices (see README).
# Run from the repository root:   julia --project=julia/NUMGPU -t auto examples/03_global_cpu.jl
using NUMGPU

s = setup_num_model()
p = GlobalParams(tEnd=30.0, tSave=10.0, bCalcAnnualAverages=true)   # any Matlab p.xxx: GlobalParams(xxx=...)
# (save at least twice: upstream plotGlobal needs more than one saved time)
paths = tm_paths("MITgcm_2.8deg")
g = load_global_model(paths, s; p)
println("loaded ", g.nb, " boxes; running on ", Threads.nthreads(), " threads")

res = simulate_global(g, s)
println("saved at days ", res.t, "; wall clock ", round(res.wall[end]; digits=1), " s")
println("global N at the saves (g): ", res.Ntot)

# Matlab-compatible sim struct (N, DOC, Si, B, L, T as nSave x nx x ny x nz, annual averages):
save_sim_mat("global_30d_sim.mat", g, res, paths.dir)
println("wrote global_30d_sim.mat; in Matlab: sim = loadJuliaSim('global_30d_sim.mat'); plotGlobal(sim)")
