# Chemostat: one well-mixed surface box exchanging with a nutrient-rich deep layer.
# Matlab equivalent:
#   p = setupGeneralistsOnly(10); p = parametersChemostat(p); sim = simulateChemostat(p, 100, 10);
#
# Run from the repository root:   julia --project=julia/NUMGPU examples/01_chemostat.jl
using NUMGPU

s = setup_generalists_only(10)            # 10 generalist size classes, nutrients N and DOC
p = parameters_chemostat(s)               # mixing rate p.d = 0.5 /day, 365 days
p.tEnd = 365.0
sim = simulate_chemostat(s, p; L=100.0, T=10.0)   # light (umol photons/m2/s), temperature (C)

nN = s.nNutrients
println("time steps taken by the solver: ", length(sim["t"]))
println("final N = ", round(sim["N"][end]; digits=3), " ugN/l, DOC = ", round(sim["DOC"][end]; digits=3), " ugC/l")
println("final biomass per size class (ugC/l):")
for (m, b) in zip(s.m, sim["B"][end, :])
    println("  mass ", round(m; sigdigits=3), " ugC: ", round(max(b, 0.0); sigdigits=4))  # extinct classes ~0
end
save_mat("chemostat.mat", sim)            # Matlab: S = load('chemostat.mat'); S.sim
println("wrote chemostat.mat")
