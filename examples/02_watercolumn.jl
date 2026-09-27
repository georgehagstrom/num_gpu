# Water column: one column of the global transport matrices, full model, one year.
# Matlab equivalent:
#   p = setupNUMmodel(); p = parametersWatercolumn(p, 1);   % 1 = MITgcm 2.8 deg
#   sim = simulateWatercolumn(p, 60, -15);                  % lat, lon
#
# Needs the MITgcm_2.8deg transport matrices (see README).
# Run from the repository root:   julia --project=julia/NUMGPU examples/02_watercolumn.jl
using NUMGPU

s = setup_num_model()                     # full model: generalists, diatoms, copepods, POM
w = watercolumn_model(tm_paths("MITgcm_2.8deg"), s; lat=60.0, lon=-15.0,
                      p=WatercolumnParams(tEnd=365.0))
sim = simulate_watercolumn(w, s)

println("column at ", w.lat, "N, ", w.lon, "E: ", w.nz, " layers, depths ", round.(w.z; digits=0), " m")
ms = [1, 91, 182, 274, 365]
println("surface nitrogen (ugN/l) on days ", ms, ": ", round.(sim["N"][ms, 1]; digits=2))
println("surface biomass (ugC/l) on the same days: ", round.(vec(sum(sim["B"][ms, 1, :]; dims=2)); digits=2))
save_mat("watercolumn.mat", sim)          # same fields as the Matlab sim struct (N, DOC, Si, B, L, T, t, z)
println("wrote watercolumn.mat")
