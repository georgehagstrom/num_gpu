# Compare a Julia global run (scripts/run_julia_global.jl) with the Matlab reference
# (scripts/run_reference_global.m) at every save time.
# Usage: julia --project=julia/NUMGPU scripts/compare_global_year.jl <matlab.mat> <julia.mat>
using MAT, NUMGPU, Printf
const ROOT = joinpath(@__DIR__, "..", "NUMmodel")

M = matread(ARGS[1]); J = matread(ARGS[2])
sim = M["sim"]
ix, iy, iz = Int.(vec(J["ixBox"])), Int.(vec(J["iyBox"])), Int.(vec(J["izBox"]))
nb = length(ix)
UJ = J["U"][:, :, 1:length(J["t"])]                 # nb×nGrid×nSave, double
nGrid, nSave = size(UJ, 2), size(UJ, 3)
# Matlab clips at the end of simulateGlobal
UJ[:, 2, :] .= max.(UJ[:, 2, :], 0)
UJ[:, 4:end, :] .= max.(UJ[:, 4:end, :], 0)

# Matlab sim (single, nSave×nx×ny×nz[×nB]) -> nb×nGrid×nSave
UM = zeros(nb, nGrid, nSave)
for (j, f) in enumerate(("N", "DOC", "Si"))
    F = sim[f]
    for t in 1:nSave, b in 1:nb
        UM[b, j, t] = F[t, ix[b], iy[b], iz[b]]
    end
end
B = sim["B"]
for k in 1:nGrid-3, t in 1:nSave, b in 1:nb
    UM[b, 3+k, t] = B[t, ix[b], iy[b], iz[b], k]
end

@assert vec(sim["t"]) == vec(J["t"]) "save times differ"
s = setup_num_model(joinpath(ROOT, "input", "input.yaml"))
dv = matread(joinpath(ROOT, "TMs", "MITgcm_2.8deg", "grid.mat"))["dv"]
vol = [dv[ix[b], iy[b], iz[b]] for b in 1:nb] ./ 1000 .* 1e-6
Ntot(U) = sum(vol .* (U[:, 1] .+ vec(sum(U[:, 4:end]; dims=2)) ./ s.rhoCN))

names = vcat(["N", "DOC", "Si"], ["B$k" for k in 1:nGrid-3])
println("Relative difference Julia vs Matlab (Matlab output is single precision, eps = 6e-8)")
@printf("%6s %12s %12s %12s %14s %14s\n", "day", "max|dU|/max", "worst var", "inventory", "Ntot Julia", "Ntot Matlab")
worst_overall = 0.0
for t in 1:nSave
    UJs = Float64.(Float32.(UJ[:, :, t]))          # round like the Matlab output
    d = [maximum(abs.(UJs[:, j] .- UM[:, j, t])) / max(maximum(abs.(UM[:, j, t])), 1e-300) for j in 1:nGrid]
    inv = [abs(sum(vol .* UJ[:, j, t]) - sum(vol .* UM[:, j, t])) / max(abs(sum(vol .* UM[:, j, t])), 1e-300) for j in 1:nGrid]
    global worst_overall = max(worst_overall, maximum(d))
    @printf("%6.1f %12.2e %12s %12.2e %14.8e %14.8e\n", J["t"][t], maximum(d), names[argmax(d)],
            maximum(inv), Ntot(UJ[:, :, t]), sim["Ntot"][t])
end
t = nSave
e = abs.(Float64.(Float32.(UJ[:, :, t])) .- UM[:, :, t]) ./ (abs.(UM[:, :, t]) .+ 1e-6 .* maximum(abs.(UM[:, :, t]); dims=1))
println("final save, per-element relative error (floor 1e-6 of the field max): median ",
        sort(vec(e))[end÷2], ", 99.9th pct ", sort(vec(e))[ceil(Int, 0.999 * length(e))], ", max ", maximum(e))
NJ = [Ntot(UJ[:, :, k]) for k in 1:nSave]
println("Julia N budget, last vs first save: ", NJ[end] / NJ[1] - 1,
        "; Matlab: ", sim["Ntot"][end] / sim["Ntot"][1] - 1)
haskey(M, "info") && println("wall clock: Matlab ", M["info"]["wall_s"], " s on ", M["info"]["nWorkers"],
                            " workers; Julia ", J["wall_s"], " s on 1 thread")
println("max field-relative difference over the year: ", worst_overall)
