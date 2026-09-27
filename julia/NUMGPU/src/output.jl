# Write simulations in the layout of the Matlab `sim` structs, so the upstream plot scripts can
# read them (scripts/loadJuliaSim.m attaches `p` and `Ntot` for global runs).

"""
    save_sim_mat(path, g::GlobalModel, res, tmdir; nNutrients=3)

`res` is the result of `simulate_global`. Fields: x, y, z, dznom, bathy (from grid.mat),
t (1×nSave), N, DOC, [Si], L, T (nSave×nx×ny×nz), B (nSave×nx×ny×nz×nB), and, for runs with
bCalcAnnualAverages, the annual fields of simulateGlobal. As there, negative B and DOC are 0.
"""
function save_sim_mat(path::AbstractString, g::GlobalModel, res, tmdir::AbstractString; nNutrients=3)
    grid = matread(joinpath(tmdir, "grid.mat"))
    length(unique(g.izBox)) == g.nz ||
        error("empty depth layers: matrixToGrid would drop them; not supported")
    nS = length(res.t)
    field(v) = Float32.(matrix_to_grid(g, v))
    stackt(f) = permutedims(cat((f(k) for k in 1:nS)...; dims=ndims(f(1)) + 1), (ndims(f(1)) + 1, 1:ndims(f(1))...))
    col(j, k) = field(res.U[:, j, k])
    nN = nNutrients
    nB = size(res.U, 2) - nN
    sim = Dict{String,Any}(
        "x" => grid["x"], "y" => grid["y"], "z" => grid["z"], "dznom" => grid["dznom"], "bathy" => grid["bathy"],
        "t" => reshape(collect(res.t), 1, :),
        "N" => stackt(k -> col(1, k)),
        "DOC" => max.(stackt(k -> col(2, k)), 0.0f0),
        "L" => stackt(k -> field(res.L[:, k])),
        "T" => stackt(k -> field(res.T[:, k])),
        "B" => max.(stackt(k -> cat((col(nN + j, k) for j in 1:nB)...; dims=4)), 0.0f0),
    )
    nN > 2 && (sim["Si"] = stackt(k -> col(3, k)))
    # N budget per save period (g), as simulateGlobal (Develop)
    for f in (:Ntot, :Nloss, :NlossHTL, :Nprod)
        haskey(res, f) && (sim[string(f)] = reshape(collect(Float64, res[f]), :, 1))
    end
    if haskey(res, :annual)
        for (f, v) in pairs(res.annual)
            sim[string(f)] = v
        end
    end
    matwrite(path, Dict("sim" => sim); compress=true)
    return path
end

# NamedTuples (rates, functions) become structs in the .mat file
_matvalue(v::NamedTuple) = Dict(string(k) => _matvalue(x) for (k, x) in pairs(v))
_matvalue(v) = v

"""
    save_mat(path, sim::AbstractDict)

Write a water-column or chemostat result (the Dict returned by simulate_watercolumn,
simulate_chemostat, simulate_chemostat_euler) as a Matlab `sim` struct.
"""
function save_mat(path::AbstractString, sim::AbstractDict)
    matwrite(path, Dict("sim" => Dict(k => _matvalue(v) for (k, v) in sim)); compress=true)
    return path
end
