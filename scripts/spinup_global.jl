# Multi-year spin-up of the global model with the fused kernel, with annual diagnostics as in
# Serra-Pompei et al. (2022, GBC): the final year's mean state, integrated net primary production
# and monthly snapshots, plus per-year global N and group biomass to show the spin-up trajectory.
#
# Usage: julia --project=julia/gpu scripts/spinup_global.jl [years=30] [outfile]
#   NUM_DEVICE=gpu|cpu (default gpu), NUM_TMDIR (transport matrices), NUM_RESTART=<file> to start
#   from the final state of an earlier run (its "U_final"), NUM_POM_CLOSED=1 for a closed bottom
#   boundary for POM (no loss of sinking POM through the sea floor), NUM_TM_CONVERSION=develop|volume|v1|none
#   (removal of negative TM entries, default develop; see GlobalParams), NUM_DT (Euler step, default 0.1 d),
#   NUM_NPP_ALL=1 (record NPP every year), NUM_MONTHLY=0 (no final-year monthly snapshots),
#   NUM_CHECKPOINT_YEARS (restart file <outfile>.restart.mat every N years, default 10; 0: none).
#   The matrix folder (Matrix5 for MITgcm_2.8deg, Matrix1 for MITgcm_ECCO) is detected from NUM_TMDIR.
using NUMGPU, MAT, Adapt, Printf, LinearAlgebra
const ROOT = NUMGPU.numroot()
const GPU = get(ENV, "NUM_DEVICE", "gpu") == "gpu"
if GPU
    using CUDA
end
years = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 30
out = length(ARGS) >= 2 ? ARGS[2] : "spinup_$(years)yr.mat"
tmdir = get(ENV, "NUM_TMDIR", joinpath(ROOT, "TMs", "MITgcm_2.8deg"))
sync() = GPU ? CUDA.synchronize() : nothing
dev(x) = GPU ? adapt(CuArray, x) : x

t0 = time()
s = setup_num_model(joinpath(ROOT, "input", "input.yaml"))
p = GlobalParams(tEnd=365.0 * years, dt=parse(Float64, get(ENV, "NUM_DT", "0.1")), BC_POMclosed=get(ENV, "NUM_POM_CLOSED", "0") == "1",
                 TMconversion=Symbol(get(ENV, "NUM_TM_CONVERSION", "develop")))
matrixdir = isdir(joinpath(tmdir, "Matrix5")) ? "Matrix5" : "Matrix1"      # MITgcm_2.8deg : MITgcm_ECCO
g = load_global_model(tmdir, s; p, matrixdir)
kd = dev(kernel_setup(s)); gd = dev(g)
U = haskey(ENV, "NUM_RESTART") ? dev(matread(ENV["NUM_RESTART"])["U_final"]) : copy(gd.U0)
dv = matread(joinpath(tmdir, "grid.mat"))["dv"]
vol = [dv[g.ixBox[b], g.iyBox[b], g.izBox[b]] for b in 1:g.nb]      # m^3
t_setup = time() - t0
println("setup: $(round(t_setup; digits=1)) s; device: ", GPU ? CUDA.name(CUDA.device()) : "CPU",
        "; years: $years; dt: $(p.dt); BC_POMclosed: $(p.BC_POMclosed); TMconversion: $(p.TMconversion)", haskey(ENV, "NUM_RESTART") ? "; restart from $(ENV["NUM_RESTART"])" : "")
flush(stdout)

nPerYear = round(Int, 365 / p.dtTransport)
groups = [(string(gr.type), gr.ix .+ s.nNutrients) for gr in s.groups]
gnames = unique(first.(groups))
# per-year diagnostics (host): global N (g N) and total biomass per group type (g C);
# NUM_NPP_ALL=1 also integrates NPP in every year (per box, and global in PgC/yr)
const NPP_ALL = get(ENV, "NUM_NPP_ALL", "0") == "1"
const MONTHLY = get(ENV, "NUM_MONTHLY", "1") == "1"     # NUM_MONTHLY=0: no final-year monthly snapshots (ECCO)
const CHECKPOINT = parse(Int, get(ENV, "NUM_CHECKPOINT_YEARS", "10"))   # restart file every N years (0: none)
NPP_boxyear = zeros(Float32, g.nb, NPP_ALL ? years : 0); NPP_global = Float64[]
N_year = Float64[]; B_year = zeros(years, length(gnames)); wall_year = Float64[]
function year_diagnostics!(Uh, y)
    push!(N_year, sum(vol .* (Uh[:, 1] .+ vec(sum(Uh[:, 4:end]; dims=2)) ./ s.rhoCN)) * 1e-3)   # mgN/m3*m3 -> gN
    for (k, nm) in enumerate(gnames)
        cols = reduce(vcat, [collect(ix) for (t, ix) in groups if t == nm])
        B_year[y, k] = sum(vol .* vec(sum(Uh[:, cols]; dims=2))) * 1e-3   # mgC/m3 * m3 -> gC
    end
end

work = similar(U); ms = Ref((0, 0))
Usum = zero(U); NPP = similar(U, size(U, 1)); NPPsum = zero(NPP)
Umonth = Matrix{Float64}[]; tmonth = Float64[]
ty = time(); sync()
for i in 1:nPerYear*years
    NUMGPU.transport_step!(U, gd, i, ms, kd, work)
    final = i > nPerYear * (years - 1)
    if final || NPP_ALL
        iL = mod(i, nPerYear) + 1
        net_primary_production!(NPP, U, view(gd.L0, :, iL), view(gd.Tmat, :, ms[][2]), p.dt, kd)
        NPPsum .+= NPP .* p.dtTransport                  # mgC/m3 per year
    end
    if final                                            # final year: accumulate diagnostics
        Usum .+= U
        il = i - nPerYear * (years - 1)
        if floor(il * p.dtTransport / p.tSave) > floor((il - 1) * p.dtTransport / p.tSave) || il == nPerYear
            MONTHLY && (push!(Umonth, Array(U)); push!(tmonth, il * p.dtTransport))
        end
    end
    if i % nPerYear == 0
        sync()
        y = i ÷ nPerYear
        # restart file every CHECKPOINT years (NUM_RESTART=<file> continues from it)
        if CHECKPOINT > 0 && y % CHECKPOINT == 0 && y < years
            matwrite(out * ".restart.mat", Dict("U_final" => Array(U), "year" => y))
        end
        push!(wall_year, time() - ty); global ty = time()
        Uh = Array(U)
        any(isnan, Uh) && error("NaNs in year $y")
        year_diagnostics!(Uh, y)
        if NPP_ALL
            nh = Array(NPPsum)
            NPP_boxyear[:, y] = nh
            push!(NPP_global, sum(nh .* vol) * 1e-3 / 1e15)   # mgC/m3/yr * m3 -> PgC/yr
            y < years && fill!(NPPsum, 0)
        end
        @printf("year %3d: %6.1f s; global N %.6e g; %s%s\n", y, wall_year[end], N_year[end],
                NPP_ALL ? @sprintf("NPP %.2f PgC/yr; ", NPP_global[end]) : "",
                join(["$(gnames[k]) $(round(B_year[y, k] / 1e15; sigdigits=4)) PgC" for k in eachindex(gnames)], ", "))
        flush(stdout)
    end
end

matwrite(out, Dict(
    "U_mean" => Array(Usum) ./ nPerYear,            # final-year mean state, box order (nb x nGrid)
    "NPP_year" => Array(NPPsum),                    # final-year integrated NPP per box, mgC/m3/yr
    "U_month" => MONTHLY ? stack(Umonth) : zeros(0, 0, 0), "t_month" => tmonth, # final-year monthly snapshots
    "U_final" => Array(U),                          # restart state
    "N_year" => N_year, "B_year" => B_year, "group_names" => gnames, "wall_year" => wall_year,
    "NPP_boxyear" => NPP_boxyear, "NPP_global" => NPP_global, "setup_s" => t_setup,
    "ixBox" => g.ixBox, "iyBox" => g.iyBox, "izBox" => g.izBox, "vol" => vol,
    "velocity" => vcat(zeros(s.nNutrients), vec(s.velocity)), "years" => years); compress=false)
println("wrote $out; total $(round(time() - t0; digits=1)) s")
