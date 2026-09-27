# Global transport-matrix driver, mirroring NUMmodel/matlab/simulateGlobal.m and
# parametersGlobal.m (upstream Develop 5644bf5): per 0.5 d transport step, biology (Euler) in every box, sinking,
# bottom boundary condition for nutrients, then u = Aimp*(Aexp*u). Monthly matrices and
# temperature switch on the same step indices as upstream. Options: parday or computed light,
# the three upstream grids, restart from a previous simulation, and the annual averages of
# ecosystem functions (bCalcAnnualAverages).

using MAT
using SparseArrays

const SECONDS_PER_DAY = 24 * 60 * 60
const DAYS_BEFORE_MONTH = [0 31 28 31 30 31 30 31 31 30 31 30]   # simulateGlobal.m `mon`

Base.@kwdef struct GlobalParams
    dt::Float64 = 0.1                 # Euler step (days)
    dtTransport::Float64 = 0.5        # TM step (days)
    tEnd::Float64 = 365.0
    tSave::Float64 = 365 / 12
    bTransport::Bool = true
    BCmixing::Vector{Float64} = [1.0, 0.0, 1.0] ./ 365
    BCvalue::Union{Nothing,Vector{Float64}} = nothing   # nothing (= -1): bottom value of the initial state
    BC_POMclosed::Bool = false
    bUse_parday_light::Bool = true    # false: daily_insolation (clear sky)
    kw::Float64 = 0.1                 # light attenuation by water (1/m); 0.06 in v1.0
    EinConv::Float64 = 4.57           # W m^-2 -> umol photons m^-2 s^-1
    PARfrac::Float64 = 0.4
    bCalcAnnualAverages::Bool = false
    # removal of negative off-diagonals of the TMs (function_convert_TM_positive):
    #   :develop  upstream Develop (40ea16d): conserves tracer mass (volume-weighted column sums)
    #   :volume   the move applied to volume-weighted fluxes: conserves mass AND keeps uniform
    #             fields uniform (port option, not upstream)
    #   :v1       upstream v1.0: keeps uniform fields but loses mass (~0.5%/yr of N at 2.8°)
    #   :none     raw matrices
    TMconversion::Symbol = :develop
end

"""
Paths of an upstream transport-matrix configuration (parametersGlobal nTMmodel 1, 2, 3):
"MITgcm_2.8deg", "MITgcm_ECCO", "UVicOSUpicdefault" under `tmroot` (NUMmodel/TMs).
"""
function tm_paths(name::AbstractString; tmroot=joinpath(numroot(), "TMs"))
    dir = joinpath(tmroot, name)
    matrixdir = name == "MITgcm_2.8deg" ? "Matrix5" : "Matrix1"
    n0 = name == "UVicOSUpicdefault" ? joinpath(tmroot, "UVicOSUpicdefault_N0.mat") : joinpath(dir, "N0.mat")
    return (; dir, matrixdir, n0file=n0, si0file=joinpath(dir, "Si0.mat"), pardayfile=joinpath(dir, "parday.mat"))
end

"""
Everything the time loop needs, in box order (boxes.mat). Arrays live on the host; `adapt`
them to the device for GPU runs.
"""
struct GlobalModel{M<:AbstractMatrix{Float64},V<:AbstractVector{Float64},I<:AbstractVector{Int},S}
    p::GlobalParams
    nb::Int
    ixBox::Vector{Int}; iyBox::Vector{Int}; izBox::Vector{Int}
    nx::Int; ny::Int; nz::Int
    dznom::Vector{Float64}
    U0::M                              # nb×nGrid initial state
    L0::M                              # nb×(365/dtTransport) light
    Tmat::M                            # nb×12 temperature
    Aexp::Vector{S}                    # 12 monthly I + dtT*Aexp (after p.TMconversion)
    Aimp::Vector{S}                    # 12 monthly Aimp^(dtT/deltaT)
    # sinking (explicit upwind): U[:,j] = keep .* U[:,j] .+ gain .* U[above,j]
    ixSink::Vector{Int}                # state columns with nonzero velocity
    above::I                           # box above (itself at the surface, with gain 0)
    keep::M                            # nb×nSink
    gain::M                            # nb×nSink
    # bottom BC: u[ixBottom,k] += dtT*BCmixing[k]/dzBottom*(BCvalue[:,k] - u[ixBottom,k])
    ixBottom::I
    dzBottom::V
    BCvalue::M                         # nBottom×nNutrients
    dvBox::V                           # box volumes (m^3), for the N budget
end

"""
    function_convert_TM_positive(A)

Port of the upstream Matlab function: moves negative off-diagonal entries to the transposed
position with opposite sign and fixes the diagonal so column and row sums are preserved.
"""
function function_convert_TM_positive(A::SparseMatrixCSC)
    N = A - spdiagm(diag(A))
    N = N .* (N .< 0)
    dropzeros!(N)
    A2 = A - N + (-sparse(N'))
    # column and row sums accumulated sequentially over stored entries (Matlab's order)
    n = size(N, 2)
    csum = zeros(n); rsum = zeros(size(N, 1))
    rv = rowvals(N); nzv = nonzeros(N)
    for j in 1:n, k in nzrange(N, j)
        csum[j] += nzv[k]
        rsum[rv[k]] += nzv[k]
    end
    DA = diag(A2) .+ csum .+ rsum
    return A2 - spdiagm(diag(A2)) + spdiagm(DA)
end

"""
    convert_TM_positive_volume(A, v)

Mass-conserving variant of `function_convert_TM_positive` (not in upstream): the same move is
applied to the volume-weighted operator `diag(v)*A` (mass fluxes), so a negative entry
A[i,j] becomes the reversed flux A[j,i] = v[i]*|A[i,j]|/v[j]. Preserves the row sums of A
(a uniform field stays uniform) and the volume-weighted column sums v'A (tracer mass).
"""
convert_TM_positive_volume(A::SparseMatrixCSC, v::AbstractVector) =
    sparse(Diagonal(1 ./ v) * function_convert_TM_positive(sparse(Diagonal(v) * A)))

"""
    convert_TM_positive_develop(A, dv)

upstream Develop (40ea16d) function_convert_TM_positive(A, dv): negative off-diagonals are
moved to the transposed position, then the diagonal is set so that the volume-weighted column
sums dv'A are unchanged (mass conservation). Row sums change, so a uniform field does not stay
exactly uniform (see `convert_TM_positive_volume`).
"""
function convert_TM_positive_develop(A::SparseMatrixCSC, dv::AbstractVector)
    n = size(A, 1); w = reshape(dv, 1, n)
    N = A - spdiagm(diag(A)); N = N .* (N .< 0); dropzeros!(N)
    A2 = A - N - sparse(N'); A2 = A2 - spdiagm(diag(A2))
    DA = vec((w * A - w * A2) ./ w)
    return A2 + spdiagm(DA)
end

function convert_TM(A::SparseMatrixCSC, how::Symbol, v)
    how === :develop && return convert_TM_positive_develop(A, v)
    how === :v1 && return function_convert_TM_positive(A)
    how === :volume && return convert_TM_positive_volume(A, v)
    how === :none && return A
    error("TMconversion must be :develop, :volume, :v1 or :none, got :$how")
end

# Matlab's mpower for a positive integer power, reproduced bit for bit (checked against
# Aimp^36 of simulateGlobal with deltaT = 1200 s): bits from the least significant; the running
# square P multiplies the result from the left.
function sparse_power(A::AbstractMatrix, k::Real)
    isinteger(k) && k >= 1 || error("Aimp power $k: only positive integer powers are supported")
    k = Int(k)
    res = nothing; P = A
    while k > 0
        isodd(k) && (res = res === nothing ? copy(P) : P * res)
        k >>= 1
        k > 0 && (P = P * P)
    end
    return res
end

vecint(x) = Int.(vec(x))

"""
    load_global_model(tmdir, s::NUMSetup; p=GlobalParams(), matrixdir="Matrix5",
                      n0file, si0file, pardayfile) -> GlobalModel

Reads a transport-matrix configuration (grid.mat, <matrixdir>/Data/boxes.mat, monthly
<matrixdir>/TMs/matrix_nocorrection_XX.mat, BiogeochemData/Theta_bc.mat, optional parday.mat,
N0.mat, Si0.mat) and precomputes the operators exactly as simulateGlobal does. For the other
upstream grids use `load_global_model(tm_paths("MITgcm_ECCO"), s)`.
"""
function load_global_model(tmdir::AbstractString, s::NUMSetup; p::GlobalParams=GlobalParams(),
                           matrixdir="Matrix5", n0file=joinpath(tmdir, "N0.mat"),
                           si0file=joinpath(tmdir, "Si0.mat"), pardayfile=joinpath(tmdir, "parday.mat"))
    isfile(joinpath(tmdir, "grid.mat")) ||
        error("transport matrices not found in $tmdir (grid.mat missing). They are not part of the repository; " *
              "see the top-level README, section \"Transport matrices\".")
    isfile(joinpath(tmdir, matrixdir, "Data", "boxes.mat")) ||
        error("$(joinpath(tmdir, matrixdir)) not found: use matrixdir=\"Matrix5\" for MITgcm_2.8deg, " *
              "\"Matrix1\" for MITgcm_ECCO (or load_global_model(tm_paths(name), s)).")
    # the drivers follow upstream, which assumes two transport steps per day (light table, months)
    p.dtTransport == 0.5 || error("dtTransport must be 0.5 d (upstream simulateGlobal assumes it), got $(p.dtTransport)")
    isapprox(p.dtTransport / p.dt, round(p.dtTransport / p.dt); atol=1e-9) ||
        error("dt = $(p.dt) must divide dtTransport = $(p.dtTransport)")
    p.TMconversion in (:develop, :volume, :v1, :none) ||
        error("TMconversion must be :develop, :volume, :v1 or :none, got :$(p.TMconversion)")
    grid = matread(joinpath(tmdir, "grid.mat"))
    boxes = matread(joinpath(tmdir, matrixdir, "Data", "boxes.mat"))
    nb = Int(boxes["nb"])
    ix, iy, iz = vecint(boxes["ixBox"]), vecint(boxes["iyBox"]), vecint(boxes["izBox"])
    Ybox, Zbox = vec(boxes["Ybox"]), vec(boxes["Zbox"])
    nx, ny, nz = Int(grid["nx"]), Int(grid["ny"]), Int(grid["nz"])
    dznom = vec(grid["dznom"])
    deltaT = grid["deltaT"]
    lin3 = LinearIndices((nx, ny, nz))
    idx3 = [lin3[ix[b], iy[b], iz[b]] for b in 1:nb]
    idx2 = [LinearIndices((nx, ny))[ix[b], iy[b]] for b in 1:nb]
    to_boxes(G) = Float64.(G[idx3])   # gridToMatrix for one 3-d field

    # initial conditions: N0/Si0 fields if present, otherwise p.u0 (simulateGlobal)
    nN = s.nNutrients
    U0 = repeat(permutedims(initial_state(s)), nb)
    isfile(n0file) && (U0[:, 1] = to_boxes(matread(n0file)["N"]))
    nN > 2 && isfile(si0file) && (U0[:, 3] = to_boxes(matread(si0file)["Si"]))

    # temperature
    Tbc = matread(joinpath(tmdir, "BiogeochemData", "Theta_bc.mat"))["Tbc"]
    Tmat = reduce(hcat, [to_boxes(Tbc[:, :, :, m]) for m in 1:12])

    # light
    nL = round(Int, 365 / p.dtTransport)
    att = exp.(-p.kw .* Zbox)
    if p.bUse_parday_light
        isfile(pardayfile) || error("PARday file does not exist. Set bUse_parday_light = false")
        parday = matread(pardayfile)["parday"]   # copied to all depths: surface value at (ix, iy)
        size(parday, 4) == nL || error("parday has $(size(parday, 4)) steps, expected $nL")
        PD = reshape(Float64.(parday), nx * ny, nL)
        L0 = 1e6 .* PD[idx2, :] ./ SECONDS_PER_DAY .* att
    else
        # simulateGlobal: daily_insolation(0, Ybox, i/2, 1) (day i/2 is hard-coded upstream)
        L0 = reduce(hcat, [p.EinConv * p.PARfrac .* daily_insolation.(Ybox, i / 2) .* att for i in 1:nL])
    end

    # box volumes (simulateGlobal: gridToMatrix(grid.dv)), for the TM conversion and the N budget
    dvBox = to_boxes(grid["dv"])

    # monthly transport operators
    Ix = sparse(1.0I, nb, nb)
    Aexp = Vector{SparseMatrixCSC{Float64,Int}}(undef, 12)
    Aimp = similar(Aexp)
    for m in 1:12
        TM = matread(joinpath(tmdir, matrixdir, "TMs", "matrix_nocorrection_" * lpad(m, 2, '0') * ".mat"))
        Ae = convert_TM(TM["Aexp"], p.TMconversion, dvBox)
        Ai = convert_TM(TM["Aimp"], p.TMconversion, dvBox)
        Aexp[m] = Ix + (p.dtTransport * SECONDS_PER_DAY) * Ae
        Aimp[m] = sparse_power(Ai, p.dtTransport * SECONDS_PER_DAY / deltaT)
    end

    # water columns: box index per (ix, iy, iz)
    col = zeros(Int, nx, ny, nz)
    for b in 1:nb
        col[ix[b], iy[b], iz[b]] = b
    end
    above = collect(1:nb)
    for b in 1:nb
        iz[b] > 1 && col[ix[b], iy[b], iz[b]-1] > 0 && (above[b] = col[ix[b], iy[b], iz[b]-1])
    end
    # bottom cell = deepest wet cell in a column whose surface cell is wet (as upstream)
    ixBottom = Int[]; dzBottom = Float64[]
    for i in 1:nx, j in 1:ny
        col[i, j, 1] == 0 && continue
        k = findlast(>(0), view(col, i, j, :))
        push!(ixBottom, col[i, j, k]); push!(dzBottom, dznom[k])
    end

    # sinking
    vel = vcat(zeros(nN), vec(s.velocity))
    ixSink = findall(!=(0.0), vel)
    keep = zeros(nb, length(ixSink)); gain = zeros(nb, length(ixSink))
    isBottom = falses(nb); isBottom[ixBottom] .= true
    for (l, j) in enumerate(ixSink), b in 1:nb
        flx(k) = min(1, vel[j] * p.dtTransport / dznom[k])
        keep[b, l] = (p.BC_POMclosed && isBottom[b]) ? 1.0 : 1 - flx(iz[b])
        above[b] != b && (gain[b, l] = flx(iz[b]))    # upstream uses dznom(k) for both entries
    end

    # BCvalue = -1: bottom value of the initial condition
    BCvalue = U0[ixBottom, 1:nN]
    if p.BCvalue !== nothing
        for k in 1:nN
            p.BCvalue[k] != -1 && (BCvalue[:, k] .= p.BCvalue[k])
        end
    end

    return GlobalModel(p, nb, ix, iy, iz, nx, ny, nz, dznom, U0, L0, Tmat, Aexp, Aimp,
                       ixSink, above, keep, gain, ixBottom, dzBottom, BCvalue, dvBox)
end
load_global_model(paths::NamedTuple, s::NUMSetup; p::GlobalParams=GlobalParams()) =
    load_global_model(paths.dir, s; p, matrixdir=paths.matrixdir, n0file=paths.n0file,
                      si0file=paths.si0file, pardayfile=paths.pardayfile)

"""
    matrix_to_grid(g::GlobalModel, v) -> nx×ny×nz array (NaN on land); matrixToGrid.
"""
function matrix_to_grid(g::GlobalModel, v::AbstractVector)
    G = fill(NaN, g.nx, g.ny, g.nz)
    for b in 1:g.nb
        G[g.ixBox[b], g.iyBox[b], g.izBox[b]] = v[b]
    end
    return G
end

"""
    state_from_sim(g, s, sim; bNinit=false) -> nb×nGrid

simulateGlobal's restart: nutrients (and, unless bNinit, biomasses) from the last saved time of
a previous `sim` struct (Matlab layout: N, DOC, Si as nt×nx×ny×nz, B as nt×nx×ny×nz×nB);
the rest from the model's initial state.
"""
function state_from_sim(g::GlobalModel, s::NUMSetup, sim::AbstractDict; bNinit=false)
    U = Array(g.U0)
    get(G, b) = Float64(G[end, g.ixBox[b], g.iyBox[b], g.izBox[b]])
    for b in 1:g.nb
        U[b, 1] = get(sim["N"], b); U[b, 2] = get(sim["DOC"], b)
        s.nNutrients > 2 && (U[b, 3] = get(sim["Si"], b))
        if !bNinit
            B = sim["B"]
            for j in 1:s.nS
                U[b, s.nNutrients+j] = Float64(B[end, g.ixBox[b], g.iyBox[b], g.izBox[b], j])
            end
        end
    end
    return U
end

function sink!(U, g::GlobalModel)
    for (l, j) in enumerate(g.ixSink)
        u = view(U, :, j)
        u .= view(g.keep, :, l) .* u .+ view(g.gain, :, l) .* u[g.above]
    end
    return U
end

function bottom_bc!(U, g::GlobalModel)
    for k in axes(g.BCvalue, 2)
        ub = U[g.ixBottom, k]
        U[g.ixBottom, k] .= ub .+ g.p.dtTransport .* g.p.BCmixing[k] ./ g.dzBottom .* (view(g.BCvalue, :, k) .- ub)
    end
    return U
end

"Total N (g) of the state, as simulateGlobal's fNtot: (N + sum(B)/rhoCN)*dv/1000."
function total_N(U, g::GlobalModel, k::KSetup{D}) where {D}
    nN = D[2]
    return (dot(view(U, :, 1), g.dvBox) + dot(vec(sum(view(U, :, nN+1:size(U, 2)); dims=2)), g.dvBox) / k.rhoCN) / 1000
end

"""
    transport_step!(U, g, i, month_state, k, work; Fn=nothing) -> U

One step `i` (1-based) of the simulateGlobal loop, including the monthly switch.
`month_state` is a Ref{Tuple{Int,Int}} of (next month index 0..11, current TM month 1..12).
With `Fn` (nb×9), the biology is simulateEulerFunctions and Fn receives the functions; `Ubio`
receives the state after the biology.
"""
function transport_step!(U, g::GlobalModel, i::Int, month_state, k::KSetup, work; Fn=nothing, Ubio=nothing,
                         budget=nothing)
    p = g.p
    nPerYear = round(Int, 365 / p.dtTransport)
    month, cur = month_state[]
    if mod(i, nPerYear) in 1 .+ cumsum(vec(DAYS_BEFORE_MONTH)) ./ p.dtTransport
        cur = month + 1
        month = mod(month + 1, 12)
        month_state[] = (month, cur)
    end
    cur == 0 && error("no transport matrix loaded at step $i")
    L = view(g.L0, :, mod(i, nPerYear) + 1)
    T = view(g.Tmat, :, cur)
    # budget (optional, 3-vector, accumulated): N lost in the biology, lost by sinking, gained at the bottom BC
    budget === nothing || (N0 = total_N(U, g, k))
    if Fn === nothing
        simulate_euler!(U, L, T, p.dtTransport, p.dt, k)
    else
        simulate_euler_functions!(U, Fn, L, T, p.dtTransport, p.dt, k)
    end
    Ubio === nothing || copyto!(Ubio, U)          # state after the biology (for annual averages)
    budget === nothing || (N1 = total_N(U, g, k); budget[1] += N0 - N1)
    isempty(g.ixSink) || sink!(U, g)
    budget === nothing || (N2 = total_N(U, g, k); budget[2] += N1 - N2)
    bottom_bc!(U, g)
    budget === nothing || (budget[3] += total_N(U, g, k) - N2)
    if p.bTransport
        mul!(work, g.Aexp[cur], U)
        mul!(U, g.Aimp[cur], work)
    end
    return U
end
transport_step!(U, g::GlobalModel, i::Int, month_state, s::NUMSetup, work; kw...) =
    transport_step!(U, g, i, month_state, kernel_setup(s), work; kw...)

"""
    simulate_global(g, s; U=copy(g.U0), verbose=true, hostsetup) -> NamedTuple

Runs the simulateGlobal loop. Returns the state at each save time in double precision
(box order): `U` is nb×nGrid×nSave, `t` the save times (days), `Tmonth` the TM month used,
`L` and `T` (nb×nSave) the light and temperature of the saved step, `wall` the wall clock at
each save. With `g.p.bCalcAnnualAverages`, `annual` holds the gridded fields simulateGlobal
adds to `sim` (ProdNet, ProdHTL, mHTL, ProdGrossAnnual, ProdNetAnnual, ProdHTLAnnual,
BpicoAnnualMean, BnanoAnnualMean, BmicroAnnualMean, mHTLAnnualMean, BHTL). Also returned: the
N budget per save period (`Ntot`, `Nloss`, `NlossHTL`, `Nprod`, g). For GPU runs pass the device
`KSetup` as `s`; annual averages then need the host setup as `hostsetup=` (a `NUMSetup`).
"""
function simulate_global(g::GlobalModel, s::Union{NUMSetup,KSetup}; U=copy(g.U0), verbose=true,
                         hostsetup::Union{Nothing,NUMSetup}=s isa NUMSetup ? s : nothing)
    k = s isa NUMSetup ? kernel_setup(s) : s
    p = g.p
    simtime = round(Int, p.tEnd / p.dtTransport)
    nPerYear = round(Int, 365 / p.dtTransport)
    nSave = floor(Int, p.tEnd / p.tSave) + (mod(p.tEnd, p.tSave) > 1e-9 * p.tSave ? 1 : 0)
    Usave = Matrix{Float64}[]    # host copies; counted as saved (Matlab's mod() snaps round-off to 0)
    tSave = Float64[]; Tmonth = Int[]; Lsave = Vector{Float64}[]; Tsave = Vector{Float64}[]
    wall = Float64[]    # wall-clock seconds since the start of the loop, at each save
    work = similar(U)
    month_state = Ref((0, 0))
    ann = p.bCalcAnnualAverages
    if ann
        hostsetup === nothing && error("bCalcAnnualAverages needs the NUMSetup (hostsetup=...)")
        Fn = similar(U, size(U, 1), length(FUNCTION_NAMES))
        nb = size(U, 1)
        periodPN = zeros(nb, nSave); periodPH = zeros(nb, nSave); mHTLstep = zeros(nb, nSave)
        acc = zeros(nb, 6)          # ProdGross, ProdNet, ProdHTL, Bpico, Bnano, Bmicro (annual sums)
        BHTL = zeros(nb, hostsetup.nS)
        # getMortHTL: in a fresh Matlab session this returns zeros for "quadratic" HTL (setHTL
        # never sets the group mortHTL), making mHTLAnnualMean NaN; the intended coefficient pHTL is used
        mortHTL = hostsetup.pHTL; qexp = hostsetup.bQuadraticHTL ? 2 : 1
        per = simtime / nSave
        # simulateGlobal averages step values over ix = floor(per*(j-1)+1):(per*j) for save j
        periods = [floor(Int, per * (j - 1) + 1):floor(Int, per * j) for j in 1:nSave]
        period_of = zeros(Int, simtime)
        for (j, r) in enumerate(periods), i in r
            i <= simtime && (period_of[i] = j)
        end
        Ubio = similar(U)
    end
    # N budget per save period (simulateGlobal, Develop): total N (g) at the save, N lost in the
    # biology (NlossHTL) plus by sinking (Nloss), and N gained through the bottom BC (Nprod), in g
    Ntot = Float64[]; Nloss = Float64[]; NlossHTL = Float64[]; Nprod = Float64[]
    budget = zeros(3)
    t0 = time()
    for i in 1:simtime
        inlast = ann && i > simtime - nPerYear
        transport_step!(U, g, i, month_state, k, work; Fn=inlast ? Fn : nothing, Ubio=inlast ? Ubio : nothing,
                        budget)
        if inlast
            F = Array(Fn); Uh = Array(Ubio)       # BHTL uses the state after the biology
            j = period_of[i]
            if j > 0
                periodPN[:, j] .+= F[:, 2]; periodPH[:, j] .+= F[:, 3]
            end
            i <= nSave && (mHTLstep[:, i] .= F[:, 9])   # indexed by step, as upstream (docs/upstream_issues.md, 17)
            acc[:, 1] .+= F[:, 1]; acc[:, 2] .+= F[:, 2]; acc[:, 3] .+= F[:, 3]
            acc[:, 4] .+= F[:, 6]; acc[:, 5] .+= F[:, 7]; acc[:, 6] .+= F[:, 8]
            nN = hostsetup.nNutrients
            BHTL .+= Uh[:, nN+1:end] .^ qexp .* permutedims(mortHTL)
        end
        if floor(i * (p.dtTransport / p.tSave)) > floor((i - 1) * (p.dtTransport / p.tSave)) || i == simtime
            any(isnan, U) && error("NaNs after step $i")
            push!(tSave, i * p.dtTransport); push!(Tmonth, month_state[][2]); push!(wall, time() - t0)
            push!(Usave, Array(U))
            push!(Lsave, Array(view(g.L0, :, mod(i, nPerYear) + 1)))
            push!(Tsave, Array(view(g.Tmat, :, month_state[][2])))
            push!(Ntot, total_N(U, g, k)); push!(Nloss, budget[1] + budget[2]); push!(NlossHTL, budget[1])
            push!(Nprod, budget[3]); fill!(budget, 0.0)
            verbose && println("t = $(round(i * p.dtTransport; digits=1)) days  ($(round(time() - t0; digits=1)) s)")
            flush(stdout)
        end
    end
    res = (; t=tSave, U=stack(Usave), Tmonth, L=stack(Lsave), T=stack(Tsave), wall, Ntot, Nloss, NlossHTL, Nprod)
    ann || return res

    # gridded annual fields as simulateGlobal (integrate_over_depth divides by 365/dtTransport)
    iod(v) = begin
        G = Float32.(matrix_to_grid(g, v)) ./ Float32(nPerYear)
        G[isnan.(G)] .= 0
        dropdims(sum(G .* reshape(Float32.(g.dznom), 1, 1, :); dims=3); dims=3)
    end
    lens = length.(periods)
    ProdNet = stack([iod(periodPN[:, j] ./ lens[j]) for j in 1:nSave]; dims=1)
    ProdHTL = stack([iod(periodPH[:, j] ./ lens[j]) for j in 1:nSave]; dims=1)
    mHTLg = stack([Float32.(matrix_to_grid(g, mHTLstep[:, j]))[:, :, 1] for j in 1:nSave]; dims=1)
    BHTLd = stack([iod(BHTL[:, j]) for j in 1:hostsetup.nS]; dims=3)
    msum = sum(log(hostsetup.m[j]) .* BHTLd[:, :, j] for j in 1:hostsetup.nS)
    annual = (; ProdNet, ProdHTL, mHTL=mHTLg,
              ProdGrossAnnual=iod(acc[:, 1]), ProdNetAnnual=iod(acc[:, 2]), ProdHTLAnnual=iod(acc[:, 3]),
              BpicoAnnualMean=iod(acc[:, 4]), BnanoAnnualMean=iod(acc[:, 5]), BmicroAnnualMean=iod(acc[:, 6]),
              mHTLAnnualMean=exp.(msum ./ dropdims(sum(BHTLd; dims=3); dims=3)),
              BHTL=cat(zeros(Float32, g.nx, g.ny, hostsetup.nNutrients), BHTLd; dims=3))   # Matlab: one slice per state variable
    return merge(res, (; annual))
end
