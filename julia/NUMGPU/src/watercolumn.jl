# Water-column driver, mirroring NUMmodel/matlab/simulateWatercolumn.m and
# parametersWatercolumn.m (upstream Develop 5644bf5): one column of a transport-matrix configuration with vertical
# mixing from the column's block of Aimp, implicit sinking, bottom nutrient exchange, and the
# biology of every depth. Upstream conventions reproduced: calcGlobalWatercolumn picks the
# (x+1, y-1) neighbour of the nearest grid point; the bottom exchange has no 1/dz factor;
# months switch on 1+2*cumsum(mon) (two steps per day hard-coded); `dayFixed` freezes forcing.

Base.@kwdef struct WatercolumnParams
    dt::Float64 = 0.1
    dtTransport::Float64 = 0.5
    tEnd::Float64 = 365.0
    tSave::Float64 = 1.0
    BCmixing::Vector{Float64} = [1.0, 0.0, 1.0] ./ 365
    BCvalue::Union{Nothing,Vector{Float64}} = nothing     # nothing (= -1): bottom value of the initial state
    BC_POMclosed::Bool = false
    bUse_parday_light::Bool = true
    kw::Float64 = 0.1                  # parametersGlobal (0.06 in v1.0)
    EinConv::Float64 = 4.57
    PARfrac::Float64 = 0.4
    dayFixed::Float64 = 0.0
    TMconversion::Symbol = :develop   # see GlobalParams
end

"calcGlobalWatercolumn: grid indices of the column at (lat, lon), including upstream's index offset."
function calc_global_watercolumn(lat, lon, x, y, bathy)
    lon < 0 && (lon += 360)
    nx, ny = length(x), length(y)
    dist = [(x[i] - lon)^2 + (y[j] - lat)^2 for i in 1:nx, j in 1:ny]
    ix = findfirst(==(minimum(dist)), vec(dist))
    idxx = mod(ix, nx) + 1               # upstream: mod(ix, nx)+1 and floor(ix/nx) (the true
    idxy = ix ÷ nx                       # indices are mod(ix-1,nx)+1 and floor((ix-1)/nx)+1)
    idxz = findall(==(1), vec(bathy[idxx, idxy, :]))
    isempty(idxz) && error("The point $lat, $lon is on land.")
    return (x=idxx, y=idxy, z=idxz)
end

"""
    watercolumn_model(paths, s; lat, lon, p=WatercolumnParams()) -> NamedTuple

Extracts one column from a transport-matrix configuration (`tm_paths(name)`) as
simulateWatercolumn does.
"""
function watercolumn_model(paths::NamedTuple, s::NUMSetup; lat=0.0, lon=0.0, p=WatercolumnParams())
    grid = matread(joinpath(paths.dir, "grid.mat"))
    boxes = matread(joinpath(paths.dir, paths.matrixdir, "Data", "boxes.mat"))
    nb = Int(boxes["nb"])
    bix, biy, biz = vecint(boxes["ixBox"]), vecint(boxes["iyBox"]), vecint(boxes["izBox"])
    x, y, z = vec(grid["x"]), vec(grid["y"]), vec(grid["z"]); dznom = vec(grid["dznom"])
    idx = calc_global_watercolumn(lat, lon, x, y, grid["bathy"])
    col = Dict((bix[b], biy[b], biz[b]) => b for b in 1:nb)
    idxGrid = [col[(idx.x, idx.y, k)] for k in idx.z]
    nz = length(idxGrid)
    deltaT = grid["deltaT"]

    dvColumn = [Float64(grid["dv"][bix[b], biy[b], biz[b]]) for b in idxGrid]   # box volumes
    AimpM = Vector{Matrix{Float64}}(undef, 12)
    for m in 1:12
        TM = matread(joinpath(paths.dir, paths.matrixdir, "TMs", "matrix_nocorrection_" * lpad(m, 2, '0') * ".mat"))
        A = Matrix(convert_TM(TM["Aimp"][idxGrid, idxGrid], p.TMconversion, dvColumn))
        AimpM[m] = A^(p.dtTransport * SECONDS_PER_DAY / deltaT)
    end
    Tbc = matread(joinpath(paths.dir, "BiogeochemData", "Theta_bc.mat"))["Tbc"]
    Tmat = [Float64(Tbc[bix[b], biy[b], biz[b], m]) for b in idxGrid, m in 1:12]

    nL = round(Int, 365 / p.dtTransport)
    if p.bUse_parday_light
        # parday is single, so Matlab's result is single: each mixed single/double operation is
        # evaluated in double and rounded to single (verified against Matlab output)
        parday = matread(paths.pardayfile)["parday"]
        r32(x) = Float32(x)
        Lup = [r32(Float64(r32(1e6 * Float64(parday[idx.x, idx.y, 1, i]))) / SECONDS_PER_DAY) for i in 1:nL]
    else
        Y1 = vec(boxes["Ybox"])[idxGrid[1]]
        Lup = [p.EinConv * p.PARfrac * daily_insolation(Y1, i * p.dtTransport) for i in 1:nL]
    end
    L0 = [Lup[i] isa Float32 ? Float64(Float32(Float64(Lup[i]) * exp(-p.kw * z[k]))) : Lup[i] * exp(-p.kw * z[k])
          for k in 1:nz, i in 1:nL]

    # implicit sinking matrices (calcSinkingMatrix)
    nN = s.nNutrients
    vel = vcat(zeros(nN), s.velocity)
    ixSink = findall(!=(0.0), vel)
    Asink = Dict{Int,Matrix{Float64}}()
    for j in ixSink
        A = zeros(nz, nz)
        for i in 1:nz
            kk = vel[j] * p.dtTransport / dznom[i]
            A[i, i] = 1 + kk
            i != 1 && (A[i, i-1] = -kk)
        end
        p.BC_POMclosed && (A[end, end] = 1)
        Asink[j] = inv(A)
    end

    # initial conditions (all boxes, then the column)
    U = repeat(permutedims(initial_state(s)), nz)
    lin3 = LinearIndices(size(grid["bathy"]))
    if isfile(paths.n0file)
        N0 = matread(paths.n0file)["N"]
        U[:, 1] = [Float64(N0[bix[b], biy[b], biz[b]]) for b in idxGrid]
    end
    if nN > 2 && isfile(paths.si0file)
        Si0 = matread(paths.si0file)["Si"]
        U[:, 3] = [Float64(Si0[bix[b], biy[b], biz[b]]) for b in idxGrid]
    end
    BCvalue = U[end, :]
    if p.BCvalue !== nothing
        for i in eachindex(p.BCvalue)
            p.BCvalue[i] != -1 && (BCvalue[i] = p.BCvalue[i])
        end
    end
    return (; p, lat, lon, idx, idxGrid, nz, z=z[1:nz], dznom=dznom[1:nz], x, y, AimpM, Tmat, L0,
            Asink, U0=U, BCvalue)
end

"""
    simulate_watercolumn(w, s; U=copy(w.U0)) -> Dict (Matlab `sim` fields)

simulateWatercolumn's time loop; returns t, N, DOC, Si, B (nSave×nz×nB), L, T, Nloss,
NlossHTL, Ntot, Nprod, z, dznom, lat, lon, and the final state `u`. N budget (gN/m2/day per save
period): Nloss = HTL and POM export (only without a POM group) + N sinking out through the
bottom; Nprod = N entering through the bottom BC (both measured on the column's N inventory).
"""
function simulate_watercolumn(w, s::NUMSetup; U=copy(w.U0))
    p = w.p; k = kernel_setup(s); nN = s.nNutrients
    simtime = round(Int, p.tEnd / p.dtTransport)
    nPerYear = round(Int, 365 / p.dtTransport)
    thresholds = 1 .+ 2 .* cumsum(vec(DAYS_BEFORE_MONTH))
    nSave = floor(Int, p.tEnd / p.tSave) + (mod(p.tEnd, p.tSave) > 1e-9 * p.tSave ? 1 : 0)
    nz = w.nz
    sim = Dict{String,Any}("N" => zeros(nSave, nz), "DOC" => zeros(nSave, nz), "B" => zeros(nSave, nz, s.nS),
                           "L" => zeros(nSave, nz), "T" => zeros(nSave, nz),
                           "Nloss" => zeros(nSave), "NlossHTL" => zeros(nSave))
    nN > 2 && (sim["Si"] = zeros(nSave, nz))
    hasPOM = !isempty(s.ixPOM)
    month = 1; T = w.Tmat[:, 1]
    iSave = 0; tSave = Float64[]
    sim["Nprod"] = zeros(nSave)
    Ncol(U) = (sum(U[:, 1] .* w.dznom) + sum(vec(sum(U[:, nN+1:end]; dims=2)) .* w.dznom) / s.rhoCN) / 1000   # gN/m2
    NlossSinking = 0.0; Nbc = 0.0
    for i in 1:simtime
        iTime = p.dayFixed != 0 ? p.dayFixed / p.dtTransport : i
        mi = mod(iTime, nPerYear)
        if mi in thresholds || i == 1
            month = findfirst(>=(mi), thresholds)
            T = w.Tmat[:, month]
        end
        L = w.L0[:, Int(mi)+1]
        simulate_euler!(U, L, T, p.dtTransport, p.dt, k)
        any(isnan, U) && error("NaNs after step $i")
        U = w.AimpM[month] * U                          # vertical diffusion
        Nb = Ncol(U)
        for (j, A) in w.Asink
            U[:, j] = A * U[:, j]
        end
        NlossSinking += Nb - Ncol(U)
        Nb = Ncol(U)
        for kk in 1:nN
            U[end, kk] = U[end, kk] + p.BCmixing[kk] * p.dtTransport * (w.BCvalue[kk] - U[end, kk])
        end
        Nbc += Ncol(U) - Nb
        if floor(i * (p.dtTransport / p.tSave)) > floor((i - 1) * (p.dtTransport / p.tSave)) || i == simtime
            iSave += 1
            sim["N"][iSave, :] = U[:, 1]; sim["DOC"][iSave, :] = U[:, 2]
            nN > 2 && (sim["Si"][iSave, :] = U[:, 3])
            sim["B"][iSave, :, :] = U[:, nN+1:end]
            sim["L"][iSave, :] = L; sim["T"][iSave, :] = T
            if !hasPOM
                r = get_rates(U, L, T, 0.0, k)
                for j in 1:nz
                    ub = U[j, nN+1:end]
                    sim["NlossHTL"][iSave] += (1 - s.fracHTL_to_N) * sum(r.mortHTL[j, :] .* ub) / 1000 * w.dznom[j] / s.rhoCN
                    sim["Nloss"][iSave] += sum(r.jPOM[j, :] .* ub) / 1000 * w.dznom[j] / s.rhoCN
                end
            end
            sim["Nloss"][iSave] += NlossSinking / p.tSave
            sim["Nprod"][iSave] = Nbc / p.tSave
            NlossSinking = 0.0; Nbc = 0.0
            push!(tSave, i * p.dtTransport)
        end
    end
    sim["B"][sim["B"] .< 0] .= 0.0
    sim["DOC"][sim["DOC"] .< 0] .= 0.0
    sim["t"] = tSave
    sim["z"] = w.z; sim["dznom"] = w.dznom; sim["lat"] = w.lat; sim["lon"] = w.lon
    sim["Ntot"] = (vec(sum(sim["N"] .* permutedims(w.dznom); dims=2)) .+
                   vec(sum(dropdims(sum(sim["B"]; dims=3); dims=3) .* permutedims(w.dznom); dims=2)) ./ s.rhoCN) ./ 1000
    sim["u"] = U
    return sim
end
