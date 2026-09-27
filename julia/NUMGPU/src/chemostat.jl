# Chemostat drivers, mirroring NUMmodel/matlab/parametersChemostat.m, simulateChemostat.m (ode23s,
# with the cumulative N budget) and simulateChemostatEuler.m (Fortran simulateChemostatEuler, then
# getFunctions of the final state), upstream Develop 5644bf5. Upstream convention reproduced:
# parametersChemostat ignores `constantValues` (d = 0.5, L = 100 are hard-coded) and sets u0(N) = 1.

Base.@kwdef mutable struct ChemostatParams
    d::Union{Float64,Vector{Float64}} = 0.5    # mixing rate (1/day), or 365 daily values
    L::Union{Float64,Vector{Float64}} = 100.0  # light, or 365 daily values (seasonal / lat_lon)
    widthProductiveLayer::Float64 = 20.0
    tEnd::Float64 = 365.0
    tSave::Float64 = 1.0
    uDeep::Vector{Float64} = Float64[]         # nutrients below the chemostat layer
    u0::Vector{Float64} = Float64[]
    kw::Float64 = 0.05
    EinConv::Float64 = 4.57
    PARfrac::Float64 = 0.4
    seasonal::Bool = false
end

"""
    parameters_chemostat(s; seasonalAmplitude=0, lat_lon=nothing, tmpaths=nothing) -> ChemostatParams

parametersChemostat: constant (d = 0.5/day, L = 100), idealized seasonal (seasonalAmplitude in
[0,1]) or from a water column of a transport matrix (lat_lon = (lat, lon); upstream uses
MITgcm_ECCO: pass `tmpaths=tm_paths("MITgcm_ECCO")`).
"""
function parameters_chemostat(s::NUMSetup; seasonalAmplitude=0.0, lat_lon=nothing, tmpaths=nothing)
    nN = s.nNutrients
    p = ChemostatParams(uDeep=s.u0[1:nN], u0=copy(s.u0))
    p.u0[1] = 1.0                        # "Nitrogen concentration"
    if lat_lon === nothing
        if seasonalAmplitude != 0
            lam = seasonalAmplitude
            d = [(t < 125 || t > 300) ? (1 - lam) * 0.5 + lam * 0.6 : (1 - lam) * 0.5 + lam * 0.02 for t in 1:365]
            L = [(t <= 60 || t >= 280) ? (1 - lam) * 100 + lam * 5 : (1 - lam) * 100 + lam * 35 for t in 1:365]
            p.d = d; p.L = L; p.seasonal = true
        end
    else
        lat, lon = lat_lon
        tmpaths === nothing && (tmpaths = tm_paths("MITgcm_ECCO"))
        w = watercolumn_model(tmpaths, s; lat, lon, p=WatercolumnParams(bUse_parday_light=false, kw=p.kw))
        depthProductiveLayer = 50; epsilon = 15
        idx_chemo = findall(zz -> depthProductiveLayer - epsilon < zz < depthProductiveLayer + epsilon, w.z)
        mon = [31 28 31 30 31 30 31 31 30 31 30 31]
        d = zeros(365); L = zeros(365)
        for m in 1:12
            A = w.AimpM[m]; j = sum(mon[1:m-1])
            for kk in 1:mon[m]
                d[kk+j] = sum(A[idx_chemo, idx_chemo[end]+1:end])
                L[kk+j] = p.EinConv * p.PARfrac * daily_insolation(lat, kk + j) * exp(-p.kw * depthProductiveLayer)
            end
        end
        p.d = d; p.L = L; p.seasonal = true
    end
    return p
end

daily(v::Real, t) = v
function daily(v::AbstractVector, t)
    ti = floor(Int, mod(t, 365)) + 1
    return v[min(ti, 365)]
end

"""
    ode23s(f, tspan, y0; rtol=1e-3, atol=1e-6) -> (t, Y)

Matlab ode23s (Shampine & Reichelt 1997): modified Rosenbrock (2,3) pair with a finite-difference
Jacobian at every step; returns every accepted step, like Matlab's default output.
"""
function ode23s(f, tspan, y0; rtol=1e-3, atol=1e-6)
    t0, tfinal = tspan
    d = 1 / (2 + sqrt(2)); e32 = 6 + sqrt(2)
    pow = 1 / 3; threshold = atol / rtol
    y = copy(y0); t = t0; n = length(y)
    hmax = 0.1 * abs(tfinal - t0)
    F0 = f(t, y)
    hmin = 16 * eps(Float64(t))
    rh = maximum(abs.(F0) ./ max.(abs.(y), threshold)) / (0.8 * rtol^pow)
    absh = min(hmax, abs(tfinal - t0))
    absh * rh > 1 && (absh = 1 / rh)
    absh = max(absh, hmin)
    ts = [t]; ys = [copy(y)]
    I = Matrix{Float64}(LinearAlgebra.I, n, n)
    J = zeros(n, n)
    while t < tfinal
        hmin = 16 * eps(Float64(t))
        absh = min(hmax, max(hmin, absh))
        h = absh
        if 1.1 * absh >= abs(tfinal - t)
            h = tfinal - t; absh = abs(h)
        end
        # Jacobian and dF/dt by finite differences (numjac-like increments)
        for j in 1:n
            del = sqrt(eps()) * max(abs(y[j]), threshold)
            yj = copy(y); yj[j] += del
            J[:, j] = (f(t, yj) .- F0) ./ del
        end
        delt = sqrt(eps()) * max(abs(t), abs(t + h))
        dfdt = (f(t + delt, y) .- F0) ./ delt
        nofailed = true
        local ynew, F2, tnew
        while true
            W = I .- (h * d) .* J
            Wf = lu(W)
            k1 = Wf \ (F0 .+ (h * d) .* dfdt)
            F1 = f(t + 0.5h, y .+ 0.5h .* k1)
            k2 = Wf \ (F1 .- k1) .+ k1
            tnew = t + h
            ynew = y .+ h .* k2
            F2 = f(tnew, ynew)
            k3 = Wf \ (F2 .- e32 .* (k2 .- F1) .- 2 .* (k1 .- F0) .+ (h * d) .* dfdt)
            err = absh / 6 * maximum(abs.(k1 .- 2 .* k2 .+ k3) ./ max.(max.(abs.(y), abs.(ynew)), threshold))
            if err > rtol
                absh <= hmin && error("ode23s: step size below minimum at t = $t")
                absh = nofailed ? max(hmin, absh * max(0.1, 0.8 * (rtol / err)^pow)) : max(hmin, 0.5 * absh)
                h = absh; nofailed = false
                continue
            end
            if nofailed
                temp = 1.25 * (err / rtol)^pow
                absh = temp > 0.2 ? absh / temp : 5.0 * absh
            end
            break
        end
        t = tnew; y = ynew; F0 = F2
        push!(ts, t); push!(ys, copy(y))
    end
    return ts, permutedims(reduce(hcat, ys))
end

function chemostat_mixing_indices(s::NUMSetup, bUnicellularloss)
    nN = s.nNutrients
    if bUnicellularloss       # nutrients and all classes up to the last unicellular group
        g = findlast(gr -> Int(gr.type) < 10, s.groups)
        return 1:(nN + last(s.groups[g].ix))
    else
        return 1:nN
    end
end

"""
    simulate_chemostat(s, p=parameters_chemostat(s); L=100, T=10, bUnicellularloss=true) -> Dict

simulateChemostat: integrates the chemostat with ode23s. The state is augmented with the
cumulative N budget of the layer (mugN/l from t = 0): `Nprod` (mixed in from the deep), `Nloss`
(mixing of unicellulars, sinking of POM, and HTL/POM export without a POM group) and `NlossHTL`
(the export part, from getLost).
"""
function simulate_chemostat(s::NUMSetup, p::ChemostatParams=parameters_chemostat(s); L=100.0, T=10.0,
                            bUnicellularloss=true)
    k = kernel_setup(s); nN = s.nNutrients; nG = nN + s.nS
    p.seasonal && (L = p.L)
    ix = chemostat_mixing_indices(s, bUnicellularloss)
    ixUniB = filter(>(nN), collect(ix))            # the biomass part of what is mixed
    uDeep = vcat(p.uDeep, zeros(s.nS))
    ixPOM = nN .+ s.ixPOM
    vel = vcat(zeros(nN), s.velocity)
    function fderiv(t, y)
        u = y[1:nG]
        Lnow = daily(L, t); dnow = daily(p.d, t)
        dudt = vec(calc_derivatives(permutedims(u), [Lnow], [Float64(T)], 0.0, k))
        dudt[ix] .+= dnow .* (uDeep[ix] .- u[ix])
        sinkPOM = vel[ixPOM] .* u[ixPOM] ./ p.widthProductiveLayer
        dudt[ixPOM] .-= sinkPOM
        Nlost = get_balance(permutedims(u), [Lnow], [Float64(T)], 0.0, k).Nlost[1]
        Nprod = dnow * (uDeep[1] - u[1])
        Nloss = dnow * sum(u[ixUniB]) / s.rhoCN + sum(sinkPOM) / s.rhoCN + Nlost
        return vcat(dudt, Nprod, Nloss, Nlost)
    end
    t, Y = ode23s(fderiv, (0.0, p.tEnd), vcat(p.u0, 0.0, 0.0, 0.0))
    sim = Dict{String,Any}()
    sim["Nprod"] = Y[:, nG+1]; sim["Nloss"] = Y[:, nG+2]; sim["NlossHTL"] = Y[:, nG+3]
    Y = Y[:, 1:nG]
    sim["u"] = Y; sim["t"] = t
    sim["N"] = Y[:, 1]; sim["DOC"] = Y[:, 2]; nN > 2 && (sim["Si"] = Y[:, 3])
    sim["B"] = Y[:, nN+1:end]
    Lm = L isa AbstractVector ? sum(L) / length(L) : Float64(L)
    sim["L"] = L; sim["T"] = T
    sim["rates"] = get_rates(Y[end:end, :], [Lm], [Float64(T)], 0.0, k)
    sim["Bgroup"] = reduce(hcat, [vec(sum(Y[:, nN .+ gr.ix]; dims=2)) for gr in s.groups])
    bal = get_balance(Y[end:end, :], [Lm], [Float64(T)], 0.0, k)
    sim["Cbalance"], sim["Nbalance"], sim["Sibalance"] = bal.Cbalance[1], bal.Nbalance[1], bal.Sibalance[1]
    return sim
end

"""
    chemostat_euler!(u, s, L, T, Ndeep, diff, widthProductiveLayer, tEnd, dt, bLosses) -> u

Fortran simulateChemostatEuler for one box: nutrients mix with the deep layer; with `bLosses`
the unicellular groups are mixed out (copepods and POM are not); every group sinks at
velocity/widthProductiveLayer.
"""
function chemostat_euler!(u::Vector{Float64}, s::NUMSetup, L, T, Ndeep, diff, widthProductiveLayer,
                          tEnd, dt, bLosses)
    k = kernel_setup(s); nN = s.nNutrients; nG = nN + s.nS
    du = MVector{nG,Float64}(undef); uv = MVector{nG,Float64}(u)
    for _ in 1:floor(Int, tEnd / dt)
        box_derivs!(du, uv, L, T, dt, k)
        du[1] += diff * (Ndeep[1] - uv[1])
        du[2] += diff * (Ndeep[2] - uv[2])
        nN > 2 && (du[3] += diff * (Ndeep[3] - uv[3]))
        if bLosses
            for gr in s.groups
                is_unicellular(gr.type) || continue
                for i in gr.ix
                    du[nN+i] -= diff * uv[nN+i]
                end
            end
        end
        for i in 1:s.nS
            du[nN+i] -= s.velocity[i] / widthProductiveLayer * uv[nN+i]
        end
        for j in 1:nG
            uv[j] = uv[j] + du[j] * dt
        end
    end
    u .= uv
    return u
end

"""
    simulate_chemostat_euler(s, p=parameters_chemostat(s); L=100, T=10, bUnicellularloss=true) -> Dict

simulateChemostatEuler: Fortran chemostat Euler (dt = 0.01) for tEnd, mixing with p.uDeep, then
the functions (getFunctions), rates and balances of the final state.
"""
function simulate_chemostat_euler(s::NUMSetup, p::ChemostatParams=parameters_chemostat(s); L=100.0, T=10.0,
                                  bUnicellularloss=true)
    k = kernel_setup(s); nN = s.nNutrients
    u = chemostat_euler!(copy(p.u0), s, Float64(L), Float64(T), p.uDeep[1:nN], p.d, p.widthProductiveLayer,
                         p.tEnd, 0.01, bUnicellularloss)
    U = permutedims(u)
    fn = get_functions(U, [Float64(L)], [Float64(T)], 0.0, k)
    sim = Dict{String,Any}("t" => p.tEnd, "u" => u, "N" => u[1], "DOC" => u[2], "B" => u[nN+1:end],
                           "L" => L, "T" => T,
                           "functions" => NamedTuple{FUNCTION_NAMES}(Tuple(getproperty(fn, f)[1] for f in FUNCTION_NAMES)))
    nN > 2 && (sim["Si"] = u[3])
    sim["rates"] = get_rates(U, [Float64(L)], [Float64(T)], 0.0, k)
    sim["Bgroup"] = [sum(u[nN .+ gr.ix]) for gr in s.groups]
    bal = get_balance(U, [Float64(L)], [Float64(T)], 0.0, k)
    sim["Cbalance"], sim["Nbalance"], sim["Sibalance"] = bal.Cbalance[1], bal.Nbalance[1], bal.Sibalance[1]
    return sim
end
