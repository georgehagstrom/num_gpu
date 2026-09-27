# Test states for comparing the port against the Fortran library.
using Random

"""
Random states covering the regimes the global model visits: log-uniform biomasses over
many decades, exact zeros, small negatives (transport can make u < 0), and nutrient
levels from depleted (triggers the gamma correction) to deep-water values.
Returns (U, L, T) with U nb×nGrid.
"""
function random_states(rng, s, nb)
    nN = s.nNutrients; nG = nN + s.nS
    U = zeros(nb, nG)
    for k in 1:nb
        U[k, 1] = 10.0^rand(rng, -6:0.01:2.5)          # N
        U[k, 2] = 10.0^rand(rng, -6:0.01:1.5)          # DOC
        nN > 2 && (U[k, 3] = 10.0^rand(rng, -6:0.01:2.5))   # Si
        U[k, nN+1:end] = 10.0 .^ (rand(rng, nG - nN) .* 10 .- 8)
        r = rand(rng, nG)
        U[k, r .< 0.05] .= 0.0
        U[k, r .> 0.97] .= -1e-3 .* rand(rng, count(r .> 0.97))
    end
    L = rand(rng, nb) .* 300
    L[rand(rng, nb) .< 0.2] .= 0.0                       # dark (deep boxes, night-free daily mean)
    T = rand(rng, nb) .* 32 .- 2
    return U, L, T
end

"States along Fortran Euler trajectories started from the initial condition (realistic)."
function trajectory_states(rng, s, oracle_simulate, ntraj, nsnap; dt=0.1, tStep=5.0)
    rows = Vector{Vector{Float64}}(); Ls = Float64[]; Ts = Float64[]
    for _ in 1:ntraj
        u = initial_state(s)
        u[1] = 10.0^rand(rng, 0:0.01:2.2)
        u[2] = rand(rng) * 5
        s.nNutrients > 2 && (u[3] = 10.0^rand(rng, 0:0.01:2.3))
        L = rand(rng) * 250; T = rand(rng) * 30 - 1
        for _ in 1:nsnap
            u = oracle_simulate(u, L, T, tStep, dt)
            push!(rows, copy(u)); push!(Ls, L); push!(Ts, T)
        end
    end
    return permutedims(reduce(hcat, rows)), Ls, Ts
end
