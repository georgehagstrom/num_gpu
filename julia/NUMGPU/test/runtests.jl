# Compare the port with the Fortran reference library for every upstream setup: setup constants,
# derivatives, Euler steps and all diagnostics (getFunctions, getRates, getLost, getBalance).
using Test, Random, NUMGPU
include("tm_conversion.jl")        # self-contained
include("FortranOracle.jl")
using .FortranOracle
include("oracle_states.jl")

"""
Per-element error of `a` against the reference `b`, relative to |b| or to the turnover |u|×(1/day),
whichever is larger (floored at 1e-12 of the box's largest |b|). Derivatives that are near-exact
cancellations of large gains and losses can only be reproduced to round-off of those terms,
so a pure relative error is not meaningful for them.
"""
function relerr(a, b, U)
    scale = maximum(abs.(b); dims=2)
    return abs.(a .- b) ./ (abs.(b) .+ abs.(U) .+ 1e-12 .* scale .+ 1e-300)
end
fortran_rows(f, U, L, T) = permutedims(reduce(hcat, [f(U[k, :], L[k], T[k]) for k in axes(U, 1)]))

const TOL = 1e-12
# Derivatives: the copepod stage transfer gamma = (g - mort)/(1 - z^(1 - mort/g)) is ill-conditioned
# when mort ≈ g and amplifies last-bit differences in mortpred (worst case seen 6e-12, in about
# 1e-5 of the elements; all others below 1e-12).
const TOL_DERIV = 1e-11
# Euler steps: in boxes where the explicit step is near its stability limit a variable oscillates
# from substep to substep and last-bit differences grow (generalists_simple_copepod, Develop: DOC
# 0.056 -> 0.18 -> 0.088 -> 0.17 -> 0.16 -> 0.24, relative difference 3e-13 after 2 substeps,
# 5e-12 after 5, in one element of ~10^4; no gamma correction active).
const TOL_EULER = 1e-11
# theta (calcPhi) is ill-conditioned: differences of erf terms cancel, so last-bit differences
# between Julia's and glibc's erf/log/exp give up to ~1e-12 relative differences in theta.
# The dynamics are therefore tested with the library's theta, and the port's own theta separately.
const TOL_OWN_THETA = 1e-6

# (name, Julia setup, Fortran setup call)
const SETUPS = [
    (:generalists_simple_only, () -> setup_generalists_simple_only(), (s) -> FortranOracle.setup!(:generalists_simple_only, s.nNutrients + s.nS, s.nNutrients)),
    (:generalists_only, () -> setup_generalists_only(), (s) -> FortranOracle.setup!(:generalists_only, s.nNutrients + s.nS, s.nNutrients)),
    (:generalists_simple_pom, () -> setup_generalists_simple_pom(), (s) -> FortranOracle.setup!(:generalists_simple_pom, s.nNutrients + s.nS, s.nNutrients)),
    (:generalists_pom, () -> setup_generalists_pom(), (s) -> FortranOracle.setup!(:generalists_pom, s.nNutrients + s.nS, s.nNutrients)),
    (:diatoms_only, () -> setup_diatoms_only(), (s) -> FortranOracle.setup!(:diatoms_only, s.nNutrients + s.nS, s.nNutrients)),
    (:diatoms_simple_only, () -> setup_diatoms_simple_only(), (s) -> FortranOracle.setup!(:diatoms_simple_only, s.nNutrients + s.nS, s.nNutrients)),
    (:generalists_diatoms, () -> setup_generalists_diatoms(), (s) -> FortranOracle.setup!(:generalists_diatoms, s.nNutrients + s.nS, s.nNutrients)),
    (:generalists_diatoms_simple, () -> setup_generalists_diatoms_simple(), (s) -> FortranOracle.setup!(:generalists_diatoms_simple, s.nNutrients + s.nS, s.nNutrients)),
    (:generalists_simple_copepod, () -> setup_generalists_simple_copepod(), (s) -> FortranOracle.setup!(:generalists_simple_copepod, s.nNutrients + s.nS, s.nNutrients)),
    (:generic, () -> setup_generic([0.1, 10.0]), (s) -> FortranOracle.setup!(:generic, s.nNutrients + s.nS, s.nNutrients; mAdult=[0.1, 10.0])),
    (:num_model, () -> setup_num_model(), (s) -> FortranOracle.setup_num_model!()),
    (:num_model_simple, () -> setup_num_model_simple(), (s) -> FortranOracle.setup!(:num_model_simple, s.nNutrients + s.nS, s.nNutrients; mAdult=[0.1, 1, 10, 100, 1000], nCopepod=10, nPOM=10)),
    (:gen_diat_cope, () -> setup_gen_diat_cope(), (s) -> FortranOracle.setup!(:gen_diat_cope, s.nNutrients + s.nS, s.nNutrients; mAdult=[0.1, 1, 10, 100, 1000], nCopepod=10, nPOM=10)),
]

@testset "all setups" begin
@testset "setup $name" for (name, jsetup, fsetup) in SETUPS
    s = jsetup(); fsetup(s)
    nN = s.nNutrients
    m_f, _ = FortranOracle.get_mass()
    @test m_f[nN+1:end] == s.m
    th = FortranOracle.get_theta()[nN+1:end, nN+1:end]
    @test maximum(abs.(th .- s.theta)) <= 1e-12 * max(1.0, maximum(abs.(th)))
    k = kernel_setup(s); k.theta .= th        # dynamics with the library's theta
    ko = kernel_setup(s)                      # the port's own theta

    rng = MersenneTwister(hash(name))
    Ur, Lr, Tr = random_states(rng, s, 1000)
    Ut, Lt, Tt = trajectory_states(rng, s, FortranOracle.simulate_euler, 20, 10)
    dt = 0.1
    for (label, (U, L, T)) in ("random" => (Ur, Lr, Tr), "trajectories" => (Ut, Lt, Tt))
        D_f = fortran_rows((u, l, t) -> FortranOracle.calc_derivatives(u, l, t, dt), U, L, T)
        e = maximum(relerr(calc_derivatives(U, L, T, dt, k), D_f, U))
        eo = maximum(relerr(calc_derivatives(U, L, T, dt, ko), D_f, U))
        @info "$name, $label: derivatives max err $e (own theta $eo)"
        @test e < TOL_DERIV
        @test eo < TOL_OWN_THETA

        # diagnostics
        fnF = [FortranOracle.get_functions(U[b, :], L[b], T[b], dt) for b in axes(U, 1)]
        fnJ = get_functions(U, L, T, dt, k)
        # functions are sums of max(0, a - b) terms; errors relative to the box's gross production or
        # biomass turnover (sum |u| per day), which is the size of the terms that cancel
        turnover = vec(sum(abs.(U[:, nN+1:end]); dims=2))
        for f in (:ProdGross, :ProdNet, :ProdBact, :ProdHTL, :Bpico, :Bnano, :Bmicro)
            a = fnJ[f]; r = [x[f] for x in fnF]
            scale = max.(abs.(r), [x.ProdGross for x in fnF], turnover) .+ 1e-300
            ef = maximum(abs.(a .- r) ./ scale)
            ef < TOL || @info "  $name $label $f: max err $ef"
            @test ef < TOL
        end
        # eHTL = ProdHTL/ProdNet inherits the cancellation in ProdNet (tested above): check the definition
        @test isequal(fnJ.eHTL, fnJ.ProdHTL ./ fnJ.ProdNet)
        a = fnJ.mHTL; r = [x.mHTL for x in fnF]; ok = isfinite.(r)
        @test all(isequal.(isfinite.(a), ok))
        @test maximum(abs.(a[ok] .- r[ok]) ./ (abs.(r[ok]) .+ 1e-300); init=0.0) < TOL
        balF = [FortranOracle.get_balance(U[b, :], L[b], T[b], dt) for b in axes(U, 1)]
        balJ = get_balance(U, L, T, dt, k)
        # simple diatoms never initialize JCloss_photouptake in Fortran (getLost then reads undefined
        # memory, e.g. Clost = -6e196); the port uses 0, so C terms are not compared for them
        dias = any(gr.type == NUMGPU.diatom_simple for gr in s.groups)
        for f in (dias ? (:Nlost, :SiLost) : (:Clost, :Nlost, :SiLost))
            a = balJ[f]; r = [x[f] for x in balF]
            # losses sum respiration/exudation terms of several times the biomass per day
            ef = maximum(abs.(a .- r) ./ (abs.(r) .+ 1e-12 .* turnover .+ 1e-300))
            ef < 1e-11 || @info "  $name $label $f: max err $ef"
            @test ef < 1e-11
        end
        for (f, col) in ((:Cbalance, 2), (:Nbalance, 1), (:Sibalance, 3))
            (f == :Cbalance && dias) && continue
            (f == :Sibalance && nN < 3) && continue
            # residuals of large terms, divided by DOC/N/Si (non-finite where those are 0 in both
            # codes): compare the residuals themselves against the turnover
            a = balJ[f]; r = [x[f] for x in balF]; ok = isfinite.(r)
            @test all(.!isfinite.(a[.!ok]))
            @test maximum(abs.(a[ok] .- r[ok]) .* abs.(U[ok, col]) ./ (1e-9 .+ turnover[ok]); init=0.0) < 1e-12
        end
        rtF = [FortranOracle.get_rates(U[b, :], L[b], T[b], dt) for b in axes(U, 1)]
        rtJ = get_rates(U, L, T, dt, k)
        classscale = reduce((a, b) -> max.(a, b), [abs.(permutedims(reduce(hcat, [x[g] for x in rtF])))
                                                   for g in NUMGPU.RATE_NAMES])
        for f in NUMGPU.RATE_NAMES
            r = permutedims(reduce(hcat, [x[f] for x in rtF]))
            # many rates are cancellations (e.g. jNloss = max(0, jNreal + jFreal - jLossPassive - jTot)):
            # errors relative to the largest rate of that class in that box
            er = maximum(abs.(rtJ[f] .- r) ./ (abs.(r) .+ 1e-300 .+ classscale))
            er < TOL || @info "  $name $label rate $f: max err $er"
            @test er < TOL
        end
    end
    # Euler 0.5 d (5 substeps), as in simulateGlobal
    U, L, T = Ut, Lt, Tt
    U_f = fortran_rows((u, l, t) -> FortranOracle.simulate_euler(u, l, t, 0.5, 0.1), U, L, T)
    e = maximum(relerr(simulate_euler!(copy(U), L, T, 0.5, 0.1, k), U_f, U))
    @info "$name: Euler 0.5 d max err $e"
    @test e < TOL_EULER

    # simulateEulerFunctions (functions from the last substep's rates and the final state)
    nb = 20
    rows = [FortranOracle.simulate_euler_functions(U[b, :], L[b], T[b], 0.5, 0.1) for b in 1:nb]
    UJ = U[1:nb, :]; FnJ = zeros(nb, 9)
    simulate_euler_functions!(UJ, FnJ, L[1:nb], T[1:nb], 0.5, 0.1, k)
    @test maximum(relerr(UJ, permutedims(reduce(hcat, first.(rows))), U[1:nb, :])) < TOL
    for (j, f) in enumerate((:ProdGross, :ProdNet, :ProdHTL, :ProdBact))
        r = [x[2][f] for x in rows]
        sc = max.(abs.(r), [x[2].ProdGross for x in rows], vec(sum(abs.(UJ[:, nN+1:end]); dims=2))) .+ 1e-300
        @test maximum(abs.(FnJ[:, [1, 2, 3, 4][j]] .- r) ./ sc) < TOL
    end

    # chemostat Euler (Fortran simulateChemostatEuler), with the Julia setup's sinking
    FortranOracle.set_sinking!(vcat(zeros(nN), s.velocity))
    p = parameters_chemostat(s)
    for bl in (true, false)
        uF = FortranOracle.simulate_chemostat_euler(copy(p.u0), 100.0, 10.0, p.u0[1:nN], p.d, p.widthProductiveLayer, 5.0, 0.01, bl)
        uJ = NUMGPU.chemostat_euler!(copy(p.u0), s, 100.0, 10.0, p.u0[1:nN], p.d, p.widthProductiveLayer, 5.0, 0.01, bl)
        ec = maximum(abs.(uJ .- uF) ./ (abs.(uF) .+ 1e-12 * maximum(abs.(uF))))
        @info "$name: chemostat Euler 5 d (losses $bl) max err $ec"
        @test ec < 1e-10
    end
end
end

include("device_generic.jl")
