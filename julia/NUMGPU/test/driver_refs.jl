# Compare the drivers with reference runs of the upstream Matlab code (scripts/make_driver_refs.m):
# chemostat (ode23s and Euler), insolation, water columns, and a 30-day global run with annual
# averages. Usage: julia --project=test test/driver_refs.jl <driver_refs.mat> [<global_annual_ref.mat>]
using Test, NUMGPU, MAT

R = matread(ARGS[1])
@testset "drivers vs Matlab" begin
rel(a, b) = maximum(abs.(a .- b) ./ (abs.(b) .+ 1e-9 * maximum(abs.(b)) .+ 1e-300))

@testset "insolation" begin
    Y = vec(R["insol_Y"]); days = vec(R["insol_days"])
    J = [daily_insolation(y, d) for y in Y, d in days]
    e = rel(J, R["insol"]); @info "daily_insolation: max rel err $e"
    @test e < 1e-12
end

@testset "chemostat" begin
    # ode23s: step control differs in details from Matlab's; compare where both converge
    s = setup_generalists_only(); sim = simulate_chemostat(s, parameters_chemostat(s); L=100.0, T=10.0)
    uM = R["chem1_u"]
    e = rel(sim["u"][end, :], uM[end, :])
    @info "chemostat generalists, constant: final state rel err $e; steps Julia $(length(sim["t"])) Matlab $(size(uM, 1))"
    @test e < 1e-2
    s = setup_num_model(); p = parameters_chemostat(s; seasonalAmplitude=0.5)
    sim = simulate_chemostat(s, p; L=100.0, T=10.0)
    uM = R["chem2_u"]
    # compare the annual-mean biomass per group (seasonal cycle; solver noise within a cycle)
    tJ = sim["t"]; tM = vec(R["chem2_t"])
    mean_trapz(t, Y) = vec(sum((Y[1:end-1, :] .+ Y[2:end, :]) ./ 2 .* diff(t); dims=1)) ./ (t[end] - t[1])
    mJ = mean_trapz(tJ, sim["u"]); mM = mean_trapz(tM, uM)
    gJ = [sum(mJ[s.nNutrients .+ gr.ix]) for gr in s.groups]; gM = [sum(mM[s.nNutrients .+ gr.ix]) for gr in s.groups]
    e = maximum(abs.(gJ .- gM) ./ (abs.(gM) .+ 1e-6))
    @info "chemostat NUMmodel, seasonal: annual-mean group biomass rel err $e (Julia $(round.(gJ; sigdigits=4)), Matlab $(round.(gM; sigdigits=4)))"
    @test e < 5e-2
    s = setup_generalists_diatoms(); p = parameters_chemostat(s)
    sim = simulate_chemostat(s, p; L=60.0, T=15.0, bUnicellularloss=false)
    e = rel(sim["u"][end, :], R["chem3_u"][end, :])
    @info "chemostat generalists+diatoms, N budget: final state rel err $e; Nprod Julia $(sim["Nprod"][end]) Matlab $(R["chem3_Nprod"][end]); Nloss Julia $(sim["Nloss"][end]) Matlab $(R["chem3_Nloss"][end])"
    @test e < 1e-2
    @test abs(sim["Nprod"][end] - R["chem3_Nprod"][end]) < 1e-2 * abs(R["chem3_Nprod"][end])
    @test abs(sim["Nloss"][end] - R["chem3_Nloss"][end]) < 1e-2 * abs(R["chem3_Nloss"][end]) + 1e-9
    # the budget closes in both codes: change in total N = Nprod - Nloss (up to solver error)
    Nt(u) = u[1] + sum(u[s.nNutrients+1:end]) / s.rhoCN
    for (u, np, nl) in ((sim["u"], sim["Nprod"][end], sim["Nloss"][end]),
                        (R["chem3_u"], R["chem3_Nprod"][end], R["chem3_Nloss"][end]))
        @test abs(Nt(u[end, :]) - Nt(u[1, :]) - (np - nl)) < 1e-4 * Nt(u[end, :])
    end

    # simulateChemostatEuler: Fortran-level, round-off agreement
    s = setup_num_model(); p = parameters_chemostat(s); p.tEnd = 50
    sim = simulate_chemostat_euler(s, p; L=100.0, T=10.0)
    e = rel(sim["u"], vec(R["cheE_u"])); @info "simulateChemostatEuler: state rel err $e"
    @test e < 1e-10
    @test rel(vec(sim["rates"].jTot), vec(R["cheE_jTot"])) < 1e-9
    @test abs(sim["Nbalance"] - R["cheE_Nb"]) < 1e-9 * (1 + abs(R["cheE_Nb"]))
    for f in (:ProdGross, :ProdNet, :ProdHTL)
        @test abs(getproperty(sim["functions"], f) - R["cheE_$f"]) < 1e-9 * abs(R["cheE_$f"])
    end
end

@testset "water column" begin
    for (tag, sf, lat, lon, tEnd, parday) in (("wc1", setup_num_model, 50.0, -30.0, 60.0, true),
                                             ("wc2", setup_generalists_only, -20.0, -120.0, 30.0, false))
        s = sf()
        w = watercolumn_model(tm_paths("MITgcm_2.8deg"), s; lat, lon,
                              p=WatercolumnParams(tEnd=tEnd, bUse_parday_light=parday))
        sim = simulate_watercolumn(w, s)
        # Near-bit agreement at first; afterwards nutrient-depleted surface layers amplify round-off
        # (as in the global runs), so later days are compared relative to the field's range.
        for f in ("N", "DOC", "B", "L", "T")
            J = sim[f]; M = R["$(tag)_$(f)"]
            e1 = maximum(abs.(J[1:2, :, :] .- M[1:2, :, :])) / maximum(abs.(M))
            eall = maximum(abs.(J .- M)) / maximum(abs.(M))
            @info "$tag $f: days 1-2 max err $e1 (of field max), all days $eall"
            @test e1 < 1e-12
            # divergence of round-off-sensitive layers: a Julia twin run started 1e-15 apart diverges
            # to the same size (wc2, Develop: N 7e-6, DOC 0.068, B 4e-3 of the field max in 30 days,
            # vs Matlab 1e-5, 0.115, 6e-3), so the spread is the model's, not the port's
            @test eall < 0.25
        end
        for f in (tag == "wc2" ? ("Nloss", "NlossHTL", "Nprod") : ("Nloss", "Nprod"))
            if tag == "wc1" || f == "Nprod"
                # flows measured as differences of the column inventory (gN/m2): errors relative
                # to that inventory
                J = sim[f]; M = vec(R["$(tag)_$(f)"]); Nc = maximum(sim["Ntot"])
                e1 = maximum(abs.(J[1:2] .- M[1:2])) / Nc
                eall = maximum(abs.(J .- M)) / maximum(abs.(M))
                @info "$tag $f: days 1-2 err $e1 (of column N), all days $eall (of max)"
                @test e1 < 1e-14
                @test eall < 0.1
            end
        end
        if tag == "wc2"
            for f in ("Nloss", "NlossHTL")
                J = sim[f]; M = vec(R["$(tag)_$(f)"])
                e1 = maximum(abs.(J[1:2] .- M[1:2]) ./ abs.(M[1:2])); eall = maximum(abs.(J .- M) ./ abs.(M))
                @info "$tag $f: days 1-2 rel err $e1, all days $eall"
                @test e1 < 1e-12
                @test eall < 1e-2
            end
        else
            @test rel(sim["Ntot"], vec(R["wc1_Ntot"])) < 1e-6
        end
    end
end

if length(ARGS) >= 2
    @testset "global, 30 days, annual averages" begin
        G = matread(ARGS[2])
        s = setup_num_model()
        g = load_global_model(tm_paths("MITgcm_2.8deg"), s; p=GlobalParams(tEnd=30.0, bCalcAnnualAverages=true))
        res = simulate_global(g, s; verbose=false)
        a = res.annual
        for f in ("ProdNet", "ProdHTL", "ProdGrossAnnual", "ProdNetAnnual", "ProdHTLAnnual",
                  "BpicoAnnualMean", "BnanoAnnualMean", "BmicroAnnualMean")
            J = Float64.(getfield(a, Symbol(f))); M = Float64.(G[f])
            J = reshape(J, size(M))
            ok = isfinite.(M)
            # single-precision output; surface boxes that amplify round-off make a few columns differ
            e = abs.(J[ok] .- M[ok]) ./ (abs.(M[ok]) .+ 1e-6 * maximum(abs.(M[ok])))
            med = sort(e)[end÷2]; q = sort(e)[ceil(Int, 0.99 * length(e))]
            @info "$f: median rel err $med, 99th percentile $q"
            # most columns are bit-identical; round-off-sensitive columns diverge within this 30-day
            # run. A Julia twin run perturbed by 1e-15 gives the same spread (715 vs 709 columns of
            # ProdNetAnnual differ by >1%), so the spread is the model's, not the port's.
            @test med < 1e-5
            @test count(>(1e-2), e) / length(e) < 0.25
        end
        # BHTL: all zero upstream (built from getMortHTL, which returns zeros for "quadratic" HTL in a
        # fresh session); the port uses the intended pHTL. Same layout (one slice per state variable).
        @test size(a.BHTL) == size(G["BHTL"])
        @test all(==(0), G["BHTL"])
        @test all(==(0), a.BHTL[:, :, 1:3]) && maximum(a.BHTL) > 0
        # mHTLAnnualMean: NaN (or 0) upstream wherever defined (getMortHTL returns zeros for quadratic HTL
        # in a fresh session); the port uses the intended pHTL and gives finite values
        M = Float64.(G["mHTLAnnualMean"])
        # N budget per save (g). Surface and shallow-shelf boxes (whose bottom cell is near the
        # surface) diverge from round-off within the 30 days, so the budget differs at the level a
        # Julia twin run started 1e-15 apart differs from the unperturbed run (Develop): Ntot 1.3e-9,
        # Nloss 7e-5, Nprod 0.8% (vs Matlab 2.2e-9, 1.3e-4, 0.9%). NlossHTL (N lost in the biology)
        # is zero with a POM group; both codes give round-off of the 5e17 g inventory (~1e3 g).
        tol = Dict("Ntot" => 1e-8, "Nloss" => 1e-3, "Nprod" => 5e-2)
        Nt = maximum(res.Ntot)
        for f in ("Ntot", "Nloss", "NlossHTL", "Nprod")
            J = getfield(res, Symbol(f)); M = G[f] isa Number ? [G[f]] : vec(G[f])
            e = maximum(abs.(J .- M)) / maximum(abs.(M))
            @info "global $f: max err $e of max ($(J[end]) vs $(M[end]))"
            if f == "NlossHTL"
                @test maximum(abs.(J)) < 1e-12 * Nt && maximum(abs.(M)) < 1e-12 * Nt
            else
                @test e < tol[f]
            end
        end
        @info "mHTLAnnualMean: Matlab finite entries $(count(isfinite, M)); Julia finite entries $(count(isfinite, a.mHTLAnnualMean))"
        @test count(x -> isfinite(x) && x > 0, M) == 0   # (exp(-Inf) = 0 can appear)
        @test count(isfinite, a.mHTLAnnualMean) > 0
    end
end
end
