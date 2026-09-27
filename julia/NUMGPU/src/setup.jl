# Setup: size grids, per-size-class constants, preference matrix theta, HTL selectivity and
# POM routing, mirroring NUMmodel.f90 (setupXXX, parametersInit/AddGroup/Finalize, setHTL,
# setMortHTL, setSinking), the init routines of the group modules, and the Matlab setup
# wrappers (initial conditions, POM sinking and HTL overrides).
#
# Fortran single-precision literals (e.g. `0.4*1d-6`, `0.004/log(..)`) are reproduced with
# Float32 constants so the setup matches the compiled library to round-off.

const EPS = 1e-200          # globals.f90: eps
const TREF = 10.0           # globals.f90: Tref

@enum GroupType generalist_simple = 1 diatom = 3 diatom_simple = 4 generalist = 5 copepod_active = 10 copepod_passive = 11 pom = 100

is_unicellular(t::GroupType) = t in (generalist_simple, generalist, diatom, diatom_simple)
is_copepod(t::GroupType) = t in (copepod_active, copepod_passive)
is_diatom(t::GroupType) = t in (diatom, diatom_simple)

struct Group
    type::GroupType
    ix::UnitRange{Int}      # classes in the biomass block (1-based, excluding nutrients)
end

"""
All setup constants (host arrays). Per-size-class vectors have length nS (biomass classes).
Scalar parameters of each group type are module globals in Fortran, so one value per type
(for copepods: the last copepod group initialized). Mutable so that set_htl!, set_mort_htl!
and set_sinking! can change it like the Fortran setters.
"""
mutable struct NUMSetup
    name::String
    nNutrients::Int                 # 2 (N, DOC) or 3 (N, DOC, Si)
    nS::Int
    groups::Vector{Group}
    rhoCN::Float64
    fracHTL_to_N::Float64
    fracHTL_to_POM::Float64
    m::Vector{Float64}; mLower::Vector{Float64}; mDelta::Vector{Float64}; z::Vector{Float64}
    AF::Vector{Float64}; JFmax::Vector{Float64}; epsilonF::Vector{Float64}
    mort2constant::Vector{Float64}
    pHTL::Vector{Float64}
    velocity::Vector{Float64}
    AN::Vector{Float64}; AL::Vector{Float64}; Jmax::Vector{Float64}; Jresp::Vector{Float64}
    JlossPassive::Vector{Float64}
    JrespFactor::Vector{Float64}
    mPOM::Vector{Float64}
    theta::Matrix{Float64}          # nS×nS, theta[pred, prey]
    thetaPOM::Matrix{Float64}       # nS×nPOM routing of jPOM into POM classes (0/1)
    isPOM::Vector{Float64}
    ixPOM::UnitRange{Int}           # empty if there is no POM group
    bQuadraticHTL::Bool
    gen::NamedTuple{(:epsilonL, :bL, :bN, :bDOC, :bF, :bg, :remin2, :reminF),NTuple{8,Float64}}
    gens::NamedTuple{(:epsilonL, :remin2, :reminF),NTuple{3,Float64}}
    dia::NamedTuple{(:epsilonL, :bL, :bN, :bDOC, :bSi, :bg, :remin2, :rhoCSi),NTuple{8,Float64}}
    dias::NamedTuple{(:epsilonL, :bN, :bSi, :remin2, :rhoCSi),NTuple{5,Float64}}
    cop::NamedTuple{(:epsilonR, :kBasal, :kSDA),NTuple{3,Float64}}
    pomremin::Float64
    u0::Vector{Float64}             # initial condition (Matlab p.u0)
end

groupix(s::NUMSetup, t::GroupType) = [gr.ix for gr in s.groups if gr.type == t]

# spectrum.f90: calcGrid
function calc_grid(n, mMin, mMax)
    deltax = (log(mMax) - log(mMin)) / n
    m = zeros(n); mLower = zeros(n); mDelta = zeros(n); z = zeros(n)
    for i in 1:n
        x = log(mMin) + (i - 0.5) * deltax
        m[i] = exp(x)
        mLower[i] = exp(x - 0.5 * deltax)
        mDelta[i] = exp(x + 0.5 * deltax) - mLower[i]
        z[i] = mLower[i] / (mLower[i] + mDelta[i])
    end
    # (for n = 1, e.g. a single POM class, Fortran reads out of bounds; the value is unused)
    mort2constant = n > 1 ? Float64(0.004f0) / log((mLower[2] + mDelta[1]) / mLower[1]) : 0.0
    return (; m, mLower, mDelta, z, mort2constant)
end

# NUMmodel.f90: calcPhi
function calc_phi(z, beta, sigma, Delta)
    beta == 0.0 && return 0.0
    s = 2 * sigma * sigma
    res = max(0.0,
        (sqrt(Delta) * (((exp(-log((beta * Delta) / z)^2 / s) - 2 / exp(log(z / beta)^2 / s) +
                          exp(-log((Delta * z) / beta)^2 / s)) * s) / 2.0 -
                        (sqrt(pi) * sqrt(s) * (erf((-log(beta * Delta) + log(z)) / sqrt(s)) * log((beta * Delta) / z) +
                                               2 * erf(log(z / beta) / sqrt(s)) * log(z / beta) +
                                               erf((log(beta) - log(Delta * z)) / sqrt(s)) * log((Delta * z) / beta))) / 2.0)) /
        ((-1 + Delta) * log(Delta)))
    return isnan(res) ? 0.0 : res
end

# ---------------------------------------------------------------- group initializers
# Each returns the per-class constants of one group and its type's scalar parameters.

# class-constant fields filled by the group initializers (defaults as spectrum.f90 initSpectrum)
const CLASS_FIELDS = (:m, :mLower, :mDelta, :z, :AF, :JFmax, :epsilonF, :mort2constant, :velocity,
                      :AN, :AL, :Jmax, :Jresp, :JlossPassive, :JrespFactor, :mPOM,
                      :palat, :beta, :sigma, :diatPref)

function class_block(n, gr)
    c = Dict{Symbol,Vector{Float64}}(f => zeros(n) for f in CLASS_FIELDS)
    c[:m] .= gr.m; c[:mLower] .= gr.mLower; c[:mDelta] .= gr.mDelta; c[:z] .= gr.z
    c[:mort2constant] .= gr.mort2constant
    c[:epsilonF] .= 1.0; c[:palat] .= 1.0; c[:diatPref] .= 1.0
    return c
end

# generalists.f90: initGeneralists
function init_generalists(P, n)
    g(k) = getparam(P, "generalists", k)
    gr = calc_grid(n, g("mMinGeneralist"), g("mMaxGeneralist"))
    c = class_block(n, gr)
    rho = g("rho"); delta = g("delta")
    r = (3.0 / (4.0 * pi) .* gr.m ./ rho) .^ (1 / 3)
    nu = min.(1.0, 3 * delta ./ r)
    c[:AN] .= g("alphaN") .* r .^ (-2.0) ./ (1.0 .+ (r ./ g("rNstar")) .^ (-2.0)) .* gr.m
    c[:AL] .= g("alphaL") ./ r .* (1 .- exp.(-r ./ g("rLstar"))) .* gr.m .* (1.0 .- nu)
    c[:AL][gr.m .> g("mUpperAlphaL")] .= 0.0
    c[:AF] .= g("alphaF") .* gr.m
    c[:JFmax] .= g("cF") ./ r .* gr.m
    c[:JlossPassive] .= g("cLeakage") ./ r .* gr.m
    c[:Jmax] .= g("alphaJ") .* gr.m .* (1.0 .- nu)
    c[:Jresp] .= g("cR") * g("alphaJ") .* gr.m
    c[:epsilonF] .= g("epsilonF"); c[:beta] .= g("beta"); c[:sigma] .= g("sigma")
    c[:mPOM] .= gr.m
    par = (; epsilonL=g("epsilonL"), bL=g("bL"), bN=g("bN"), bDOC=g("bDOC"),
           bF=g("bF"), bg=g("bg"), remin2=g("remin2"), reminF=g("reminF"))
    return c, par
end

# generalists_simple.f90: initGeneralistsSimple (rho hard-coded: 0.4*1d6*1d-12, single 0.4)
function init_generalists_simple(P, n)
    g(k) = getparam(P, "generalists_simple", k)
    gr = calc_grid(n, g("mMinGeneralist"), g("mMaxGeneralist"))
    c = class_block(n, gr)
    rho = Float64(0.4f0) * 1e6 * 1e-12
    r = (3.0 / (4.0 * pi) .* gr.m ./ rho) .^ (1 / 3)
    nu = min.(1.0, 3 * g("delta") ./ r)
    c[:AN] .= g("alphaN") .* r .^ (-2.0) ./ (1.0 .+ (r ./ g("rNstar")) .^ (-2.0)) .* gr.m
    c[:AL] .= g("alphaL") ./ r .* (1 .- exp.(-r ./ g("rLstar"))) .* gr.m .* (1.0 .- nu)
    c[:AF] .= g("alphaF") .* gr.m
    c[:JFmax] .= g("cF") ./ r .* gr.m
    c[:JlossPassive] .= g("cLeakage") ./ r .* gr.m
    c[:Jmax] .= g("alphaJ") .* gr.m .* (1.0 .- nu)
    c[:Jresp] .= g("cR") * g("alphaJ") .* gr.m
    c[:epsilonF] .= g("epsilonF"); c[:beta] .= g("beta"); c[:sigma] .= g("sigma")
    c[:mPOM] .= gr.m
    par = (; epsilonL=g("epsilonL"), remin2=g("remin2"), reminF=g("reminF"))
    return c, par
end

# diatoms.f90: initDiatoms (the mMax argument of parametersAddGroup is not used)
function init_diatoms(P, n)
    g(k) = getparam(P, "diatoms", k)
    gr = calc_grid(n, g("mMinDiatom"), g("mMaxDiatom"))
    c = class_block(n, gr)
    rhoD = Float64(0.4f0) * 1e-6       # Fortran: 0.4*1d-6 (single-precision 0.4)
    v = g("v"); delta = g("delta")
    r = (0.75 / pi .* gr.m ./ rhoD ./ (1 - v)) .^ (1 / 3)
    nu = 6^(2 / 3) * pi^(1 / 3) * delta .* (gr.m ./ rhoD) .^ (-1 / 3) .* (1.0 + v^(2 / 3)) ./ (1.0 - v)^(2 / 3)
    nu = min.(1.0, nu)
    c[:AN] .= g("alphaN") .* r .^ (-2.0) ./ (1.0 .+ (r ./ g("rNstar")) .^ (-2.0)) .* gr.m ./ (1 - v)
    c[:AL] .= g("alphaL") ./ r .* (1 .- exp.(-r ./ g("rLstar"))) .* gr.m .* (1.0 .- nu) ./ (1 - v)
    c[:JlossPassive] .= g("cLeakage") ./ r .* gr.m
    c[:Jmax] .= g("alphaJ") .* gr.m .* (1.0 .- nu)
    c[:Jresp] .= g("cR") * g("alphaJ") .* gr.m
    c[:palat] .= g("palatability")
    c[:mPOM] .= gr.m
    par = (; epsilonL=g("epsilonL"), bL=g("bL"), bN=g("bN"), bDOC=g("bDOC"),
           bSi=g("bSi"), bg=g("bg"), remin2=g("remin2"), rhoCSi=g("rhoCSi"))
    return c, par
end

# diatoms_simple.f90: initDiatoms_simple (grid from input mMin to the mMax argument; no min(1,nu))
function init_diatoms_simple(P, n, mMax)
    g(k) = getparam(P, "diatoms_simple", k)
    gr = calc_grid(n, g("mMin"), mMax)
    c = class_block(n, gr)
    rho = Float64(0.4f0) * 1e6 * 1e-12
    v = g("v"); delta = g("delta")
    r = (0.75 / pi .* gr.m ./ rho ./ (1 - v)) .^ (1 / 3)
    nu = 6^(2 / 3) * pi^(1 / 3) * delta .* (gr.m ./ rho) .^ (-1 / 3) .* (v^(2 / 3) + (1.0 + v)^(2 / 3))
    c[:AN] .= g("alphaN") .* r .^ (-2.0) ./ (1.0 .+ (r ./ g("rNstar")) .^ (-2.0)) .* gr.m ./ (1 - v)
    c[:AL] .= g("alphaL") ./ r .* (1 .- exp.(-r ./ g("rLstar"))) .* gr.m .* (1.0 .- nu) ./ (1 - v)
    c[:JlossPassive] .= g("cLeakage") ./ r .* gr.m
    c[:Jmax] .= g("alphaJ") .* gr.m .* (1.0 .- nu)
    c[:Jresp] .= g("cR") * g("alphaJ") .* gr.m
    c[:palat] .= g("palatability")
    c[:mPOM] .= gr.m
    par = (; epsilonL=g("epsilonL"), bN=g("bN"), bSi=g("bSi"), remin2=g("remin2"), rhoCSi=g("rhoCSi"))
    return c, par
end

# copepods.f90: initCopepod (mAdult = the mMax argument of parametersAddGroup)
function init_copepod(P, section, n, mAdult)
    g(k) = getparam(P, section, k)
    mOff = mAdult / g("AdultOffspring")
    lnDelta = (log(mAdult) - log(mOff)) / (n - 0.5)
    gr = calc_grid(n, mOff, exp(log(mAdult) + 0.5 * lnDelta))
    c = class_block(n, gr)
    c[:mort2constant] .= 0.0            # no quadratic mortality
    c[:AF] .= g("alphaF") .* gr.m .^ g("q")
    c[:JFmax] .= g("h") .* gr.m .^ g("hExponent")
    c[:epsilonF] .= g("epsilonF")
    c[:JrespFactor] .= c[:epsilonF] .* c[:JFmax]
    c[:palat] .= g("vulnerability")
    c[:beta] .= g("beta"); c[:sigma] .= g("sigma")
    c[:diatPref] .= g("DiatomsPreference")
    c[:mPOM] .= Float64(3.5f-3) .* gr.m      # Fortran: 3.5e-3 (single precision)
    par = (; epsilonR=g("epsilonR"), kBasal=g("kBasal"), kSDA=g("kSDA"))
    return c, par
end

# POM.f90: initPOM (default sinking 400*m^0.513, palatability from input, no mort2)
function init_pom(P, n, mMax)
    g(k) = getparam(P, "POM", k)
    gr = calc_grid(n, g("mMin"), mMax)
    c = class_block(n, gr)
    c[:mort2constant] .= 0.0
    c[:velocity] .= 400 .* gr.m .^ 0.513
    c[:palat] .= g("palatability")
    return c, g("remin")
end

# ---------------------------------------------------------------- building a setup

"""
    build_setup(name, inputfile, nNutrients, specs; mortHTL, quadraticHTL, decliningHTL)

parametersInit + parametersAddGroup for each `(type, n, mMax)` in `specs` +
parametersFinalize(mortHTL, quadraticHTL, decliningHTL).
"""
function build_setup(name, inputfile, nNutrients, specs; mortHTL, quadraticHTL, decliningHTL)
    P = read_input_file(inputfile)
    blocks = Dict{Symbol,Vector{Float64}}[]
    groups = Group[]
    zeroN(k) = NamedTuple{k}(ntuple(_ -> 0.0, length(k)))
    gen = zeroN((:epsilonL, :bL, :bN, :bDOC, :bF, :bg, :remin2, :reminF))
    gens = zeroN((:epsilonL, :remin2, :reminF))
    dia = zeroN((:epsilonL, :bL, :bN, :bDOC, :bSi, :bg, :remin2, :rhoCSi))
    dias = zeroN((:epsilonL, :bN, :bSi, :remin2, :rhoCSi))
    cop = zeroN((:epsilonR, :kBasal, :kSDA))
    pomremin = 0.0
    i0 = 0
    for (t, n, mMax) in specs
        if t == generalist
            c, gen = init_generalists(P, n)
        elseif t == generalist_simple
            c, gens = init_generalists_simple(P, n)
        elseif t == diatom
            c, dia = init_diatoms(P, n)
        elseif t == diatom_simple
            c, dias = init_diatoms_simple(P, n, mMax)
        elseif is_copepod(t)
            c, cop = init_copepod(P, t == copepod_active ? "copepods_active" : "copepods_passive", n, mMax)
        elseif t == pom
            c, pomremin = init_pom(P, n, mMax)
        end
        push!(blocks, c); push!(groups, Group(t, i0+1:i0+n)); i0 += n
    end
    nS = i0
    cat_(f) = reduce(vcat, [b[f] for b in blocks])
    m = cat_(:m); z = cat_(:z); palat = cat_(:palat); beta = cat_(:beta); sigma = cat_(:sigma)
    diatPref = cat_(:diatPref)

    # theta (parametersFinalize); copepods have a lower preference for (both kinds of) diatoms
    theta = zeros(nS, nS)
    for gi in groups, gj in groups, i in gi.ix, j in gj.ix
        theta[i, j] = palat[j] * calc_phi(m[i] / m[j], beta[i], sigma[i], z[i])
        if is_copepod(gi.type) && is_diatom(gj.type)
            theta[i, j] = diatPref[i] * theta[i, j]
        end
    end
    # POM routing
    pomg = findfirst(gr -> gr.type == pom, groups)
    ixPOM = pomg === nothing ? (1:0) : groups[pomg].ix
    mLower = cat_(:mLower); mDelta = cat_(:mDelta); mPOM = cat_(:mPOM)
    thetaPOM = zeros(nS, max(length(ixPOM), 1))
    if pomg !== nothing
        for gr in groups
            gr.type == pom && continue
            for i in gr.ix
                j = 1
                while mPOM[i] > mLower[ixPOM[j]] + mDelta[ixPOM[j]] && j < length(ixPOM)
                    j += 1
                end
                thetaPOM[i, j] = 1.0
            end
        end
    end
    isPOM = zeros(nS); isPOM[ixPOM] .= 1.0

    s = NUMSetup(name, nNutrients, nS, groups,
        getparam(P, "general", "rhoCN"), getparam(P, "general", "fracHTL_to_N"),
        get(get(P, "general", Dict{String,Float64}()), "fracHTL_to_POM", NaN),   # v1.0 input.h only; unused
        m, mLower, mDelta, z, cat_(:AF), cat_(:JFmax), cat_(:epsilonF), cat_(:mort2constant),
        zeros(nS), cat_(:velocity), cat_(:AN), cat_(:AL), cat_(:Jmax), cat_(:Jresp),
        cat_(:JlossPassive), cat_(:JrespFactor), mPOM, theta, thetaPOM, isPOM, ixPOM, true,
        gen, gens, dia, dias, cop, pomremin, zeros(nNutrients + nS))
    # HTL mortality of parametersFinalize: 50% at the largest mass / 500^1.5
    mHTL = maximum(m[last(gr.ix)] for gr in groups) / 500.0^1.5
    set_htl!(s, mHTL, mortHTL, quadraticHTL, decliningHTL, false)
    return s
end

"""
    set_htl!(s, mHTL, mortalityHTL, quadratic, declining, copepodsOnly)

NUMmodel.f90 setHTL (Matlab `setHTL(mortalityHTL, mHTL, ...)` calls it with this order).
"""
function set_htl!(s::NUMSetup, mHTL, mortalityHTL, quadratic, declining, copepodsOnly)
    p = zeros(s.nS)
    for gr in s.groups
        ix = gr.ix
        if gr.type != pom
            x = s.m[ix] ./ mHTL
            p[ix] .= 1 ./ (1 .+ 1 ./ (x .* x))          # (m/mHTL)**(-2): integer power
            declining && (p[ix] .*= x .^ (-0.25))
        end
        copepodsOnly && !is_copepod(gr.type) && (p[ix] .= 0.0)
    end
    s.pHTL = quadratic ? mortalityHTL .* p ./ log.(1 ./ s.z) : mortalityHTL .* p
    s.bQuadraticHTL = quadratic
    return s
end

"""
    set_mort_htl!(s, pHTL, quadratic)

NUMmodel.f90 setMortHTL: overrides the selectivity (the Fortran `mortHTL` argument only sets
a value that calcDerivatives recomputes, so it has no effect on the dynamics).
"""
function set_mort_htl!(s::NUMSetup, pHTL::AbstractVector, quadratic::Bool)
    s.pHTL = collect(Float64, pHTL); s.bQuadraticHTL = quadratic
    return s
end

"set_sinking!(s, velocity): NUMmodel.f90 setSinking (velocity per biomass class, m/day)."
set_sinking!(s::NUMSetup, velocity::AbstractVector) = (s.velocity = collect(Float64, velocity); s)

"Matlab setSinkingPOM: POM sinking velocity (scalar or one per POM class); others 0."
function set_sinking_pom!(s::NUMSetup, velocity)
    isempty(s.ixPOM) && error("There is no POM in this setup")
    v = zeros(s.nS); v[s.ixPOM] .= velocity
    return set_sinking!(s, v)
end

# Matlab p.u0 conventions
function set_u0!(s, nutrients, biomass)
    s.u0 = zeros(s.nNutrients + s.nS)
    s.u0[1:s.nNutrients] .= nutrients
    s.u0[s.nNutrients+1:end] .= biomass
    s.u0[s.nNutrients .+ s.ixPOM] .= 0.0
    return s
end
sheldon_u0(s) = 0.1 .* log.((s.mLower .+ s.mDelta) ./ s.mLower)

"""
    numroot()

The upstream NUMmodel checkout: `ENV["NUM_ROOT"]` if set, otherwise the `NUMmodel` submodule of
this repository. Holds `input/input.yaml` and `TMs/`.
"""
numroot() = get(ENV, "NUM_ROOT", normpath(joinpath(@__DIR__, "..", "..", "..", "NUMmodel")))

function default_input()
    f = joinpath(numroot(), "input", "input.yaml")
    isfile(f) || error("NUM input file not found: $f. Run `git submodule update --init` in the repository, " *
                       "or set ENV[\"NUM_ROOT\"] to a NUMmodel checkout.")
    return f
end

# ---------------------------------------------------------------- the model setups
# Fortran setupXXX + the Matlab wrapper where one exists (initial conditions, sinking, HTL).

"setupGeneralistsSimpleOnly: simple generalists, N and DOC."
function setup_generalists_simple_only(n=10; inputfile=default_input())
    s = build_setup("GeneralistsSimpleOnly", inputfile, 2, [(generalist_simple, n, 0.0)];
                    mortHTL=0.1, quadraticHTL=false, decliningHTL=false)
    return set_u0!(s, [150.0, 0.0], 1.0)
end

"setupGeneralistsOnly: generalists, N and DOC."
function setup_generalists_only(n=10; inputfile=default_input())
    s = build_setup("GeneralistsOnly", inputfile, 2, [(generalist, n, 0.0)];
                    mortHTL=0.1, quadraticHTL=false, decliningHTL=false)
    return set_u0!(s, [150.0, 0.0], 1.0)
end

"""setupGeneralistsSimplePOM: simple generalists + POM (max 1 ugC), POM sinking 11 m/day.
(The Matlab wrapper calls setSinkingPOM(p, 11) with a scalar, which errors there unless nPOM = 1.)"""
function setup_generalists_simple_pom(n=10, nPOM=10; inputfile=default_input())
    s = build_setup("GeneralistsSimplePOM", inputfile, 2, [(generalist_simple, n, 0.0), (pom, nPOM, 1.0)];
                    mortHTL=0.1, quadraticHTL=false, decliningHTL=false)
    set_sinking_pom!(s, 11.0)
    return set_u0!(s, [150.0, 0.0], 1.0)
end

"setupGeneralistsPOM: generalists + POM (max 1 ugC), POM sinking 11 m/day."
function setup_generalists_pom(n=10, nPOM=10; inputfile=default_input())
    s = build_setup("GeneralistsPOM", inputfile, 2, [(generalist, n, 0.0), (pom, nPOM, 1.0)];
                    mortHTL=0.1, quadraticHTL=false, decliningHTL=false)
    set_sinking_pom!(s, fill(11.0, nPOM))
    return set_u0!(s, [150.0, 0.0], 1.0)
end

"setupDiatomsOnly: diatoms, N, DOC and Si."
function setup_diatoms_only(n=10; inputfile=default_input())
    s = build_setup("DiatomsOnly", inputfile, 3, [(diatom, n, 1.0)];
                    mortHTL=0.1, quadraticHTL=false, decliningHTL=false)
    return set_u0!(s, [150.0, 0.0, 200.0], 1.0)
end

"setupDiatoms_simpleOnly: simple diatoms (max 1 ugC), N, DOC and Si."
function setup_diatoms_simple_only(n=10; inputfile=default_input())
    s = build_setup("Diatoms_simpleOnly", inputfile, 3, [(diatom_simple, n, 1.0)];
                    mortHTL=0.1, quadraticHTL=false, decliningHTL=false)
    return set_u0!(s, [150.0, 0.0, 10.0], 1.0)
end

"setupGeneralistsDiatoms: generalists + diatoms."
function setup_generalists_diatoms(n=10; inputfile=default_input())
    s = build_setup("GeneralistsDiatoms", inputfile, 3, [(generalist, n, 0.0), (diatom, n, 0.0)];
                    mortHTL=0.1, quadraticHTL=false, decliningHTL=false)
    return set_u0!(s, [150.0, 0.0, 200.0], 1.0)
end

"setupGeneralistsDiatoms_simple (no Matlab wrapper; u0 as setupGeneralistsDiatoms)."
function setup_generalists_diatoms_simple(n=10; inputfile=default_input())
    s = build_setup("GeneralistsDiatoms_simple", inputfile, 3, [(generalist_simple, n, 0.0), (diatom_simple, n, 1.0)];
                    mortHTL=0.1, quadraticHTL=false, decliningHTL=false)
    return set_u0!(s, [150.0, 0.0, 200.0], 1.0)
end

"setupGeneralistsSimpleCopepod: 10 simple generalists + one active copepod (adult 0.1 ugC). No Matlab wrapper."
function setup_generalists_simple_copepod(; inputfile=default_input())
    s = build_setup("GeneralistsSimpleCopepod", inputfile, 2,
                    [(generalist_simple, 10, 0.0), (copepod_active, 10, 0.1)];
                    mortHTL=0.003, quadraticHTL=true, decliningHTL=true)
    return set_u0!(s, [150.0, 0.0], 1.0)
end

"setupGeneric(mAdult): 10 simple generalists + active copepods with the given adult masses."
function setup_generic(mAdult=Float64[]; inputfile=default_input())
    specs = vcat([(generalist_simple, 10, 0.0)], [(copepod_active, 10, ma) for ma in mAdult])
    s = build_setup("Generic", inputfile, 2, specs; mortHTL=isempty(mAdult) ? 0.1 : 0.001,
                    quadraticHTL=true, decliningHTL=true)
    return set_u0!(s, [150.0, 0.0], 1.0)
end

"""
    setup_num_model(inputfile=default; mAdultPassive=[0.2,5], mAdultActive=10 .^ range(0,3,3),
                    n=10, nCopepod=6, nPOM=1, mortHTL=0.006, mHTL=1.0, velocityPOM=20.0)

Matlab `setupNUMmodel`: Fortran setupNUMmodel + setSinkingPOM(20) + setHTL(0.006, 1, true,
false, true).
"""
function setup_num_model(inputfile::AbstractString=default_input();
                         mAdultPassive=[0.2, 5.0], mAdultActive=collect(10 .^ range(0, 3, length=3)),
                         n=10, nCopepod=6, nPOM=1,
                         mortHTL=0.006, mHTL=1.0, bQuadraticHTL=true, bDecliningHTL=false,
                         bCopepodsOnly=true, velocityPOM=20.0)
    # POM max size: largest mPOM of the last copepod group (3.5e-3 * its largest mass)
    specs = vcat([(generalist, n, 0.0), (diatom, n, 1.0)],
                 [(copepod_passive, nCopepod, ma) for ma in mAdultPassive],
                 [(copepod_active, nCopepod, ma) for ma in mAdultActive])
    s0 = build_setup("tmp", inputfile, 3, specs; mortHTL=0.13, quadraticHTL=true, decliningHTL=false)
    mMaxPOM = maximum(s0.mPOM[last(s0.groups).ix])
    s = build_setup("NUMmodel", inputfile, 3, vcat(specs, [(pom, nPOM, mMaxPOM)]);
                    mortHTL=0.13, quadraticHTL=true, decliningHTL=false)
    set_htl!(s, 1.0, 0.13, true, false, true)                        # Fortran setupNUMmodel
    set_sinking_pom!(s, fill(velocityPOM, nPOM))                      # Matlab setSinkingPOM
    set_htl!(s, mHTL, mortHTL, bQuadraticHTL, bDecliningHTL, bCopepodsOnly)   # Matlab setHTL
    return set_u0!(s, [150.0, 0.0, 200.0], sheldon_u0(s))
end

"""
    setup_num_model_simple(mAdult=[0.1,1,10,100,1000]; n=10, nCopepod=10, nPOM=10)

setupNUMmodelSimple: simple generalists, simple diatoms (max 1 ugC), active copepods, POM.
(The Matlab wrapper labels the copepod groups as passive; the Fortran makes them active.)
"""
function setup_num_model_simple(mAdult=[0.1, 1, 10, 100, 1000]; n=10, nCopepod=10, nPOM=10,
                                inputfile=default_input())
    specs = vcat([(generalist_simple, n, 0.0), (diatom_simple, n, 1.0)],
                 [(copepod_active, nCopepod, ma) for ma in mAdult])
    s0 = build_setup("tmp", inputfile, 3, specs; mortHTL=0.001, quadraticHTL=true, decliningHTL=true)
    mMaxPOM = maximum(s0.mPOM[last(s0.groups).ix])
    s = build_setup("NUMmodelSimple", inputfile, 3, vcat(specs, [(pom, nPOM, mMaxPOM)]);
                    mortHTL=0.001, quadraticHTL=true, decliningHTL=true)
    return set_u0!(s, [150.0, 0.0, 200.0], sheldon_u0(s))
end

"setupGenDiatCope(n, nCopepod, nPOM, mAdult): diatoms, generalists, active copepods, POM. No Matlab wrapper."
function setup_gen_diat_cope(n=10, nCopepod=10, nPOM=10, mAdult=[0.1, 1, 10, 100, 1000];
                             inputfile=default_input())
    specs = vcat([(diatom, n, 0.0), (generalist, n, 0.0)],
                 [(copepod_active, nCopepod, ma) for ma in mAdult])
    s0 = build_setup("tmp", inputfile, 3, specs; mortHTL=0.007, quadraticHTL=true, decliningHTL=false)
    mMaxPOM = maximum(s0.mPOM[last(s0.groups).ix])
    s = build_setup("GenDiatCope", inputfile, 3, vcat(specs, [(pom, nPOM, mMaxPOM)]);
                    mortHTL=0.007, quadraticHTL=true, decliningHTL=false)
    return set_u0!(s, [150.0, 0.0, 200.0], sheldon_u0(s))
end

"initial_state(s): Matlab p.u0 of the setup."
initial_state(s::NUMSetup) = copy(s.u0)
