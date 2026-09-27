"""
Thin ccall wrapper around the upstream library (NUMmodel/lib/libNUMmodel_matlab.so, built from
the NUMmodel checkout with cmake), used as the reference ("oracle") for the port. The library
reads `../input/input.yaml` relative to
the working directory, so setups run from NUMmodel/matlab. The library holds one setup at a time
(module globals), like a Matlab session.
"""
module FortranOracle

const NUMROOT = get(ENV, "NUM_ROOT", normpath(joinpath(@__DIR__, "..", "..", "..", "NUMmodel")))
const LIB = joinpath(NUMROOT, "lib", "libNUMmodel_matlab." * (Sys.isapple() ? "dylib" : "so"))
isfile(LIB) || error("""The Fortran reference library $LIB is not built. In $NUMROOT run
    cmake -S . -B build && cmake --build build && cmake --install build
(needs gfortran and cmake; see the top-level README).""")

const NGRID = Ref(0)
const NNUT = Ref(0)

function _check(errorio, errstr, name)
    errorio[] && error("$name: " * String(filter(!iszero, errstr)))
end

"""
    setup!(name; kwargs...) -> nGrid

Run one of the Fortran setups (names as the Julia setup functions without `setup_`):
generalists_simple_only, generalists_only, generalists_simple_pom, generalists_pom,
diatoms_only, diatoms_simple_only, generalists_diatoms, generalists_diatoms_simple,
generalists_simple_copepod, generic, num_model, num_model_simple, gen_diat_cope.
`nGrid` must be given (the library cannot report it); the Julia setup provides it.
"""
function setup!(name::Symbol, nGrid::Int, nNutrients::Int; n=10, nPOM=10, nCopepod=10,
                mAdult=Float64[], mAdultPassive=[0.2, 5.0], mAdultActive=collect(10 .^ range(0, 3, length=3)))
    errorio = Ref{Bool}(false)
    errstr = zeros(UInt8, 256)
    cd(joinpath(NUMROOT, "matlab")) do
        if name == :generalists_simple_only
            ccall((:f_setupgeneralistssimpleonly, LIB), Cvoid, (Cint, Ref{Bool}, Ptr{UInt8}), n, errorio, errstr)
        elseif name == :generalists_only
            ccall((:f_setupgeneralistsonly, LIB), Cvoid, (Cint, Ref{Bool}, Ptr{UInt8}), n, errorio, errstr)
        elseif name == :generalists_simple_pom
            ccall((:f_setupgeneralistssimplepom, LIB), Cvoid, (Cint, Cint, Ref{Bool}, Ptr{UInt8}), n, nPOM, errorio, errstr)
        elseif name == :generalists_pom
            ccall((:f_setupgeneralistspom, LIB), Cvoid, (Cint, Cint, Ref{Bool}, Ptr{UInt8}), n, nPOM, errorio, errstr)
        elseif name == :diatoms_only
            ccall((:f_setupdiatomsonly, LIB), Cvoid, (Cint, Ref{Bool}, Ptr{UInt8}), n, errorio, errstr)
        elseif name == :diatoms_simple_only
            ccall((:f_setupdiatoms_simpleonly, LIB), Cvoid, (Cint, Ref{Bool}, Ptr{UInt8}), n, errorio, errstr)
        elseif name == :generalists_diatoms
            ccall((:f_setupgeneralistsdiatoms, LIB), Cvoid, (Cint, Ref{Bool}, Ptr{UInt8}), n, errorio, errstr)
        elseif name == :generalists_diatoms_simple
            ccall((:f_setupgeneralistsdiatoms_simple, LIB), Cvoid, (Cint, Ref{Bool}, Ptr{UInt8}), n, errorio, errstr)
        elseif name == :generalists_simple_copepod
            ccall((:f_setupgeneralistssimplecopepod, LIB), Cvoid, (Ref{Bool}, Ptr{UInt8}), errorio, errstr)
        elseif name == :generic
            ma = collect(Float64, mAdult)
            ccall((:f_setupgeneric, LIB), Cvoid, (Cint, Ptr{Float64}, Ref{Bool}, Ptr{UInt8}), length(ma), ma, errorio, errstr)
        elseif name == :num_model
            mp = collect(Float64, mAdultPassive); mact = collect(Float64, mAdultActive)
            ccall((:f_setupnummodel, LIB), Cvoid,
                  (Cint, Cint, Cint, Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ref{Bool}, Ptr{UInt8}),
                  n, nCopepod, nPOM, length(mp), mp, length(mact), mact, errorio, errstr)
        elseif name == :num_model_simple
            ma = collect(Float64, mAdult)
            ccall((:f_setupnummodelsimple, LIB), Cvoid, (Cint, Cint, Cint, Cint, Ptr{Float64}, Ref{Bool}, Ptr{UInt8}),
                  n, nCopepod, nPOM, length(ma), ma, errorio, errstr)
        elseif name == :gen_diat_cope
            ma = collect(Float64, mAdult)
            ccall((:f_setupgendiatcope, LIB), Cvoid, (Cint, Cint, Cint, Cint, Ptr{Float64}, Ref{Bool}, Ptr{UInt8}),
                  n, nCopepod, nPOM, length(ma), ma, errorio, errstr)
        else
            error("unknown setup $name")
        end
    end
    _check(errorio, errstr, "setup $name")
    NGRID[] = nGrid; NNUT[] = nNutrients
    return nGrid
end

"Matlab setupNUMmodel: Fortran setup, POM sinking 20 m/d, setHTL(0.006, 1, true, false, true)."
function setup_num_model!(; n=10, nCopepod=6, nPOM=1, mAdultPassive=[0.2, 5.0],
                          mAdultActive=collect(10 .^ range(0, 3, length=3)),
                          mortHTL=0.006, mHTL=1.0, velocityPOM=20.0)
    nG = 3 + 2n + nCopepod * (length(mAdultPassive) + length(mAdultActive)) + nPOM
    setup!(:num_model, nG, 3; n, nCopepod, nPOM, mAdultPassive, mAdultActive)
    vel = zeros(nG)
    get_sinking!(vel)
    vel[end-nPOM+1:end] .= velocityPOM
    set_sinking!(vel)
    set_htl!(mHTL, mortHTL, true, false, true)
    return nG
end

get_sinking!(vel) = (ccall((:f_getsinking, LIB), Cvoid, (Ptr{Float64},), vel); vel)
set_sinking!(vel) = ccall((:f_setsinking, LIB), Cvoid, (Ptr{Float64},), collect(Float64, vel))
set_htl!(mHTL, mortHTL, quad, declining, copepodsOnly) =
    ccall((:f_sethtl, LIB), Cvoid, (Float64, Float64, Bool, Bool, Bool), mHTL, mortHTL, quad, declining, copepodsOnly)
set_mort_htl!(mortHTL, pHTL, quad) =
    ccall((:f_setmorthtl, LIB), Cvoid, (Float64, Ptr{Float64}, Bool), mortHTL, collect(Float64, pHTL), quad)

function calc_derivatives(u::Vector{Float64}, L, T, dt)
    dudt = zeros(NGRID[])
    ccall((:f_calcderivatives, LIB), Cvoid, (Ptr{Float64}, Float64, Float64, Float64, Ptr{Float64}),
          u, L, T, dt, dudt)
    return dudt
end

function simulate_euler(u::Vector{Float64}, L, T, tEnd, dt)
    u = copy(u)
    ccall((:f_simulateeuler, LIB), Cvoid, (Ptr{Float64}, Float64, Float64, Float64, Float64),
          u, L, T, tEnd, dt)
    return u
end

function simulate_chemostat_euler(u::Vector{Float64}, L, T, Ndeep, diff, widthProductiveLayer, tEnd, dt, bLosses)
    u = copy(u); nd = collect(Float64, Ndeep)
    ccall((:f_simulatechemostateuler, LIB), Cvoid,
          (Ptr{Float64}, Float64, Float64, Cint, Ptr{Float64}, Float64, Float64, Float64, Float64, Bool),
          u, L, T, length(nd), nd, diff, widthProductiveLayer, tEnd, dt, bLosses)
    return u
end

"Fortran getFunctions after a calcDerivatives call (uses the rates stored by that call)."
function get_functions(u::Vector{Float64}, L, T, dt)
    calc_derivatives(u, L, T, dt)
    return get_functions_last(u)
end
function get_functions_last(u::Vector{Float64})
    r1, r2, r3, r4, r5, r6, r7, r8, r9 = (Ref(0.0) for _ in 1:9)
    ccall((:f_getfunctions, LIB), Cvoid,
          (Ptr{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64},
           Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}), u, r1, r2, r3, r4, r5, r6, r7, r8, r9)
    return (ProdGross=r1[], ProdNet=r2[], ProdHTL=r3[], ProdBact=r4[], eHTL=r5[],
            Bpico=r6[], Bnano=r7[], Bmicro=r8[], mHTL=r9[])
end

"simulateEulerFunctions: Euler steps, then getFunctions with the rates of the last substep."
function simulate_euler_functions(u::Vector{Float64}, L, T, tEnd, dt)
    u = copy(u)
    r = [Ref(0.0) for _ in 1:9]
    ccall((:f_simulateeulerfunctions, LIB), Cvoid,
          (Ptr{Float64}, Float64, Float64, Float64, Float64, Ref{Float64}, Ref{Float64}, Ref{Float64},
           Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}),
          u, L, T, tEnd, dt, r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9])
    return u, (ProdGross=r[1][], ProdNet=r[2][], ProdHTL=r[3][], ProdBact=r[4][], eHTL=r[5][],
               Bpico=r[6][], Bnano=r[7][], Bmicro=r[8][], mHTL=r[9][])
end

const RATE_NAMES = (:jN, :jDOC, :jL, :jSi, :jF, :jFreal, :f, :jTot, :jMax, :jFmax, :jR, :jResptot,
                    :jLossPassive, :jNloss, :jLreal, :jPOM, :mortpred, :mortHTL, :mort2, :mort)
"Fortran getRates after a calcDerivatives call."
function get_rates(u::Vector{Float64}, L, T, dt)
    calc_derivatives(u, L, T, dt)
    nB = NGRID[] - NNUT[]
    a = [zeros(nB) for _ in RATE_NAMES]
    ccall((:f_getrates, LIB), Cvoid,
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
          a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15],
          a[16], a[17], a[18], a[19], a[20])
    return NamedTuple{RATE_NAMES}(Tuple(a))
end

"Fortran getBalance(u, dudt) and getLost(u) after a calcDerivatives call."
function get_balance(u::Vector{Float64}, L, T, dt)
    dudt = calc_derivatives(u, L, T, dt)
    c, n, si = Ref(0.0), Ref(0.0), Ref(0.0)
    ccall((:f_getbalance, LIB), Cvoid, (Ptr{Float64}, Ptr{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}),
          u, dudt, c, n, si)
    cl, nl, sl = Ref(0.0), Ref(0.0), Ref(0.0)
    ccall((:f_getlost, LIB), Cvoid, (Ptr{Float64}, Ref{Float64}, Ref{Float64}, Ref{Float64}), u, cl, nl, sl)
    return (Cbalance=c[], Nbalance=n[], Sibalance=si[], Clost=cl[], Nlost=nl[], SiLost=sl[])
end

function get_mass()
    m = zeros(NGRID[]); mDelta = zeros(NGRID[])
    ccall((:f_getmass, LIB), Cvoid, (Ptr{Float64}, Ptr{Float64}), m, mDelta)
    return m, mDelta
end

function get_theta()
    th = zeros(NGRID[], NGRID[])
    ccall((:f_gettheta, LIB), Cvoid, (Ptr{Float64},), th)
    return th
end

end
