# Diagnostics of NUMmodel.f90 for every box, from the rates of one calcDerivatives evaluation:
# getFunctions (production, HTL, pico/nano/micro biomass), getRates (per-class rates),
# getLost and getBalance (C, N, Si conservation), and simulateEulerFunctions (Euler steps, then
# the functions from the rates of the last substep). Same formulas and quirks as the Fortran,
# e.g. getRates multiplies jN and jDOC by fTemp15 a second time; getLost/getBalance hard-code
# rhoCSi = 3.4.

const FUNCTION_NAMES = (:ProdGross, :ProdNet, :ProdHTL, :ProdBact, :eHTL, :Bpico, :Bnano, :Bmicro, :mHTL)
const BALANCE_NAMES = (:Clost, :Nlost, :SiLost, :Cbalance, :Nbalance, :Sibalance)
const RATE_NAMES = (:jN, :jDOC, :jL, :jSi, :jF, :jFreal, :f, :jTot, :jMax, :jFmax, :jR, :jResptot,
                    :jLossPassive, :jNloss, :jLreal, :jPOM, :mortpred, :mortHTL, :mort2, :mort)
const NDIAG = length(FUNCTION_NAMES) + length(BALANCE_NAMES)

@inline is_uni(t) = (t == T_GEN) | (t == T_GENS) | (t == T_DIA) | (t == T_DIAS)
@inline eps_light(t, k) = t == T_GEN ? k.gen.epsilonL : t == T_GENS ? k.gens.epsilonL :
                          t == T_DIA ? k.dia.epsilonL : k.dias.epsilonL

# getFunctions (conversion factors are 1: mgC/m3/day and mgC/m3), with the raw state u
@inline function box_functions(u, R, k::KSetup{D}) where {D}
    nS, nN = D[1], D[2]
    ProdGross = 0.0; ProdNet = 0.0; ProdBact = 0.0
    @inbounds for (t, i0, n) in k.groups
        is_uni(t) || continue
        sG = 0.0; sN = 0.0; sB = 0.0; eL = eps_light(t, k)
        for i in i0:i0+n-1
            m = k.m[i]; ui = u[nN+i]
            sG += R[i, R_JLREAL] * ui / m
            sN += max(0.0, (R[i, R_JLREAL] / eL - R[i, R_JRESPTOT]) * ui / m)   # Develop getProdNet
            sB += t == T_GENS ? max(0.0, R[i, R_JDOC] - R[i, R_JRESPTOT]) * ui / m :
                                max(0.0, (R[i, R_JDOCREAL] - R[i, R_JRESPTOT]) * ui / m)
        end
        ProdGross = ProdGross + 1.0 * sG; ProdNet = ProdNet + 1.0 * sN; ProdBact = ProdBact + 1.0 * sB
    end
    Bpico = 0.0; Bnano = 0.0; Bmicro = 0.0
    @inbounds for i in 1:nS
        esd = 15000.0 * (k.m[i] * 1e-6)^(1.0 / 3.0)
        ui = u[nN+i]
        esd <= 2.0 && (Bpico += 1.0 * ui)
        (esd > 2.0) & (esd <= 20.0) && (Bnano += 1.0 * ui)
        esd > 20.0 && (Bmicro += 1.0 * ui)
    end
    ProdHTL = 0.0; mHTL = 0.0
    @inbounds for (t, i0, n) in k.groups
        sP = 0.0; sM = 0.0
        for i in i0:i0+n-1
            sP += R[i, R_MORTHTL] * u[nN+i]
            sM += log(k.m[i]) * R[i, R_MORTHTL] * u[nN+i]
        end
        ProdHTL = ProdHTL + 1.0 * sP; mHTL = mHTL + 1.0 * sM
    end
    mHTL = exp(mHTL / ProdHTL)
    return (ProdGross, ProdNet, ProdHTL, ProdBact, ProdHTL / ProdNet, Bpico, Bnano, Bmicro, mHTL)
end

# getLost with the raw state u
@inline function box_lost(u, R, k::KSetup{D}) where {D}
    nN = D[2]
    Clost = 0.0; Nlost = 0.0; SiLost = 0.0
    @inbounds for (t, i0, n) in k.groups
        sC = 0.0; sH = 0.0; sP = 0.0; sM = 0.0
        for i in i0:i0+n-1
            m = k.m[i]; ui = u[nN+i]
            sC += is_uni(t) ? (-R[i, R_JLREAL] - R[i, R_CLOSSPHOTO] + R[i, R_JRESPTOT]) / m * ui :
                              R[i, R_JRESPTOT] * ui / m
            sH += R[i, R_MORTHTL] * ui
            sP += R[i, R_JPOM] * ui
            sM += R[i, R_MORTPRED] * ui
        end
        Clost = Clost + sC
        if k.iP == 0
            Nlost = Nlost + (1 - k.fracHTL_to_N) * sH / k.rhoCN
            Clost = Clost + sH
            Nlost = Nlost + sP / k.rhoCN
            Clost = Clost + sP
        else
            Clost = Clost + k.fracHTL_to_N * sH
        end
        if (t == T_DIA) | (t == T_DIAS)
            SiLost = SiLost + (sM + sH + sP) / 3.4
        end
    end
    return (Clost, Nlost, SiLost)
end

# getBalance(u, dudt), normalized by DOC, N and Si
@inline function box_balance(u, du, lost, k::KSetup{D}) where {D}
    nS, nN = D[1], D[2]
    Clost, Nlost, SiLost = lost
    sB = 0.0
    @inbounds for i in 1:nS
        sB += du[nN+i]
    end
    C = du[2] + sB + Clost
    N = du[1] + sB / k.rhoCN + Nlost
    Si = du[3] + SiLost          # (for 2 nutrients this is a biomass class; the result is set to 0)
    @inbounds for (t, i0, n) in k.groups
        if (t == T_DIA) | (t == T_DIAS)
            sD = 0.0
            for i in i0:i0+n-1
                sD += du[nN+i]
            end
            Si = Si + sD / 3.4
        end
    end
    return (C / u[2], N / u[1], nN > 2 ? Si / u[3] : 0.0)
end

@kernel inbounds = true function diagnostics_kernel!(Out, @Const(U), @Const(L), @Const(T), dt, k)
    b = @index(Global)
    box_diagnostics_global!(Out, U, L, T, dt, k, b)
end
@inline function box_diagnostics_global!(Out, U, L, T, dt, k::KSetup{D}, b) where {D}
    nS = D[1]; nGrid = D[1] + D[2]
    u = load_box(U, b, Val(nGrid)); du = MVector{nGrid,Float64}(undef)
    R = MMatrix{nS,NRATES,Float64}(undef)
    box_derivs!(du, u, L[b], T[b], dt, k, R)
    fn = box_functions(u, R, k)
    lost = box_lost(u, R, k)
    bal = box_balance(u, du, lost, k)
    @inbounds for j in 1:9
        Out[b, j] = fn[j]
    end
    @inbounds for j in 1:3
        Out[b, 9+j] = lost[j]; Out[b, 12+j] = bal[j]
    end
end

@kernel inbounds = true function rates_kernel!(Rt, @Const(U), @Const(L), @Const(T), dt, k)
    b = @index(Global)
    box_rates_global!(Rt, U, L, T, dt, k, b)
end
@inline function box_rates_global!(Rt, U, L, T, dt, k::KSetup{D}, b) where {D}
    nS, nN = D[1], D[2]; nGrid = nS + nN
    u = load_box(U, b, Val(nGrid)); du = MVector{nGrid,Float64}(undef)
    R = MMatrix{nS,NRATES,Float64}(undef)
    Tb = T[b]
    box_derivs!(du, u, L[b], Tb, dt, k, R)
    fT2 = 2.0^((Tb - TREF) / 10.0); fT15 = 1.5^((Tb - TREF) / 10.0)
    @inbounds for (t, i0, n) in k.groups
        uni = is_uni(t)
        for i in i0:i0+n-1
            m = k.m[i]
            Rt[b, i, 1] = uni ? fT15 * R[i, R_JN] / m : 0.0                    # jN
            Rt[b, i, 2] = uni & (t != T_DIAS) ? fT15 * R[i, R_JDOC] / m : 0.0  # jDOC
            Rt[b, i, 3] = uni ? R[i, R_JL] / m : 0.0                           # jL
            Rt[b, i, 4] = (t == T_DIA) | (t == T_DIAS) ? R[i, R_JSI] / m : 0.0  # jSi
            Rt[b, i, 5] = R[i, R_FLVL] * k.JFmax[i] / m                        # jF
            Rt[b, i, 6] = R[i, R_JF] / m                                       # jFreal
            Rt[b, i, 7] = uni ? R[i, R_F] : 0.0                                # f
            Rt[b, i, 8] = R[i, R_JTOT] / m                                     # jTot
            Rt[b, i, 9] = uni ? fT2 * k.Jmax[i] / m : 0.0                      # jMax
            Rt[b, i, 10] = fT2 * k.JFmax[i] / m                                # jFmax
            Rt[b, i, 11] = fT2 * k.Jresp[i] / m                                # jR
            Rt[b, i, 12] = R[i, R_JRESPTOT] / m                                # jResptot
            Rt[b, i, 13] = uni ? k.JlossPassive[i] / m : 0.0                   # jLossPassive
            Rt[b, i, 14] = R[i, R_JNLOSS] / m                                  # jNloss
            Rt[b, i, 15] = uni ? R[i, R_JLREAL] / m : 0.0                      # jLreal
            Rt[b, i, 16] = R[i, R_JPOM]                                        # jPOM
            Rt[b, i, 17] = R[i, R_MORTPRED]                                    # mortpred
            Rt[b, i, 18] = R[i, R_MORTHTL]                                     # mortHTL
            Rt[b, i, 19] = R[i, R_MORT2]                                       # mort2
            Rt[b, i, 20] = 0.0                                                 # mort (Fortran sets 0)
        end
    end
end

@kernel inbounds = true function euler_functions_kernel!(U, Fn, @Const(L), @Const(T), dt, nsub, k)
    b = @index(Global)
    box_euler_functions!(U, Fn, L, T, dt, nsub, k, b)
end
@inline function box_euler_functions!(U, Fn, L, T, dt, nsub, k::KSetup{D}, b) where {D}
    nS = D[1]; nGrid = D[1] + D[2]
    u = load_box(U, b, Val(nGrid)); du = MVector{nGrid,Float64}(undef)
    R = MMatrix{nS,NRATES,Float64}(undef)
    Lb = L[b]; Tb = T[b]
    for s in 1:nsub
        s == nsub ? box_derivs!(du, u, Lb, Tb, dt, k, R) : box_derivs!(du, u, Lb, Tb, dt, k)
        @inbounds for j in 1:nGrid
            u[j] = u[j] + du[j] * dt
        end
    end
    fn = box_functions(u, R, k)
    @inbounds for j in 1:nGrid
        U[b, j] = u[j]
    end
    @inbounds for j in 1:9
        Fn[b, j] = fn[j]
    end
end

"""
    get_functions(U, L, T, dt, k) -> NamedTuple of length-nb vectors

Fortran getFunctions after calcDerivatives for every box: ProdGross, ProdNet, ProdBact
(mgC/m3/day), ProdHTL, eHTL, Bpico, Bnano, Bmicro (mgC/m3), mHTL (ugC).
"""
function get_functions(U, L, T, dt, k::KSetup)
    Out = similar(U, size(U, 1), NDIAG)
    backend = get_backend(U)
    diagnostics_kernel!(backend, KERNEL_WORKGROUP)(Out, U, L, T, Float64(dt), k; ndrange=size(U, 1))
    sync(backend)
    O = Array(Out)
    return NamedTuple{(FUNCTION_NAMES..., BALANCE_NAMES...)}(Tuple(O[:, j] for j in 1:NDIAG))
end

"get_balance(U, L, T, dt, k): getLost and getBalance of every box (same kernel as get_functions)."
get_balance(U, L, T, dt, k::KSetup) = get_functions(U, L, T, dt, k)[BALANCE_NAMES]

"""
    get_rates(U, L, T, dt, k) -> NamedTuple of nb×nS arrays (getRates field names, units 1/day)
"""
function get_rates(U, L, T, dt, k::KSetup{D}) where {D}
    Rt = similar(U, size(U, 1), D[1], length(RATE_NAMES))
    backend = get_backend(U)
    rates_kernel!(backend, KERNEL_WORKGROUP)(Rt, U, L, T, Float64(dt), k; ndrange=size(U, 1))
    sync(backend)
    A = Array(Rt)
    return NamedTuple{RATE_NAMES}(Tuple(A[:, :, j] for j in 1:length(RATE_NAMES)))
end

"""
    simulate_euler_functions!(U, Fn, L, T, tEnd, dt, k) -> (U, Fn)

Fortran simulateEulerFunctions for every box: floor(tEnd/dt) Euler steps, then getFunctions
from the rates of the last substep and the final state; Fn is nb×9 (FUNCTION_NAMES order).
"""
function simulate_euler_functions!(U, Fn, L, T, tEnd, dt, k::KSetup)
    backend = get_backend(U)
    euler_functions_kernel!(backend, KERNEL_WORKGROUP)(U, Fn, L, T, Float64(dt), floor(Int, tEnd / dt), k;
                                                       ndrange=size(U, 1))
    sync(backend)
    return U, Fn
end

# convenience methods taking the host setup
get_functions(U, L, T, dt, s::NUMSetup) = get_functions(U, L, T, dt, kernel_setup(s))
get_balance(U, L, T, dt, s::NUMSetup) = get_balance(U, L, T, dt, kernel_setup(s))
get_rates(U, L, T, dt, s::NUMSetup) = get_rates(U, L, T, dt, kernel_setup(s))
