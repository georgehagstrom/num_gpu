# Per-box biology: calcDerivatives of NUMmodel.f90 for one box, for any setup (generalists,
# simple generalists, diatoms, simple diatoms, active/passive copepods, POM; 2 or 3 nutrients).
# One thread (GPU) or loop iteration (CPU) per box; `euler_kernel!` does all Euler substeps of a
# transport step in one launch with the box state in local memory. The order of operations
# follows the Fortran (groups in setup order, sequential accumulation into the nutrient
# derivatives), so results agree with the library to round-off.
#
# Deliberately reproduced upstream behaviour:
#  * Generalists (both kinds) overwrite JF with JFreal; the overwritten value is used for
#    predation mortality and as the feeding input of the corrected (second) unicellular pass.
#  * Copepods multiply mortHTL by fTemp2 in place; the POM and N terms use that value.
#  * Simple diatoms: remineralized mortality is subtracted from N, and mortpred is added to
#    dSi/dt without multiplying by biomass.
#
# Use: k = kernel_setup(s) (or adapt(CuArray, kernel_setup(s))); pass k to calc_derivatives,
# simulate_euler!, transport_step!, simulate_global and the diagnostics.

using Adapt
using KernelAbstractions
using StaticArrays

const T_GENS = Int(generalist_simple); const T_GEN = Int(generalist)
const T_DIA = Int(diatom); const T_DIAS = Int(diatom_simple)
const T_COPA = Int(copepod_active); const T_COPP = Int(copepod_passive); const T_POM = Int(pom)

"""
Setup constants in a form usable inside kernels. `D = (nS, nNutrients, nPOM, nCmax)` are
compile-time sizes; `groups` is a tuple of (type, first class, n) in setup order.
"""
struct KSetup{D,V<:AbstractVector{Float64},M<:AbstractMatrix{Float64},NG}
    dims::Val{D}
    rhoCN::Float64
    fracHTL_to_N::Float64
    quadraticHTL::Bool
    m::V; z::V; AF::V; JFmax::V; epsilonF::V; mort2constant::V; pHTL::V
    AN::V; AL::V; Jmax::V; Jresp::V; JlossPassive::V; JrespFactor::V; isPOM::V
    theta::M
    thetaPOM::M
    gen::NamedTuple{(:epsilonL, :bL, :bN, :bDOC, :bF, :bg, :remin2, :reminF),NTuple{8,Float64}}
    gens::NamedTuple{(:epsilonL, :remin2, :reminF),NTuple{3,Float64}}
    dia::NamedTuple{(:epsilonL, :bL, :bN, :bDOC, :bSi, :bg, :remin2, :rhoCSi),NTuple{8,Float64}}
    dias::NamedTuple{(:epsilonL, :bN, :bSi, :remin2, :rhoCSi),NTuple{5,Float64}}
    cop::NamedTuple{(:epsilonR, :kBasal, :kSDA),NTuple{3,Float64}}
    pomremin::Float64
    iP::Int                          # first POM class (0: no POM)
    groups::NTuple{NG,Tuple{Int,Int,Int}}
end

Adapt.@adapt_structure KSetup

function kernel_setup(s::NUMSetup)
    cops = [gr.ix for gr in s.groups if is_copepod(gr.type)]
    nC = isempty(cops) ? 2 : maximum(length.(cops))
    D = (s.nS, s.nNutrients, max(length(s.ixPOM), 1), nC)
    v(x) = collect(Float64, vec(x))
    groups = Tuple((Int(gr.type), first(gr.ix), length(gr.ix)) for gr in s.groups)
    return KSetup(Val(D), s.rhoCN, s.fracHTL_to_N, s.bQuadraticHTL,
        v(s.m), v(s.z), v(s.AF), v(s.JFmax), v(s.epsilonF), v(s.mort2constant), v(s.pHTL),
        v(s.AN), v(s.AL), v(s.Jmax), v(s.Jresp), v(s.JlossPassive), v(s.JrespFactor), v(s.isPOM),
        Matrix(s.theta), Matrix(s.thetaPOM), s.gen, s.gens, s.dia, s.dias, s.cop, s.pomremin,
        isempty(s.ixPOM) ? 0 : first(s.ixPOM), groups)
end

# ---- rate recorder: per-class values as held by the Fortran group objects after calcDerivatives
const R_JN, R_JDOC, R_JL, R_JSI, R_FLVL, R_JF, R_F, R_JTOT, R_JRESPTOT, R_JNLOSS, R_JLREAL,
      R_JDOCREAL, R_CLOSSPHOTO, R_JPOM, R_MORTPRED, R_MORTHTL, R_MORT2 = 1:17
const NRATES = 17
@inline rec!(::Nothing, i, f, x) = nothing
@inline rec!(R, i, f, x) = (@inbounds R[i, f] = x; nothing)

@inline mortHTL_base(k, i, ub) = k.quadraticHTL ? k.pHTL[i] * ub : k.pHTL[i]
@inline is_cop(t) = (t == T_COPA) | (t == T_COPP)

# ---- one unicellular pass (calcDerivativesUnicellulars): rates of all unicellular groups,
# predation mortality of all classes, derivatives of the unicellular groups. JF holds the
# current feeding (generalists overwrite theirs with JFreal). Returns (dN, dDOC, dSi, npp).
@inline function unicellular_pass!(du, mortpred, JF, Jtot, jPOM, u, F, L, fT2, fT15,
                                   gN, gDOC, gSi, k::KSetup{D}, R) where {D}
    nS, nN = D[1], D[2]
    uN = max(u[1], 0.0); uDOC = max(u[2], 0.0); uSi = nN > 2 ? max(u[3], 0.0) : 0.0
    # contributions of each class to the nutrient derivatives (added after mortpred, in class order)
    cN = MVector{nS,Float64}(undef); cDOC = MVector{nS,Float64}(undef); cSi = MVector{nS,Float64}(undef)
    npp = 0.0
    @inbounds for (t, i0, n) in k.groups
        if t == T_GEN
            p = k.gen
            for i in i0:i0+n-1
                AN = k.AN[i]; Jresp = k.Jresp[i]; JlossPassive = k.JlossPassive[i]; m = k.m[i]
                JFi = JF[i]
                JN = gN * fT15 * AN * uN * k.rhoCN
                JDOC = gDOC * fT15 * AN * uDOC
                JL = p.epsilonL * k.AL[i] * L
                JmaxT = fT2 * k.Jmax[i]
                Jnetp = JL * (1 - p.bL) + JDOC * (1 - p.bDOC) + JFi * (1 - p.bF) - fT2 * Jresp
                if Jnetp < 0
                    Jnet = Jnetp; dN = 0.0
                else
                    dN = JN == 0 ? 1.0 : max(0.0, min(1.0, (Jnetp - JFi * (p.bg + 1)) / (JN * (1 + p.bg + p.bN))))
                    Jnet = min((Jnetp - p.bN * (dN * JN)) / (1 + p.bg), JFi + dN * JN)
                end
                f = 0.0
                if Jnet > JlossPassive
                    f = Jnet / (Jnet + JmaxT); Jnet = JmaxT * f
                end
                Jt = Jnet - JlossPassive
                JNreal = max(0.0, Jnet - JFi)
                JFreal = min(JFi, (Jnet + p.bg * max(0.0, Jnet) + fT2 * Jresp + p.bN * JNreal) / (1 - p.bF))
                tmp = (1 - p.bDOC) * JDOC + (1 - p.bL) * JL
                if tmp == 0.0
                    JDOCreal = 0.0; JLreal = 0.0
                else
                    tmp = (Jnet + p.bg * max(0.0, Jnet) + p.bN * JNreal + fT2 * Jresp - JFreal * (1 - p.bF)) / tmp
                    JDOCreal = tmp * JDOC; JLreal = tmp * JL
                end
                JNlossLiebig = max(0.0, JNreal + JFreal - JlossPassive - Jt)
                JCloss_feeding = (1.0 - k.epsilonF[i]) / k.epsilonF[i] * JFreal
                JCloss_photouptake = (1.0 - p.epsilonL) / p.epsilonL * JLreal
                Jresptot = fT2 * Jresp + p.bDOC * JDOCreal + p.bL * JLreal + p.bN * JNreal + p.bF * JFreal +
                           max(0.0, p.bg * Jnet)
                ub = max(u[nN+i], 0.0)
                mort2 = k.mort2constant[i] * ub
                jPOM[i] = (1 - p.remin2) * mort2 + (1 - p.reminF) * JCloss_feeding / m
                cN[i] = ((-JNreal + JlossPassive + JNlossLiebig + p.reminF * JCloss_feeding) / m +
                         p.remin2 * mort2) * ub / k.rhoCN
                cDOC[i] = ((-JDOCreal + JlossPassive + JCloss_photouptake + p.reminF * JCloss_feeding) / m +
                           p.remin2 * mort2) * ub
                Jtot[i] = Jt; JF[i] = JFreal
                npp += max(0.0, (JLreal / p.epsilonL - Jresptot) * u[nN+i] / m)
                rec!(R, i, R_JN, JNreal); rec!(R, i, R_JDOC, JDOCreal); rec!(R, i, R_JL, JL)
                rec!(R, i, R_F, f); rec!(R, i, R_JRESPTOT, Jresptot); rec!(R, i, R_JNLOSS, JNlossLiebig)
                rec!(R, i, R_JLREAL, JLreal); rec!(R, i, R_JDOCREAL, JDOCreal)
                rec!(R, i, R_CLOSSPHOTO, JCloss_photouptake); rec!(R, i, R_MORT2, mort2)
            end
        elseif t == T_GENS
            p = k.gens
            for i in i0:i0+n-1
                AN = k.AN[i]; Jresp = k.Jresp[i]; JlossPassive = k.JlossPassive[i]; m = k.m[i]
                JFi = JF[i]
                JN = gN * fT15 * AN * uN * k.rhoCN
                JDOC = gDOC * fT15 * AN * uDOC
                JL = p.epsilonL * k.AL[i] * L
                JNtot = JN + JFi - JlossPassive
                JCtot = JL + JFi + JDOC - fT2 * Jresp - JlossPassive
                Jt = min(JNtot, JCtot)
                JmaxT = fT2 * k.Jmax[i]
                if Jt > 0
                    f = Jt / (Jt + max(0.0, JmaxT))
                    JFreal = max(0.0, min(JmaxT, JFi - (Jt - f * JmaxT)))
                    Jt = f * JmaxT
                else
                    JFreal = max(0.0, JFi)
                end
                JLreal = JL - max(0.0, min((JCtot - (JFi - JFreal) - Jt), JL))
                JCtot = +JLreal + JDOC + JFreal - fT2 * Jresp - JlossPassive
                JNtot = JN + JFreal - JlossPassive
                JCloss_feeding = (1.0 - k.epsilonF[i]) / k.epsilonF[i] * JFreal
                JCloss_photouptake = (1.0 - p.epsilonL) / p.epsilonL * JLreal
                JNlossLiebig = max(0.0, JNtot - Jt)
                JClossLiebig = max(0.0, JCtot - Jt)
                JNloss = JCloss_feeding + JNlossLiebig + JlossPassive
                Jresptot = fT2 * Jresp
                JDOCreal = JDOC - JClossLiebig
                ub = max(u[nN+i], 0.0)
                mort2 = k.mort2constant[i] * ub
                jPOM[i] = (1 - p.remin2) * mort2
                cN[i] = ((-JN + JlossPassive + JNlossLiebig + JCloss_feeding) / m + p.remin2 * mort2) * ub / k.rhoCN
                cDOC[i] = ((-JDOC + JlossPassive + JClossLiebig + JCloss_photouptake + p.reminF * JCloss_feeding) / m +
                           p.remin2 * mort2) * ub
                Jtot[i] = Jt; JF[i] = JFreal
                npp += max(0.0, (JLreal / p.epsilonL - Jresptot) * u[nN+i] / m)
                rec!(R, i, R_JN, JN); rec!(R, i, R_JDOC, JDOC); rec!(R, i, R_JL, JL)
                rec!(R, i, R_JRESPTOT, Jresptot); rec!(R, i, R_JNLOSS, JNloss); rec!(R, i, R_JLREAL, JLreal)
                rec!(R, i, R_JDOCREAL, JDOCreal); rec!(R, i, R_CLOSSPHOTO, JCloss_photouptake)
                rec!(R, i, R_MORT2, mort2)
            end
        elseif t == T_DIA
            q = k.dia
            for i in i0:i0+n-1
                AN = k.AN[i]; Jresp = k.Jresp[i]; JlossPassive = k.JlossPassive[i]; m = k.m[i]
                JN = fT15 * gN * AN * uN * k.rhoCN
                JDOC = fT15 * gDOC * AN * uDOC
                JSi = fT15 * gSi * AN * uSi * q.rhoCSi
                JL = q.epsilonL * k.AL[i] * L
                JmaxT = fT2 * k.Jmax[i]
                Jnetp = JL * (1 - q.bL) + JDOC * (1 - q.bDOC) - fT2 * Jresp
                tmp = Jnetp / (1.0 + q.bg + q.bN + q.bSi)
                dN = JN == 0.0 ? 1.0 : min(1.0, tmp / JN)
                dSi = JSi == 0.0 ? 1.0 : min(1.0, tmp / JSi)
                Jnet = 1.0 / (1 + q.bg) * (JDOC * (1 - q.bDOC) + JL * (1 - q.bL) -
                                           fT2 * Jresp - q.bN * dN * JN - q.bSi * dSi * JSi)
                if Jnetp < 0.0
                    Jnet = -fT2 * Jresp + JDOC * (1 - q.bDOC) + JL * (1 - q.bL)
                elseif (dSi >= 1.0) | (dN >= 1.0)
                    Jnet = JSi > JN ? JN : JSi
                end
                f = 0.0
                if Jnet > JlossPassive
                    f = Jnet / (Jnet + JmaxT); Jnet = JmaxT * f
                end
                Jt = Jnet - JlossPassive
                JNreal = max(0.0, Jnet); JSireal = max(0.0, Jnet)
                den = (1 - q.bDOC) * JDOC + (1 - q.bL) * JL
                if den == 0.0
                    JDOCreal = 0.0; JLreal = 0.0
                else
                    tmp2 = (Jnet + q.bg * max(0.0, Jnet) + q.bN * JNreal + q.bSi * JSireal + fT2 * Jresp) / den
                    JDOCreal = tmp2 * JDOC; JLreal = tmp2 * JL
                end
                JNlossLiebig = max(0.0, -Jnet)
                JCloss_photouptake = (1.0 - q.epsilonL) / q.epsilonL * JLreal
                Jresptot = fT2 * Jresp + q.bDOC * JDOCreal + q.bL * JLreal + q.bN * JNreal + q.bSi * JSireal +
                           max(0.0, q.bg * Jnet)
                ub = max(u[nN+i], 0.0)
                mort2 = k.mort2constant[i] * ub
                jPOM[i] = (1 - q.remin2) * mort2
                cN[i] = ((-JNreal + JlossPassive + JNlossLiebig) / m + q.remin2 * mort2) * ub / k.rhoCN
                cDOC[i] = ((-JDOCreal + JlossPassive + JCloss_photouptake) / m + q.remin2 * mort2) * ub
                cSi[i] = ((-JSireal + JlossPassive + JNlossLiebig) / m + q.remin2 * mort2) * ub / q.rhoCSi
                Jtot[i] = Jt
                npp += max(0.0, (JLreal / q.epsilonL - Jresptot) * u[nN+i] / m)
                rec!(R, i, R_JN, JNreal); rec!(R, i, R_JDOC, JDOCreal); rec!(R, i, R_JL, JL)
                rec!(R, i, R_JSI, JSireal); rec!(R, i, R_F, f); rec!(R, i, R_JRESPTOT, Jresptot)
                rec!(R, i, R_JNLOSS, JNlossLiebig); rec!(R, i, R_JLREAL, JLreal)
                rec!(R, i, R_JDOCREAL, JDOCreal); rec!(R, i, R_CLOSSPHOTO, JCloss_photouptake)
                rec!(R, i, R_MORT2, mort2)
            end
        elseif t == T_DIAS
            q = k.dias
            for i in i0:i0+n-1
                AN = k.AN[i]; Jresp = k.Jresp[i]; m = k.m[i]
                JN = fT15 * gN * AN * uN * k.rhoCN
                JSi = fT15 * gSi * AN * uSi * q.rhoCSi
                JL = q.epsilonL * k.AL[i] * L
                JmaxT = max(0.0, fT2 * k.Jmax[i])
                Jt = min(JmaxT, JN, JL - fT2 * Jresp, JSi)
                Jt = min(Jt, JL - fT2 * Jresp - (q.bN / k.rhoCN + q.bSi / q.rhoCSi) * Jt)
                if Jt > 0.0
                    Jt = JmaxT * Jt / (Jt + JmaxT)
                end
                Jresptot = fT2 * Jresp
                ub = max(u[nN+i], 0.0)
                mort2 = k.mort2constant[i] * ub
                jPOM[i] = (1 - q.remin2) * mort2
                mortloss = q.remin2 * mort2 * ub
                cN[i] = -((Jt * ub / m + mortloss) / k.rhoCN)
                cDOC[i] = 0.0
                cSi[i] = mortloss                  # dSi is completed after mortpred (see below)
                Jtot[i] = Jt
                npp += max(0.0, (JL / q.epsilonL - Jresptot) * u[nN+i] / m)
                rec!(R, i, R_JN, JN); rec!(R, i, R_JL, JL); rec!(R, i, R_JSI, JSi)
                rec!(R, i, R_JRESPTOT, Jresptot); rec!(R, i, R_JLREAL, JL); rec!(R, i, R_MORT2, mort2)
            end
        end
    end
    # predation mortality of all classes: mortpred(i) = sum_j theta(j,i) JF_j u_j / (epsF_j m_j F_j)
    @inbounds for i in 1:nS
        mortpred[i] = 0.0
    end
    @inbounds for j in 1:nS
        W = F[j] > 0.0 ? JF[j] * max(u[nN+j], 0.0) / (k.epsilonF[j] * k.m[j] * F[j]) : 0.0
        for i in 1:nS
            mortpred[i] += k.theta[j, i] * W
        end
    end
    # assemble the derivatives of the unicellular groups (nutrients accumulate in class order)
    dN = 0.0; dDOC = 0.0; dSi = 0.0
    @inbounds for (t, i0, n) in k.groups
        ((t == T_GEN) | (t == T_GENS) | (t == T_DIA) | (t == T_DIAS)) || continue
        for i in i0:i0+n-1
            ub = max(u[nN+i], 0.0)
            mort2 = k.mort2constant[i] * ub
            dN += cN[i]
            if t == T_DIAS
                dSi += ((-Jtot[i]) * ub / k.m[i] + cSi[i] + mortpred[i]) / k.dias.rhoCSi
            else
                dDOC += cDOC[i]
                t == T_DIA && (dSi += cSi[i])
            end
            du[nN+i] = (Jtot[i] / k.m[i] - mortpred[i] - mort2 - mortHTL_base(k, i, ub)) * ub
            rec!(R, i, R_JPOM, jPOM[i])
        end
    end
    return (dN, dDOC, dSi, npp)
end

"""
    box_derivs!(du, u, L, T, dt, k::KSetup, R=nothing) -> npp

calcDerivatives for one box (u, du: local vectors of length nGrid). Returns the net primary
production (mgC/m3/day, Fortran getProdNet) from the rates of that evaluation. If `R` is an
nS×NRATES matrix, the per-class rates are recorded in it (see diagnostics.jl).
"""
@inline function box_derivs!(du, u, L, T, dt, k::KSetup{D}, R=nothing) where {D}
    nS, nN, nP, nC = D
    fT2 = 2.0^((T - TREF) / 10.0)
    fT15 = 1.5^((T - TREF) / 10.0)
    if R !== nothing
        @inbounds for f in 1:NRATES, i in 1:nS
            R[i, f] = 0.0
        end
    end
    F = MVector{nS,Float64}(undef)
    JF = MVector{nS,Float64}(undef)
    mortpred = MVector{nS,Float64}(undef)
    Jtot = MVector{nS,Float64}(undef)
    jPOM = MVector{nS,Float64}(undef)
    @inbounds for i in 1:nS
        acc = 0.0
        for j in 1:nS
            acc += k.theta[i, j] * max(u[nN+j], 0.0)
        end
        F[i] = acc
        flvl = k.epsilonF[i] * k.AF[i] * acc / ((k.AF[i] * acc + EPS) + fT2 * k.JFmax[i])
        JF[i] = flvl * fT2 * k.JFmax[i]
        Jtot[i] = 0.0; jPOM[i] = 0.0; du[nN+i] = 0.0
        rec!(R, i, R_FLVL, flvl)
    end
    Lf = Float64(L)
    dN, dDOC, dSi, npp = unicellular_pass!(du, mortpred, JF, Jtot, jPOM, u, F, Lf, fT2, fT15,
                                           1.0, 1.0, 1.0, k, R)
    # correction if a nutrient would become negative (uses u, not upositive)
    gN = u[1] + dN * dt < 0 ? max(0.0, min(1.0, -u[1] / (dN * dt))) : 1.0
    gDOC = u[2] + dDOC * dt < 0 ? max(0.0, min(1.0, -u[2] / (dDOC * dt))) : 1.0
    gSi = (nN > 2 && u[3] + dSi * dt < 0) ? max(0.0, min(1.0, -u[3] / (dSi * dt))) : 1.0
    if (gN < 1.0) | (gDOC < 1.0) | (gSi < 1.0)
        dN, dDOC, dSi, npp = unicellular_pass!(du, mortpred, JF, Jtot, jPOM, u, F, Lf, fT2, fT15,
                                               gN, gDOC, gSi, k, R)
    end

    # ---- copepods (calcDerivativesCopepod)
    c = k.cop
    g = MVector{nC,Float64}(undef); mort = MVector{nC,Float64}(undef); gam = MVector{nC,Float64}(undef)
    @inbounds for (t, i0, n) in k.groups
        is_cop(t) || continue
        sJ = 0.0
        for kk in 1:n
            i = i0 + kk - 1
            m = k.m[i]; ub = max(u[nN+i], 0.0)
            Jresptot = k.JrespFactor[i] * c.kBasal * fT2 + c.kSDA * JF[i]
            nu = JF[i] - Jresptot
            g[kk] = max(0.0, nu) / m
            mortStarve = -min(0.0, nu) / m
            mHTL = mortHTL_base(k, i, ub) * fT2
            mort[kk] = mortpred[i] + mortStarve + mHTL
            # the exponent is guarded so the discarded g == 0 branch is not z^(-Inf), which Julia
            # evaluates as an integer power by repeated squaring (slow)
            gam[kk] = g[kk] != 0.0 ? (g[kk] - mort[kk]) / (1 - k.z[i]^(g[kk] != 0.0 ? 1 - mort[kk] / g[kk] : 0.0)) : 0.0
            Jtot[i] = nu
            sJ += Jresptot * ub / m
            rec!(R, i, R_JRESPTOT, Jresptot); rec!(R, i, R_MORTHTL, mHTL)
        end
        iN = i0 + n - 1
        uN_ = max(u[nN+iN], 0.0)
        du[nN+i0] = c.epsilonR * g[n] * uN_ + (g[1] - gam[1] - mort[1]) * max(u[nN+i0], 0.0)
        for kk in 2:n-1
            i = i0 + kk - 1
            du[nN+i] = gam[kk-1] * max(u[nN+i-1], 0.0) + (g[kk] - gam[kk] - mort[kk]) * max(u[nN+i], 0.0)
        end
        du[nN+iN] = gam[n-1] * max(u[nN+iN-1], 0.0) - mort[n] * uN_
        for kk in 1:n
            i = i0 + kk - 1
            epsF = k.epsilonF[i]
            jPOM[i] = (1 - epsF) * JF[i] / (k.m[i] * epsF)
            kk == n && (jPOM[i] += (1.0 - c.epsilonR) * g[n])
            rec!(R, i, R_JPOM, jPOM[i])
        end
        dN = dN + sJ / k.rhoCN
    end

    # ---- POM: routing of jPOM and of a fraction of HTL mortality (class order), then POM itself
    if k.iP != 0
        @inbounds for (t, i0, n) in k.groups
            t == T_POM && continue
            isc = is_cop(t)
            for i in i0:i0+n-1
                ub = max(u[nN+i], 0.0)
                for j in 1:nP
                    k.thetaPOM[i, j] != 0.0 && (du[nN+k.iP+j-1] += jPOM[i] * ub)
                end
                mHTL = isc ? mortHTL_base(k, i, ub) * fT2 : mortHTL_base(k, i, ub)
                du[nN+k.iP+nP-1] += (1 - k.fracHTL_to_N) * ub * mHTL
            end
        end
        sP = 0.0
        @inbounds for j in 1:nP
            i = k.iP + j - 1
            uP = max(u[nN+i], 0.0)
            du[nN+i] = du[nN+i] - fT2 * k.pomremin * uP - mortpred[i] * uP
            sP += uP
            rec!(R, i, R_JRESPTOT, fT2 * k.pomremin * k.m[i])
        end
        dN = dN + fT2 * k.pomremin * sP / k.rhoCN
    end
    # ---- some HTL mortality ends up as nutrients (per group)
    @inbounds for (t, i0, n) in k.groups
        t == T_POM && continue
        isc = is_cop(t)
        sH = 0.0
        for i in i0:i0+n-1
            ub = max(u[nN+i], 0.0)
            mHTL = isc ? mortHTL_base(k, i, ub) * fT2 : mortHTL_base(k, i, ub)
            sH += ub * mHTL
            isc || rec!(R, i, R_MORTHTL, mHTL)
        end
        dN = dN + k.fracHTL_to_N * sH / k.rhoCN
    end
    if R !== nothing
        @inbounds for i in 1:nS
            rec!(R, i, R_JF, JF[i]); rec!(R, i, R_JTOT, Jtot[i]); rec!(R, i, R_MORTPRED, mortpred[i])
        end
    end
    du[1] = dN; du[2] = dDOC
    nN > 2 && (du[3] = dSi)
    return npp
end

# ---------------------------------------------------------------- kernels and host functions

@inline function load_box(U, b, ::Val{nGrid}) where {nGrid}
    u = MVector{nGrid,Float64}(undef)
    @inbounds for j in 1:nGrid
        u[j] = U[b, j]
    end
    return u
end

@kernel inbounds = true function derivs_kernel!(dU, @Const(U), @Const(L), @Const(T), dt, k)
    b = @index(Global)
    box_derivs_global!(dU, U, L, T, dt, k, b)
end
@inline function box_derivs_global!(dU, U, L, T, dt, k::KSetup{D}, b) where {D}
    nGrid = D[1] + D[2]
    u = load_box(U, b, Val(nGrid)); du = MVector{nGrid,Float64}(undef)
    box_derivs!(du, u, L[b], T[b], dt, k)
    @inbounds for j in 1:nGrid
        dU[b, j] = du[j]
    end
end

@kernel inbounds = true function euler_kernel!(U, @Const(L), @Const(T), dt, nsub, k)
    b = @index(Global)
    box_euler!(U, L, T, dt, nsub, k, b)
end
@inline function box_euler!(U, L, T, dt, nsub, k::KSetup{D}, b) where {D}
    nGrid = D[1] + D[2]
    u = load_box(U, b, Val(nGrid)); du = MVector{nGrid,Float64}(undef)
    Lb = L[b]; Tb = T[b]
    for _ in 1:nsub
        box_derivs!(du, u, Lb, Tb, dt, k)
        @inbounds for j in 1:nGrid
            u[j] = u[j] + du[j] * dt
        end
    end
    @inbounds for j in 1:nGrid
        U[b, j] = u[j]
    end
end

@kernel inbounds = true function npp_kernel!(NPP, @Const(U), @Const(L), @Const(T), dt, k)
    b = @index(Global)
    box_npp_global!(NPP, U, L, T, dt, k, b)
end
@inline function box_npp_global!(NPP, U, L, T, dt, k::KSetup{D}, b) where {D}
    nGrid = D[1] + D[2]
    u = load_box(U, b, Val(nGrid)); du = MVector{nGrid,Float64}(undef)
    NPP[b] = box_derivs!(du, u, L[b], T[b], dt, k)
end

const KERNEL_WORKGROUP = 128

# (JLArrays' test backend has no synchronize method)
sync(backend) = applicable(KernelAbstractions.synchronize, backend) && KernelAbstractions.synchronize(backend)

"""
    calc_derivatives(U, L, T, dt, k::KSetup) -> dU

Fortran calcDerivatives for every row (box) of U (nb×nGrid); L, T are length-nb vectors.
"""
function calc_derivatives(U::AbstractMatrix, L::AbstractVector, T::AbstractVector, dt::Real, k::KSetup)
    dU = similar(U)
    backend = get_backend(U)
    derivs_kernel!(backend, KERNEL_WORKGROUP)(dU, U, L, T, Float64(dt), k; ndrange=size(U, 1))
    sync(backend)
    return dU
end
calc_derivatives(U::AbstractMatrix, L::AbstractVector, T::AbstractVector, dt::Real, s::NUMSetup) =
    calc_derivatives(U, L, T, dt, kernel_setup(s))

"""
    net_primary_production!(NPP, U, L, T, dt, k) -> NPP

Net primary production of every box (mgC/m3/day), as Fortran getFunctions/getProdNet
(unicellular groups: max(0, JLreal/epsilonL - Jresptot) * u / m, i.e. photosynthesis including
the part exuded as DOC), from the rates at state U.
"""
function net_primary_production!(NPP, U, L, T, dt, k::KSetup)
    backend = get_backend(U)
    npp_kernel!(backend, KERNEL_WORKGROUP)(NPP, U, L, T, Float64(dt), k; ndrange=size(U, 1))
    sync(backend)
    return NPP
end

"""
    simulate_euler!(U, L, T, tEnd, dt, k::KSetup)

Fortran simulateEuler for all boxes: floor(tEnd/dt) Euler steps, all in one kernel launch.
"""
function simulate_euler!(U, L, T, tEnd, dt, k::KSetup)
    backend = get_backend(U)
    euler_kernel!(backend, KERNEL_WORKGROUP)(U, L, T, Float64(dt), floor(Int, tEnd / dt), k;
                                             ndrange=size(U, 1))
    sync(backend)
    return U
end
simulate_euler!(U, L, T, tEnd, dt, s::NUMSetup) = simulate_euler!(U, L, T, tEnd, dt, kernel_setup(s))
