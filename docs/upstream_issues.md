# Issues found in NUMmodel v1.0 while porting it

Line numbers refer to the v1.0 tag. The port now follows upstream **Develop (5644bf5, 2026-09-25)**,
where several of these are fixed by the upstream authors; the status of each is given below
(**fixed in Develop** — the port follows the fix; otherwise the port still reproduces the
behaviour so that results match, marked in the Julia source).

| # | Status in Develop 5644bf5 |
|---|---|
| 1 gammaDOC sign | fixed (924c2c9) |
| 2, 3 simple diatoms N/Si terms | not changed |
| 4 shared copepod parameters | not changed |
| 5 JCloss_photouptake of simple diatoms | not changed (port uses 0) |
| 6 getRates double temperature factor | not changed |
| 7 calcGrid mLower(2) | not changed |
| 8 spec%type never set | fixed (ef400cf) |
| 9 chemostat Euler array shape | fixed (e93a08a: only unicellular groups mixed out, each with its own biomass) |
| 10 simulateChemostatEuler simulates twice | fixed (e93a08a) |
| 11 parametersChemostat ignores constantValues | not changed |
| 12–20 | not changed |
| 21 TM positivity conversion loses mass | fixed (40ea16d, Ken Andersen, 2026-09-18) — conserves mass; see note there |

## Model equations (Fortran)

1. **Sign of the DOC uptake correction** — `Fortran/NUMmodel.f90:609`:
   `gammaDOC = max(0, min(1, u(idxDOC)/(dudt(idxDOC)*dt)))`; N and Si use `-u/(dudt*dt)`. When
   DOC would become negative, DOC uptake is switched off entirely instead of scaled down.
2. **Simple diatoms: remineralized mortality leaves the N pool** — `Fortran/diatoms_simple.f90:141-143`:
   `dNdt = dNdt - ((Jtot*u/m + mortloss)/rhoCN)`; remineralized mortality should add to N.
3. **Simple diatoms: dimension error in dSi/dt** — `Fortran/diatoms_simple.f90:157`: `mortpred`
   (1/day) is added to terms in ugC/l/day without multiplying by biomass.
4. **Copepod parameters shared between groups** — `Fortran/copepods.f90:14`: `epsilonR`,
   `kBasal`, `kSDA` are module variables, so the values of the last copepod group read (active
   copepods in setupNUMmodel) apply to all copepod groups. Harmless while both input sections
   hold the same values.

## Diagnostics (Fortran)

5. **Undefined memory in getLost for simple diatoms** — `JCloss_photouptake` is allocated
   (`spectrum.f90:251`) but never set for simple diatoms, and getLost/getClostUnicellular
   (`spectrum.f90:271`) reads it. Observed: `Clost = -6.5e196` in setupDiatoms_simpleOnly.
   The port uses 0.
6. **getRates applies the temperature factor twice** — `Fortran/NUMmodel.f90:1235-1236`:
   `jN = fTemp15*JN/m`, `jDOC = fTemp15*JDOC/m`, but `JN` and `JDOC` already include `fTemp15`.
7. **Out-of-bounds read for single-class groups** — `Fortran/spectrum.f90:168`: `calcGrid` reads
   `mLower(2)` also when n = 1 (every setup with one POM class). The value is not used.

## Chemostat

8. **Copepods are not exempted from chemostat losses** — `Fortran/NUMmodel.f90:794`: the test
   `group(iGroup)%spec%type .ne. typeCopepodActive` uses a field that is never assigned anywhere,
   so all groups (including copepods) are flushed out.
9. **Array-shape mismatch in the same loss** — `NUMmodel.f90:795`:
   `dudt(ixStart:ixEnd) += diff*(0 - u(idxB:nGrid))` assigns an nGrid-idxB+1 array to a group
   section; with gfortran's default (no bounds check) each group loses at the rate of the first
   ng biomass classes of the whole spectrum.
10. **simulateChemostatEuler simulates twice** — `matlab/simulateChemostatEuler.m:58`: after
    `f_simulatechemostateuler` for tEnd days it calls `f_simulateeulerfunctions` for another tEnd
    days without the chemostat (to get the functions).
11. **parametersChemostat ignores `constantValues`** — `matlab/parametersChemostat.m:33, 84`:
    d = 0.5 and L = 100 are hard-coded; the argument is not used.

## Setups (Matlab)

12. **setupGeneralistsSimplePOM fails with its defaults** — `matlab/setupGeneralistsSimplePOM.m:60`:
    `setSinkingPOM(p, 11)` passes a scalar, but setSinkingPOM requires one value per POM class
    (default nPOM = 10). The port spreads the scalar over the classes.
13. **setupNUMmodelSimple labels copepods as passive** — `matlab/setupNUMmodelSimple.m:59`
    adds type 11 (passive) to `p`, while the Fortran setup adds active copepods (type 10).
    Only Matlab's bookkeeping (`p.typeGroups`) is affected.

## Drivers (Matlab)

14. **calcGlobalWatercolumn picks a neighbouring column** — `matlab/calcGlobalWatercolumn.m:27-28`:
    `mod(ix, nx)+1` and `floor(ix/nx)` should be `mod(ix-1, nx)+1` and `floor((ix-1)/nx)+1`;
    the column used is (x+1, y-1) of the nearest grid point (and y = 0 fails for the first row).
15. **Annual HTL averages are empty** — `matlab/simulateGlobal.m:245`: `getMortHTL` returns the
    groups' `mortHTL`, which setHTL never sets for "quadratic" HTL, so in a fresh session it is
    zero: `sim.BHTL` is all zero and `sim.mHTLAnnualMean` is NaN. The port uses the intended
    coefficient `pHTL` (a documented deviation).
16. **Monthly production divided by the number of steps per year** — `simulateGlobal.m:450`:
    `integrate_over_depth` divides by 365/dtTransport, which is right for the annual sums but
    also applied to the monthly means `sim.ProdNet`, `sim.ProdHTL`.
17. **sim.mHTL indexes steps, not months** — `simulateGlobal.m:425`: `mHTL(i,:)` is the value at
    time step i (i = 1..nSave), not the average over save period i.
18. **Hard-coded two steps per day** — `simulateGlobal.m:140` (`daily_insolation(0,Ybox,i/2,1)`)
    and `simulateWatercolumn.m:244-247` (`1+2*cumsum(mon)`) assume dtTransport = 0.5 d.
19. **Water-column light in single precision** — `simulateWatercolumn.m` multiplies the
    single-precision `parday` array directly, so its light field is single precision (the global
    driver converts to double first).

21. **Positivity conversion of the transport matrices does not conserve mass** —
    `matlab/function_convert_TM_positive.m`: a negative off-diagonal A[i,j] is moved to A[j,i] with
    opposite sign and the diagonals are fixed so that row and column sums are unchanged. Tracer mass
    needs the *volume-weighted* column sums (v'A) unchanged; they change by (v[i]-v[j])|A[i,j]| per
    move. With the official MITgcm 2.8° matrices (~1.4e6 negative Aexp entries) the
    volume integral of nitrate falls by 6-9e-6 per 0.5 d step, ~0.4-0.5%/yr, the dominant global N
    loss of long runs (-34% in 100 years; closing the POM bottom changes it by <1 point). The raw
    matrices conserve to ~1e-9 per step. The official MITgcm_2.8deg download has the same matrices
    (identical to 2e-15), so this is upstream behaviour. **Fixed in Develop** (40ea16d): the diagonal
    is set from the volume-weighted column sums, which conserves mass (port default,
    `TMconversion=:develop`). Row sums then change, so a uniform field is not kept exactly (1523 of
    52749 boxes change by >0.1% per 0.5 d step, max 1.8%, official matrices, month 1).
    `TMconversion=:volume` scales the moved entry by v[i]/v[j] instead, which keeps both
    (`test/tm_conversion.jl`); `:v1` reproduces v1.0.

## Numerical behaviour (not a bug)

20. **Round-off sensitivity** — in nutrient-depleted surface boxes (tropical and subtropical
    gyres, parts of the Southern Ocean) differences of 1e-15 grow to O(1) within 5-15 days, also
    with 10x smaller time steps. Runs agree statistically; box-by-box comparisons of long runs
    (Julia vs Matlab, or two Julia runs 1e-15 apart) differ there to the same extent.
