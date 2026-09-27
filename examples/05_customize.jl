# Changing the model: parameters, setups and the global driver's options.
# Run from the repository root:   julia --project=julia/NUMGPU examples/05_customize.jl
using NUMGPU

# --- setups: all 13 upstream setups exist, with the same arguments (snake_case names) ---
s1 = setup_generalists_diatoms()                     # Matlab: setupGeneralistsDiatoms
s2 = setup_generic([0.1, 10.0])                      # Matlab: setupGeneric([0.1 10]) (copepod adult masses)
s  = setup_num_model(mortHTL=0.01, velocityPOM=30.0) # Matlab: setupNUMmodel, then setHTL/setSinkingPOM
println("setupNUMmodel: ", s.nS, " size classes in ", length(s.groups), " groups; ",
        s.nNutrients, " nutrients (N, DOC, Si)")

# --- the same changes after the setup, as in Matlab ---
# Matlab setHTL(mortHTL, mHTL, bQuadratic, bDeclining, bCopepodsOnly); NOTE the Julia order:
# set_htl!(s, mHTL, mortHTL, quadratic, declining, copepodsOnly)
set_htl!(s, 1.0, 0.006, true, false, true)
set_sinking_pom!(s, fill(20.0, length(s.ixPOM)))     # Matlab: setSinkingPOM(p, 20)

# --- input parameters (input.yaml) are read at setup; use another file with a modified copy ---
#     s = setup_num_model(joinpath("my_inputs", "input.yaml"))

# --- global driver options: every field of GlobalParams (defaults = upstream Develop) ---
p = GlobalParams(
    tEnd = 365.0 * 10,        # days
    dt = 0.05,                # Euler step (default 0.1; 0.05 gives converged NPP, see README)
    tSave = 365 / 12,         # save interval (days)
    BC_POMclosed = true,      # POM does not sink out of the bottom cell
    kw = 0.1,                 # light attenuation by water (1/m)
    bUse_parday_light = true, # false: clear-sky insolation instead of the parday climatology
    TMconversion = :develop,  # removal of negative transport-matrix entries (see README)
)
for f in (:tEnd, :dt, :tSave, :BC_POMclosed, :kw, :bUse_parday_light, :TMconversion)
    println("  ", rpad(f, 18), getfield(p, f))
end
# then: g = load_global_model(tm_paths("MITgcm_2.8deg"), s; p); res = simulate_global(g, s)
