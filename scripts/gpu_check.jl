# GPU correctness checks and benchmarks. Run on a CUDA machine:
#   julia --project=julia/gpu scripts/gpu_check.jl [outdir]
# NUM_TMDIR overrides the transport-matrix directory. Everything printed is also written to
# <outdir>/gpu_check.log (default ./gpu_check_out).
using CUDA, NUMGPU, Adapt, Random, Printf, LinearAlgebra, SparseArrays, Statistics

const ROOT = NUMGPU.numroot()
const TMDIR = get(ENV, "NUM_TMDIR", joinpath(ROOT, "TMs", "MITgcm_2.8deg"))
const OUT = length(ARGS) >= 1 ? ARGS[1] : "gpu_check_out"
mkpath(OUT)
const LOG = open(joinpath(OUT, "gpu_check.log"), "w")
say(x...) = (s = string(x...); println(s); println(LOG, s); flush(stdout); flush(LOG))
include(joinpath(@__DIR__, "..", "julia", "NUMGPU", "test", "oracle_states.jl"))   # random_states

# per-element error, floored for near-cancellations (as in test/runtests.jl)
relerr(a, b, U) = maximum(abs.(a .- b) ./ (abs.(b) .+ abs.(U) .+ 1e-12 .* maximum(abs.(b); dims=2) .+ 1e-300))
"median wall time of f() over n runs, synchronizing the GPU"
function bench(f, n)
    f(); CUDA.synchronize()
    ts = [(t = time_ns(); f(); CUDA.synchronize(); (time_ns() - t) / 1e9) for _ in 1:n]
    return median(ts)
end
failures = String[]
check(name, ok) = (say(ok ? "  PASS " : "  FAIL ", name); ok || push!(failures, name))

# ---------------------------------------------------------------- device
CUDA.functional() || error("CUDA not functional: check the NVIDIA driver (nvidia-smi)")
dev = CUDA.device()
say("device: ", CUDA.name(dev), ", memory ", round(CUDA.totalmem(dev) / 2^30; digits=1), " GiB, ",
    "capability ", CUDA.capability(dev), ", driver ", CUDA.driver_version(), ", runtime ", CUDA.runtime_version())
say("julia threads: ", Threads.nthreads())

# FP64 FMA throughput (compute-bound kernel)
function fma_kernel!(x, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= length(x)
        v = x[i]
        for _ in 1:n
            v = fma(v, 0.999999, 1e-7); v = fma(v, 1.000001, -1e-7)
            v = fma(v, 0.999999, 1e-7); v = fma(v, 1.000001, -1e-7)
        end
        x[i] = v
    end
    return
end
let x = CUDA.rand(Float64, 2^22), n = 2000
    k() = @cuda threads=256 blocks=cld(length(x), 256) fma_kernel!(x, n)
    t = bench(k, 3)
    say(@sprintf("FP64 FMA throughput: %.2f TFLOPS (4 FMA x %d x %d elements)", 2 * 4 * n * length(x) / t / 1e12, n, length(x)))
end
let x = CUDA.rand(Float64, 2^27), y = similar(x)
    t = bench(() -> (y .= 2.0 .* x), 5)
    say(@sprintf("memory bandwidth (y .= 2x): %.0f GB/s", 2 * sizeof(x) / t / 1e9))
end

# ---------------------------------------------------------------- correctness
say("\n== correctness: GPU vs CPU (CPU path is verified against Fortran/Matlab) ==")
s = setup_num_model(joinpath(ROOT, "input", "input.yaml"))
# the CPU path is verified against the Fortran library and Matlab (julia/NUMGPU/test)
kc = kernel_setup(s)
U, L, T = random_states(MersenneTwister(1), s, 5000)
D = calc_derivatives(U, L, T, 0.1, kc)
Ue = simulate_euler!(copy(U), L, T, 0.5, 0.1, kc)
kd = adapt(CuArray, kernel_setup(s))
Dk = Array(calc_derivatives(CuArray(U), CuArray(L), CuArray(T), 0.1, kd))
e = relerr(Dk, D, U); say(@sprintf("derivatives, 5000 random states: max err %.2e", e))
check("derivatives < 1e-10", e < 1e-10 && all(isfinite, Dk))
Uek = Array(simulate_euler!(CuArray(U), CuArray(L), CuArray(T), 0.5, 0.1, kd))
e = maximum(abs.(Uek .- Ue) ./ (abs.(Ue) .+ 1e-12)); say(@sprintf("Euler 0.5 d: max rel err %.2e", e))
check("Euler < 1e-10", e < 1e-10)

t0 = time()
g = load_global_model(TMDIR, s)
say(@sprintf("loaded global model (%d boxes) in %.1f s", g.nb, time() - t0))
gd = adapt(CuArray, g)
say("transport matrix type on device: ", typeof(gd.Aexp[1]).name.name)
Uc = copy(g.U0); Ug = copy(gd.U0)
wc = similar(Uc); wg = similar(Ug); mc = Ref((0, 0)); mg = Ref((0, 0))
for i in 1:3
    NUMGPU.transport_step!(Uc, g, i, mc, kc, wc)
    NUMGPU.transport_step!(Ug, gd, i, mg, kd, wg)
    local e = maximum(abs.(Array(Ug) .- Uc) ./ (abs.(Uc) .+ 1e-12 .* maximum(abs.(Uc); dims=1)))
    say(@sprintf("global transport step %d: max rel err %.2e", i, e))
    check("transport step $i < 1e-10", e < 1e-10)
end

# every upstream setup compiles to its own kernel specialization: check each on the GPU
say("\n== all setups: GPU vs CPU derivatives and Euler step ==")
for (nm, sf) in (("generalists_simple_only", setup_generalists_simple_only), ("generalists_only", setup_generalists_only),
                 ("generalists_simple_pom", setup_generalists_simple_pom), ("generalists_pom", setup_generalists_pom),
                 ("diatoms_only", setup_diatoms_only), ("diatoms_simple_only", setup_diatoms_simple_only),
                 ("generalists_diatoms", setup_generalists_diatoms), ("generalists_diatoms_simple", setup_generalists_diatoms_simple),
                 ("generalists_simple_copepod", setup_generalists_simple_copepod), ("generic", () -> setup_generic([0.1, 10.0])),
                 ("num_model", setup_num_model), ("num_model_simple", setup_num_model_simple), ("gen_diat_cope", setup_gen_diat_cope))
    sx = sf(); kx = kernel_setup(sx); kxd = adapt(CuArray, kx)
    Ux, Lx, Tx = random_states(MersenneTwister(7), sx, 2000)
    Dc = calc_derivatives(Ux, Lx, Tx, 0.1, kx)
    Dg = Array(calc_derivatives(CuArray(Ux), CuArray(Lx), CuArray(Tx), 0.1, kxd))
    ed = relerr(Dg, Dc, Ux)
    Ec = simulate_euler!(copy(Ux), Lx, Tx, 0.5, 0.1, kx)
    Eg = Array(simulate_euler!(CuArray(Ux), CuArray(Lx), CuArray(Tx), 0.5, 0.1, kxd))
    ee = maximum(abs.(Eg .- Ec) ./ (abs.(Ec) .+ 1e-12))
    say(@sprintf("%-28s derivatives %.1e  Euler %.1e", nm, ed, ee))
    check("$nm on GPU", ed < 1e-10 && ee < 1e-10)
end

# ---------------------------------------------------------------- benchmarks
say("\n== benchmarks (2.8 deg, $(g.nb) boxes) ==")
Ub = copy(gd.U0); Lb = gd.L0[:, 1]; Tb = gd.Tmat[:, 1]
t_tm = bench(() -> (mul!(wg, gd.Aexp[1], Ub); mul!(Ub, gd.Aimp[1], wg)), 10)
say(@sprintf("transport Aimp*(Aexp*U):            %.2f ms", 1e3 * t_tm))
t_step = Dict{String,Float64}()
for (label, sx) in ("kernel" => kd,)
    t_der = bench(() -> calc_derivatives(Ub, Lb, Tb, 0.1, sx), 10)
    Ue_ = copy(gd.U0)
    t_eul = bench(() -> simulate_euler!(Ue_, Lb, Tb, 0.5, 0.1, sx), 5)
    ms = Ref((0, 0)); Ut = copy(gd.U0)
    t_step[label] = bench(() -> NUMGPU.transport_step!(Ut, gd, 1, ms, sx, wg), 10)
    say(@sprintf("[%s] biology substep %.2f ms; Euler 0.5 d (5 substeps) %.2f ms; full transport step %.2f ms => model year %.1f s",
                 label, 1e3 * t_der, 1e3 * t_eul, 1e3 * t_step[label], 730 * t_step[label]))
end
say("(measured full runs, Develop, dt 0.1 d: A100 SXM 8.4 s and H100 SXM ~5.6 s per model year at 2.8 deg)")

# ECCO-sized biology (about 4e5 boxes): time and memory, both paths
nE = 400_000
UE = CuArray(repeat(U, cld(nE, size(U, 1)))[1:nE, :]); LE = CuArray(repeat(L, cld(nE, length(L)))[1:nE])
TE = CuArray(repeat(T, cld(nE, length(T)))[1:nE])
for (label, sx) in ("kernel" => kd,)
    GC.gc(); CUDA.reclaim()
    t_E = bench(() -> simulate_euler!(copy(UE), LE, TE, 0.5, 0.1, sx), 3)
    say(@sprintf("[%s] ECCO-sized Euler 0.5 d (%d boxes): %.1f ms => biology per model year %.0f s; pool in use %.1f GiB",
                 label, nE, 1e3 * t_E, 730 * t_E, CUDA.used_memory() / 2^30))
end
UE = LE = TE = nothing; GC.gc(); CUDA.reclaim()

# ---------------------------------------------------------------- short global run
say("\n== 61-day global run on the GPU ==")
g61 = load_global_model(TMDIR, s; p=GlobalParams(tEnd=61.0))
t0 = time()
res = simulate_global(adapt(CuArray, g61), kd; verbose=false)
say(@sprintf("61 days in %.1f s", time() - t0))
dv_ok = !any(isnan, res.U)
check("no NaNs", dv_ok)
# global N inventory vs the CPU run (1e-6 agreement expected; pointwise fields diverge by chaos)
using MAT
dv = matread(joinpath(TMDIR, "grid.mat"))["dv"]
vol = [dv[g.ixBox[b], g.iyBox[b], g.izBox[b]] for b in 1:g.nb] ./ 1000 .* 1e-6
Ntot(Uk) = sum(vol .* (Uk[:, 1] .+ vec(sum(Uk[:, 4:end]; dims=2)) ./ s.rhoCN))
ref = [5.01133404e11, 5.01119701e11]          # CPU Julia run, days 30.5 and 61 (Develop 5644bf5, official 2.8 deg TMs)
for k in 1:2
    r = Ntot(res.U[:, :, k]) / ref[k] - 1
    say(@sprintf("day %.1f: N inventory %.8e, rel. diff to CPU run %.1e", res.t[k], Ntot(res.U[:, :, k]), r))
    check("N inventory day $(res.t[k]) within 1e-5 of CPU", abs(r) < 1e-5)
end

say("\n", isempty(failures) ? "ALL CHECKS PASSED" : "FAILED: " * join(failures, ", "))
close(LOG)
