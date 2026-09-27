# CPU baseline on a many-core machine: upstream Fortran biology per box (what Matlab's parfor
# calls) on P worker processes, and the Julia fused kernel on all threads, on a realistic state.
# The Fortran library keeps module-level state, so it is parallelized over processes (like
# Matlab's parfor workers), not threads. Its timing is a best case for Matlab: no parfor
# data-transfer or Matlab overhead.
#
# Usage: julia --project=julia/gpu -t <threads> scripts/cpu_bench.jl <state.mat> [P1,P2,...]
#   NUM_TMDIR: transport-matrix directory; NUM_FORTRAN_ORACLE: path to test/FortranOracle.jl
using Distributed, NUMGPU, MAT, Printf, Statistics, LinearAlgebra, Random

const ROOT = NUMGPU.numroot()
const TMDIR = get(ENV, "NUM_TMDIR", joinpath(ROOT, "TMs", "MITgcm_2.8deg"))
const ORACLE = get(ENV, "NUM_FORTRAN_ORACLE", joinpath(@__DIR__, "..", "julia", "NUMGPU", "test", "FortranOracle.jl"))
state_file = ARGS[1]
procs_list = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ",")) : [Sys.CPU_THREADS]
LinearAlgebra.BLAS.set_num_threads(1)

cpu = try strip(split(filter(l -> occursin("Model name", l), readlines(`lscpu`))[1], ":")[2]) catch; "?" end
println("CPU: $cpu; logical CPUs: $(Sys.CPU_THREADS); Julia threads: $(Threads.nthreads())")

s = setup_num_model(joinpath(ROOT, "input", "input.yaml"))
g = load_global_model(TMDIR, s)
U = matread(state_file)["U"]                    # nb×nGrid state (day 183 of a CPU run)
day = 183; iL = 2 * day + 1                     # step index used by simulateGlobal on that day
L = g.L0[:, iL]; T = g.Tmat[:, 7]               # July temperature
nb = size(U, 1)
println("boxes: $nb")

# ---------------------------------------------------------------- transport part on the CPU
work = similar(U); Ut = copy(U)
t_tm = minimum(@elapsed((mul!(work, g.Aexp[7], Ut); mul!(Ut, g.Aimp[7], work))) for _ in 1:5)
t_sb = minimum(@elapsed((NUMGPU.sink!(Ut, g); NUMGPU.bottom_bc!(Ut, g))) for _ in 1:5)
# threaded transport: tracer columns split across threads (sparse*dense is single-threaded)
function tm_threaded!(U, work, Ae, Ai)
    cols = collect(Iterators.partition(axes(U, 2), cld(size(U, 2), Threads.nthreads())))
    Threads.@threads for c in cols
        mul!(view(work, :, c), Ae, view(U, :, c)); mul!(view(U, :, c), Ai, view(work, :, c))
    end
    return U
end
tm_threaded!(Ut, work, g.Aexp[7], g.Aimp[7])
t_tmT = minimum(@elapsed(tm_threaded!(Ut, work, g.Aexp[7], g.Aimp[7])) for _ in 1:5)
@printf("transport Aimp*(Aexp*U): %.1f ms on 1 thread, %.1f ms on %d threads; sinking + bottom BC: %.1f ms\n",
        1e3t_tm, 1e3t_tmT, Threads.nthreads(), 1e3t_sb)
t_rest = t_tmT + t_sb    # best case: threaded transport

# ---------------------------------------------------------------- Fortran, one process
include(ORACLE)
using .FortranOracle
FortranOracle.setup_num_model!()
sample = randperm(MersenneTwister(1), nb)[1:2000]
FortranOracle.simulate_euler(U[1, :], L[1], T[1], 0.5, 0.1)
t1 = @elapsed for b in sample
    FortranOracle.simulate_euler(U[b, :], L[b], T[b], 0.5, 0.1)
end
us_box = t1 / length(sample) * 1e6
@printf("Fortran, 1 core: %.0f us per box per 0.5 d step => %.2f core-hours of biology per model year\n",
        us_box, us_box * 1e-6 * nb * 730 / 3600)

# ---------------------------------------------------------------- Fortran, P processes
function fortran_parallel(P, U, L, T; nsteps=2)
    ws = addprocs(P; exeflags="--project=$(Base.active_project())", enable_threaded_blas=false)
    try
        Distributed.remotecall_eval(Main, ws, quote
            include($ORACLE)
            using .FortranOracle
            FortranOracle.setup_num_model!()
            global DATA = nothing
            function run_slice(Us, Ls, Ts, nsteps)
                for _ in 1:nsteps, b in axes(Us, 1)
                    Us[b, :] = FortranOracle.simulate_euler(Us[b, :], Ls[b], Ts[b], 0.5, 0.1)
                end
                return Us
            end
        end)
        # interleaved slices balance deep/surface boxes; data is sent once, before timing
        slices = [collect(p:P:size(U, 1)) for p in 1:P]
        for (w, sl) in zip(ws, slices)
            remotecall_wait((Us, Ls, Ts) -> (global DATA = (Us, Ls, Ts); nothing), w, U[sl, :], L[sl], T[sl])
        end
        foreach(fetch, [remotecall(() -> (run_slice(DATA[1][1:10, :], DATA[2][1:10], DATA[3][1:10], 1); nothing), w) for w in ws])
        t = @elapsed foreach(fetch, [remotecall(() -> (run_slice(DATA[1], DATA[2], DATA[3], nsteps); nothing), w) for w in ws])
        return t / nsteps
    finally
        rmprocs(ws)
    end
end

for P in procs_list
    t_step = fortran_parallel(P, U, L, T)
    year = 730 * (t_step + t_rest)
    @printf("Fortran, %d processes: %.2f s per 0.5 d biology step => model year %.0f s (%.1f min) incl. transport (best case for Matlab parfor)\n",
            P, t_step, year, year / 60)
end

# ---------------------------------------------------------------- Julia fused kernel, all threads
k = kernel_setup(s)
Uk = copy(U); simulate_euler!(Uk, L, T, 0.5, 0.1, k)
t_k = minimum(@elapsed(simulate_euler!(copy(U), L, T, 0.5, 0.1, k)) for _ in 1:3)
year_k = 730 * (t_k + t_rest)
@printf("Julia fused kernel, %d threads: %.3f s per 0.5 d biology step => model year %.0f s (%.1f min) incl. transport\n",
        Threads.nthreads(), t_k, year_k, year_k / 60)
println("(for comparison: A100 SXM GPU, full runs, ~8.4 s per model year at 2.8 deg)")
