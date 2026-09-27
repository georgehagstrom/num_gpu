# GPU-readiness without a GPU: run the biology and the non-sparse parts of the transport step
# on JLArrays (reference GPU array type that forbids scalar indexing) and compare with CPU.
using Test, NUMGPU, JLArrays, Adapt, Random
JLArrays.allowscalar(false)
const ROOT = NUMGPU.numroot()
isdefined(Main, :random_states) || include("oracle_states.jl")
s = setup_num_model(joinpath(ROOT, "input", "input.yaml"))
sd = adapt(JLArray, s)
U, L, T = random_states(MersenneTwister(5), s, 300)

@testset "biology on a GPU-like array type" begin
    D = calc_derivatives(U, L, T, 0.1, s)
    Dd = calc_derivatives(JLArray(U), JLArray(L), JLArray(T), 0.1, sd)
    @test Dd isa JLArray
    # generic (non-BLAS) matmul sums in a different order: round-off-level differences only
    # same measure as runtests.jl: floor of 1e-12 × the box's largest |dudt| for near-cancellations
    e = maximum(abs.(Array(Dd) .- D) ./ (abs.(D) .+ abs.(U) .+ 1e-12 .* maximum(abs.(D); dims=2) .+ 1e-300))
    @info "device-generic derivatives: max rel err $e"
    # 1e-10: copepod stage transfer gamma = (g-mort)/(1-z^(1-mort/g)) amplifies last-bit
    # differences in mortpred when mort ≈ g (worst case seen 9e-12; all others ≤ 4e-13)
    @test e < 1e-10
    Ud = simulate_euler!(JLArray(copy(U)), JLArray(L), JLArray(T), 0.5, 0.1, sd)
    Uc = simulate_euler!(copy(U), L, T, 0.5, 0.1, s)
    e = maximum(abs.(Array(Ud) .- Uc) ./ (abs.(Uc) .+ 1e-12))
    @info "device-generic Euler 0.5 d: max rel err $e"
    @test e < 1e-10
end

@testset "fused kernel on a GPU-like array type" begin
    kd = adapt(JLArray, kernel_setup(s))
    D = calc_derivatives(U, L, T, 0.1, s)
    Dk = calc_derivatives(JLArray(U), JLArray(L), JLArray(T), 0.1, kd)
    @test Dk isa JLArray
    e = maximum(abs.(Array(Dk) .- D) ./ (abs.(D) .+ abs.(U) .+ 1e-12 .* maximum(abs.(D); dims=2) .+ 1e-300))
    @info "fused kernel on JLArray: max rel err $e"
    @test e < 1e-10
end

if !isfile(joinpath(ROOT, "TMs", "MITgcm_2.8deg", "grid.mat"))
    @info "MITgcm_2.8deg transport matrices not found: skipping the transport part of device_generic.jl"
else
@testset "sinking and bottom BC on a GPU-like array type" begin
    g = load_global_model(joinpath(ROOT, "TMs", "MITgcm_2.8deg"), s)
    gd = adapt(JLArray, g)
    Ug = g.U0 .* (1 .+ 0.1 .* rand(MersenneTwister(6), size(g.U0)))
    Uc = NUMGPU.bottom_bc!(NUMGPU.sink!(copy(Ug), g), g)
    Ud = NUMGPU.bottom_bc!(NUMGPU.sink!(JLArray(Ug), gd), gd)
    @test Array(Ud) == Uc
end
end
