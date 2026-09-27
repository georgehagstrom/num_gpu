# Compare the global driver with a dump from the upstream Matlab simulateGlobal
# (preprocessed operators and the first transport steps, double precision).
# Usage: julia --project=test test/global_steps.jl <matlab_dump.mat>
# The dump comes from a copy of simulateGlobal.m with save() calls added after the setup of
# BCvalue and after each of the first 3 steps: python scripts/dump_global_reference.py <dir>.
using Test, NUMGPU, MAT, SparseArrays, LinearAlgebra
include("FortranOracle.jl")
using .FortranOracle

const ROOT = FortranOracle.NUMROOT
dump = matread(ARGS[1])
s = setup_num_model(joinpath(ROOT, "input", "input.yaml"))
g = load_global_model(joinpath(ROOT, "TMs", "MITgcm_2.8deg"), s)

relmax(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

@testset "setup vs Matlab" begin
    @test relmax(initial_state(s), vec(dump["u0"])) < 1e-15
    @test vcat(zeros(3), vec(s.velocity)) == vec(dump["velocity"])
    @test g.U0[:, 1:3] == dump["U_init"][:, 1:3]
    @test relmax(g.U0, dump["U_init"]) < 1e-15
    @test relmax(g.L0, dump["L0"]) < 1e-15
    @test isequal(g.Tmat, dump["Tmat"])
    @test g.ixBottom == Int.(vec(dump["ixBottom"]))
    @test g.dzBottom == vec(dump["dzBottom"])
    @test g.BCvalue == dump["BCvalue"][:, 1:3]
    @test g.ixSink == Int.(vec([dump["idxSinking"];]))
    Asink = dump["Asink"]
    for (l, j) in enumerate(g.ixSink)
        A = Asink isa AbstractArray ? Asink[l] : Asink
        @test diag(A) == g.keep[:, l]
        J = sparse(1:g.nb, g.above, g.gain[:, l], g.nb, g.nb)   # gain part
        @test A - spdiagm(diag(A)) == dropzeros(J)
    end
    @test g.Aexp[1] == dump["Aexp_m1"]
    @test g.Aimp[1] == dump["Aimp_m1"]
    @test isequal(vec(dump["T_m1"]), g.Tmat[:, 1])
end

@testset "first 3 transport steps from Matlab's initial state ($label)" for (label, sk) in
        ("kernel" => kernel_setup(s),)
    U = copy(dump["U_init"])
    work = similar(U)
    ms = Ref((0, 0))
    for i in 1:3
        NUMGPU.transport_step!(U, g, i, ms, sk, work)
        Um = dump["U_step$i"]
        e = abs.(U .- Um) ./ (abs.(Um) .+ 1e-12 .* maximum(abs.(Um); dims=1))
        worst = argmax(e)
        @info "$label, step $i: max rel err $(maximum(e)) at box $(worst[1]) var $(worst[2]); " *
              "fraction bit-identical $(count(U .== Um) / length(U))"
        @test maximum(e) < 1e-10
    end
end
