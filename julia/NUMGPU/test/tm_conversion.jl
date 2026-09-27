# TM positivity conversions (GlobalParams.TMconversion): on a random operator with negative
# off-diagonals, zero row sums and zero volume-weighted column sums (like a TM's Aexp).
using NUMGPU, SparseArrays, LinearAlgebra, Random, Test

@testset "TM conversion" begin
    rng = MersenneTwister(3)
    n = 300
    v = 1 .+ 9 .* rand(rng, n)                            # box volumes
    # exchange fluxes S (symmetric, some negative); W = S - diag(S*1) has zero row and column
    # sums, so A = diag(1/v)*W keeps a uniform field and conserves the volume integral
    S = sprand(rng, n, n, 0.03) .- 0.3 .* sprand(rng, n, n, 0.01)
    S = S - spdiagm(diag(S)); S = S + sparse(S')
    W = S - spdiagm(vec(sum(S; dims=2)))
    A = sparse(Diagonal(1 ./ v) * W)
    x = rand(rng, n)
    offneg(B) = count(<(0), nonzeros(B - spdiagm(diag(B))))
    @test offneg(A) > 0
    @test maximum(abs, A * ones(n)) < 1e-12
    @test abs(dot(v, A * x)) < 1e-12
    for how in (:v1, :develop, :volume)
        B = NUMGPU.convert_TM(A, how, v)
        @test offneg(B) == 0
        how === :develop || @test maximum(abs, B * ones(n)) < 1e-12   # uniform field stays uniform
    end
    # Develop's version conserves mass but changes row sums (uniform field not exactly kept)
    @test abs(dot(v, NUMGPU.convert_TM(A, :develop, v) * x)) < 1e-12
    @test maximum(abs, NUMGPU.convert_TM(A, :develop, v) * ones(n)) > 1e-6
    @test abs(dot(v, NUMGPU.convert_TM(A, :volume, v) * x)) < 1e-12     # mass conserved
    @test abs(dot(v, NUMGPU.convert_TM(A, :v1, v) * x)) > 1e-6          # v1.0: not conserved
    @test NUMGPU.convert_TM(A, :none, v) == A
    @test_throws ErrorException NUMGPU.convert_TM(A, :foo, v)
end
