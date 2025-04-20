s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using LRUCache, LinearAlgebra, Manifolds, Manopt, ManoptTestSuite, Test

@testset "Difference of Convex Plan" begin
    n = 2
    M = SymmetricPositiveDefinite(n)
    g(M, p) = log(det(p))^4 + 1 / 4
    h(M, p) = log(det(p))^2
    grad_h(M, p) = 2 * log(det(p)) * p
    f(M, p) = g(M, p) - h(M, p)
    p = log(2) * Matrix{Float64}(I, n, n)

    dc_obj = ManifoldDifferenceOfConvexObjective(f, grad_h)
    dcp_obj = ManifoldDifferenceOfConvexProximalObjective(grad_h; cost=f)
    @testset "Objective Decorator passthrough" begin
        for obj in [dc_obj, dcp_obj]
            ddo = ManoptTestSuite.DummyDecoratedObjective(obj)
            X = get_subtrahend_gradient(M, ddo, p)
            @test X == get_subtrahend_gradient(M, obj, p)
            Y = zero_vector(M, p)
            Z = zero_vector(M, p)
            get_subtrahend_gradient!(M, Y, ddo, p)
            get_subtrahend_gradient!(M, Z, obj, p)
            @test Y == Z
        end
    end
    @testset "Count" begin
        for obj in [dc_obj, dcp_obj]
            ddo = ManifoldCountObjective(M, obj, [:SubtrahendGradient])
            X = get_subtrahend_gradient(M, ddo, p)
            @test X == get_subtrahend_gradient(M, obj, p)
            Y = zero_vector(M, p)
            Z = zero_vector(M, p)
            get_subtrahend_gradient!(M, Y, ddo, p)
            get_subtrahend_gradient!(M, Z, obj, p)
            @test Y == Z
            @test get_count(ddo, :SubtrahendGradient) == 2
        end
    end
    @testset "Cache" begin
        for obj in [dc_obj, dcp_obj]
            ddo = ManifoldCountObjective(M, obj, [:SubtrahendGradient])
            cddo = objective_cache_factory(M, ddo, (:LRU, [:SubtrahendGradient]))
            X = get_subtrahend_gradient(M, obj, p)
            @test X == get_subtrahend_gradient(M, cddo, p)
            @test X == get_subtrahend_gradient(M, cddo, p) # Cached
            Y = zero_vector(M, p)
            get_subtrahend_gradient!(M, Y, cddo, p) # also cached
            @test Y == X
            @test get_count(ddo, :SubtrahendGradient) == 1

            X = get_subtrahend_gradient(M, obj, 2 .* p)
            Y = zero_vector(M, 2 .* p)
            get_subtrahend_gradient!(M, Y, cddo, 2 .* p)
            @test Y == X
            get_subtrahend_gradient!(M, Y, cddo, 2 .* p) # cached
            @test Y == X
            @test X == get_subtrahend_gradient(M, cddo, 2 .* p) # Cached
            @test get_count(ddo, :SubtrahendGradient) == 2
        end
    end
end
