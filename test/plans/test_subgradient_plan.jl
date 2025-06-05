s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using LRUCache, Manifolds, Manopt, ManoptTestSuite, Test

@testset "Subgradient Plan" begin
    M = Euclidean(2)
    p = [1.0, 2.0]
    f(M, q) = distance(M, q, p)
    function ∂f(M, q)
        if distance(M, p, q) == 0
            return zero_vector(M, q)
        end
        return -log(M, q, p) / max(10 * eps(Float64), distance(M, p, q))
    end
    mso = ManifoldSubgradientObjective(f, ∂f)
    @testset "Objective Decorator passthrough" begin
        ddo = ManoptTestSuite.DummyDecoratedObjective(mso)
        @test get_cost(M, mso, p) == get_cost(M, ddo, p)
        @test get_subgradient(M, mso, p) == get_subgradient(M, ddo, p)
        X = zero_vector(M, p)
        Y = zero_vector(M, p)
        get_subgradient!(M, X, mso, p)
        get_subgradient!(M, Y, ddo, p)
        @test X == Y
    end
    @testset "Count" begin
        ddo = ManifoldCountObjective(M, mso, [:SubGradient])
        @test get_subgradient(M, mso, p) == get_subgradient(M, ddo, p)
        X = zero_vector(M, p)
        Y = zero_vector(M, p)
        get_subgradient!(M, X, mso, p)
        get_subgradient!(M, Y, ddo, p)
        @test X == Y
        @test get_count(ddo, :SubGradient) == 2
    end
    @testset "Cache" begin
        ddo = ManifoldCountObjective(M, mso, [:SubGradient])
        cddo = objective_cache_factory(M, ddo, (:LRU, [:SubGradient]))
        X = get_subgradient(M, mso, p)
        @test get_subgradient(M, cddo, p) == X
        @test get_subgradient(M, cddo, p) == X #Cached
        Y = zero_vector(M, p)
        get_subgradient!(M, Y, cddo, p) # Cached
        @test X == Y
        @test get_count(ddo, :SubGradient) == 1

        X = get_subgradient(M, mso, -p)
        Y = zero_vector(M, p)
        get_subgradient!(M, Y, cddo, -p)
        @test X == Y
        get_subgradient!(M, Y, cddo, -p) # Cached
        @test X == Y
        @test get_subgradient(M, cddo, -p) == X #Cached
        @test get_count(ddo, :SubGradient) == 2
    end
end
