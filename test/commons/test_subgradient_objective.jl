using LRUCache, Manifolds, Manopt, Test

@testset "Subgradient Objective" begin
    M = Euclidean(2)
    p = [1.0, 2.0]
    f, ∂f, ∂f! = Manopt.Test.distance_task(M, p)
    mso = ManifoldSubgradientObjective(f, ∂f)
    msoi = ManifoldSubgradientObjective(f, ∂f!; evaluation = InplaceEvaluation())
    # a point where the subgradient is not the zero vector
    q = [3.0, -1.0]
    @testset "Objective Decorator passthrough" begin
        ddo = Manopt.Test.DummyDecoratedObjective(mso)
        @test get_cost(M, mso, q) == get_cost(M, ddo, q)
        @test get_subgradient(M, mso, q) == get_subgradient(M, ddo, q)
        X = zero_vector(M, q)
        Y = zero_vector(M, q)
        get_subgradient!(M, X, mso, q)
        get_subgradient!(M, Y, ddo, q)
        @test X == Y
        # Forms an alloc wrapper
        @test Manopt.get_subgradient_function(msoi; evaluation = AllocatingEvaluation())(M, q) == X
        # “unwraps” alloc version
        @test Manopt.get_subgradient_function(ddo; evaluation = AllocatingEvaluation()) == ∂f
    end
    @testset "Count" begin
        ddo = ManifoldCountObjective(M, mso, [:SubGradient])
        @test get_subgradient(M, mso, q) == get_subgradient(M, ddo, q)
        X = zero_vector(M, q)
        Y = zero_vector(M, q)
        get_subgradient!(M, X, mso, q)
        get_subgradient!(M, Y, ddo, q)
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
