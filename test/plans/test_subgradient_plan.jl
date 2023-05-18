using Manopt, Manifolds, Test
include("../utils/dummy_types.jl")
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
    @testset "Objetive Decorator passthrough" begin
        ddo = DummyDecoratedObjective(mso)
        @test get_cost(M, mso, p) == get_cost(M, ddo, p)
        @test get_subgradient(M, mso, p) == get_subgradient(M, ddo, p)
        X = zero_vector(M, p)
        Y = zero_vector(M, p)
        get_subgradient!(M, X, ddo, p)
        get_subgradient!(M, Y, ddo, p)
        @test X == Y
    end
end
