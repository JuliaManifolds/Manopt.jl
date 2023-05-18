using LinearAlgebra, Manopt, Manifolds, Test
include("../utils/dummy_types.jl")

@testset "Difference of Convex Plan" begin
    n = 2
    M = SymmetricPositiveDefinite(n)
    g(M, p) = log(det(p))^4 + 1 / 4
    h(M, p) = log(det(p))^2
    grad_h(M, p) = 2 * log(det(p)) * p
    f(M, p) = g(M, p) - h(M, p)
    p = log(2) * Matrix{Float64}(I, n, n)

    dc_obj = ManifoldDifferenceOfConvexObjective(f, grad_h)
    dcp_obj = ManifoldDifferenceOfConvexProximalObjective(grad_h)
    @testset "Objetive Decorator passthrough" begin
        for obj in [dc_obj, dcp_obj]
            ddo = DummyDecoratedObjective(obj)
            X = get_subtrahend_gradient(M, ddo, p)
            @test X == get_subtrahend_gradient(M, obj, p)
            Y = zero_vector(M, p)
            Z = zero_vector(M, p)
            get_subtrahend_gradient!(M, Y, ddo, p)
            get_subtrahend_gradient!(M, Z, obj, p)
            @test Y == Z
        end
    end
end
