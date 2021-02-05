using Manifolds, Manopt, LinearAlgebra, Test
import Random: seed!

include("trust_region_model.jl")

@testset "Manopt Trust-Region" begin
    seed!(141)
    n=size(A,1)
    p=2
    N = Grassmann(n,p)
    M = PowerManifold(N, NestedPowerRepresentation(),2)

    x = random_point(M)

    @test_throws ErrorException trust_regions(M, cost, rgrad, rhess, x; ρ_prime=0.3)
    @test_throws ErrorException trust_regions(
        M, cost, rgrad, rhess, x; max_trust_region_radius=-0.1
    )
    @test_throws ErrorException trust_regions(
        M, cost, rgrad, rhess, x; trust_region_radius=-0.1
    )
    @test_throws ErrorException trust_regions(
        M, cost, rgrad, rhess, x; max_trust_region_radius=0.1, trust_region_radius=0.11
    )

    @testset "Allocating Variant" begin
        X = trust_regions(M, cost, rgrad, rhess, x; max_trust_region_radius=8.0,debug=[:Stop])
        opt = trust_regions(
            M, cost, rgrad, rhess, x; max_trust_region_radius=8.0, return_options=true
        )
        @test isapprox(M, X, get_solver_result(opt))

        X2 = deepcopy(x)
        trust_regions!(M, cost, rgrad, rhess, X2; max_trust_region_radius=8.0)
        @test isapprox(M, X, X2)
        XuR = trust_regions(
            M, cost, rgrad, rhess, x; max_trust_region_radius=8.0, randomize=true
        )

        @test cost(M, XuR) ≈ cost(M, X)

        XaH = trust_regions(
            M,
            cost,
            rgrad,
            ApproxHessianFiniteDifference(
                M, x, rgrad; steplength=2^(-9), vector_transport_method=ProjectionTransport()
            ),
            x;
            max_trust_region_radius=8.0,
            stopping_criterion=StopWhenAny(
                StopAfterIteration(2000), StopWhenGradientNormLess(10^(-6))
            ),
        )
        XaH2 = deepcopy(x)
        trust_regions!(
            M,
            cost,
            rgrad,
            ApproxHessianFiniteDifference(
                M, x, rgrad; steplength=2^(-9), vector_transport_method=ProjectionTransport()
            ),
            XaH2;
            stopping_criterion=StopWhenAny(
                StopAfterIteration(2000), StopWhenGradientNormLess(10^(-6))
            ),
            max_trust_region_radius=8.0,
        )
        @test isapprox(M, XaH, XaH2)
        @test cost(M, XaH) ≈ cost(M, X)

        ξ = random_tangent(M, x)
        @test_throws MethodError get_hessian(SubGradientProblem(M, cost, rgrad), x, ξ)

        η = truncated_conjugate_gradient_descent(M, cost, rgrad, x, ξ, rhess, 0.5)
        ηOpt = truncated_conjugate_gradient_descent(
            M, cost, rgrad, x, ξ, rhess, 0.5; return_options=true
        )
        @test get_solver_result(ηOpt) == η
    end
    @testset "Mutating" begin
        h = RHess(A,p)
        g = RGrad(A,p)
        x3 = deepcopy(x)
        trust_regions!(M, cost, g, h, x3;
                max_trust_region_radius=8.0, evaluation=MutatingEvaluation(), debug=[:Stop],
            )
        x4 = deepcopy(x)
        opt = trust_regions!(
            M, cost, g, h, x4; max_trust_region_radius=8.0, evaluation=MutatingEvaluation(), return_options=true
        )
        println(cost(x3))
        @test isapprox(M, x3, x4)
    end
end
