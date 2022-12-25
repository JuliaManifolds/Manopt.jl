using Manifolds, Manopt, Test
import Random: seed!

include("trust_region_model.jl")

@testset "Riemannian Trust-Region" begin
    n = size(A, 1)
    N = Grassmann(n, 2)
    M = PowerManifold(N, ArrayPowerRepresentation(), 2)
    p = [1.0; 0.0; 0.0;; 0.0; 1.0; 0.0;;; 0.0; 1.0; 0.0;; 0.0; 0.0; 1.0]

    @test_throws ErrorException trust_regions(M, cost, rgrad, rhess, p; ρ_prime=0.3)
    @test_throws ErrorException trust_regions(
        M, cost, rgrad, rhess, p; max_trust_region_radius=-0.1
    )
    @test_throws ErrorException trust_regions(
        M, cost, rgrad, rhess, p; trust_region_radius=-0.1
    )
    @test_throws ErrorException trust_regions(
        M, cost, rgrad, rhess, p; max_trust_region_radius=0.1, trust_region_radius=0.11
    )

    @testset "Allocating Variant" begin
        p1 = trust_regions(
            M, cost, rgrad, rhess, p; max_trust_region_radius=8.0, debug=[:Stop]
        )
        opt = trust_regions(
            M, cost, rgrad, rhess, p; max_trust_region_radius=8.0, return_state=true
        )
        @test isapprox(M, X, get_solver_result(opt))

        q = copy(M, p)
        trust_regions!(M, cost, rgrad, rhess, q; max_trust_region_radius=8.0)
        @test isapprox(M, X, q)
        p2 = trust_regions(
            M, cost, rgrad, rhess, p; max_trust_region_radius=8.0, randomize=true
        )

        @test cost(M, p2) ≈ cost(M, p1)

        p3 = trust_regions(
            M,
            cost,
            rgrad,
            ApproxHessianFiniteDifference(
                M,
                p,
                rgrad;
                steplength=2^(-9),
                vector_transport_method=ProjectionTransport(),
            ),
            p;
            max_trust_region_radius=8.0,
            stopping_criterion=StopAfterIteration(2000) | StopWhenGradientNormLess(1e-6),
        )
        q2 = copy(M, p)
        trust_regions!(
            M,
            cost,
            rgrad,
            ApproxHessianFiniteDifference(
                M,
                p,
                rgrad;
                steplength=2^(-9),
                vector_transport_method=ProjectionTransport(),
            ),
            q2;
            stopping_criterion=StopAfterIteration(2000) | StopWhenGradientNormLess(1e-6),
            max_trust_region_radius=8.0,
        )
        @test isapprox(M, p3, q2; atol=1e-6)
        @test cost(M, p3) ≈ cost(M, p1)

        X =  zero_vector(M,p)
        @test_throws MethodError get_hessian(SubGradientProblem(M, cost, rgrad), x, X)

        Y = truncated_conjugate_gradient_descent(
            M, cost, rgrad, p, X, rhess; trust_region_radius=0.5
        )
        cost(M, X) > cost(M, Y)
    end
    @testset "Mutating" begin
        g = RGrad(M, A)
        h = RHess(M, A, p)
        p1 = copy(M; p)
        trust_regions!(
            M,
            cost,
            g,
            h,
            p1;
            max_trust_region_radius=8.0,
            evaluation=InplaceEvaluation(),
        )
        p2 = copy(M, p)
        trust_regions!(
            M,
            cost,
            g,
            ApproxHessianFiniteDifference(
                M,
                deepcopy(x),
                g;
                steplength=2^(-9),
                vector_transport_method=ProjectionTransport(),
                evaluation=InplaceEvaluation(),
            ),
            p2;
            stopping_criterion=StopAfterIteration(2000) | StopWhenGradientNormLess(1e-6),
            max_trust_region_radius=8.0,
            evaluation=InplaceEvaluation(),
        )
        @test cost(M, p2) ≈ cost(M, p1)
    end
end
