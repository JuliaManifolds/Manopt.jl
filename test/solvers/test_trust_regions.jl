using Manifolds, Manopt, Test
import Random: seed!

include("trust_region_model.jl")

@testset "Riemannian Trust-Region" begin
    n = size(A, 1)
    m = 2
    N = Grassmann(n, m)
    M = PowerManifold(N, ArrayPowerRepresentation(), 2)
    # generate a 3x2x2 Array
    p = zeros(3, 2, 2)
    p[:, :, 1] = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    p[:, :, 2] = [0.0 0.0; 1.0 0.0; 0.0 1.0]

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
        s = trust_regions(
            M, cost, rgrad, rhess, p; max_trust_region_radius=8.0, return_state=true
        )
        @test startswith(repr(s), "# Solver state for `Manopt.jl`s Trust Region Method\n")
        p1 = get_solver_result(s)
        q = copy(M, p)
        set_gradient!(s, M, p, zero_vector(M, p))
        @test norm(M, p, get_gradient(s)) ≈ 0.0
        trust_regions!(M, cost, rgrad, rhess, q; max_trust_region_radius=8.0)
        @test isapprox(M, p1, q)
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

        X = zero_vector(M, p)

        Y = truncated_conjugate_gradient_descent(
            M, cost, rgrad, p, X, rhess; trust_region_radius=0.5
        )
        cost(M, X) > cost(M, Y)
    end
    @testset "Mutating" begin
        g = RGrad(M, A)
        h = RHess(M, A, m)
        p1 = copy(M, p)
        trust_regions!(
            M, cost, g, h, p1; max_trust_region_radius=8.0, evaluation=InplaceEvaluation()
        )
        p2 = copy(M, p)
        trust_regions!(
            M,
            cost,
            g,
            ApproxHessianFiniteDifference(
                M,
                copy(M, p),
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
