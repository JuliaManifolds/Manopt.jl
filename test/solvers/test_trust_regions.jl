using Manifolds, Manopt, Test
import Random: seed!

include("trust_region_model.jl")

@testset "Riemannian Trust-Region" begin
    seed!(141)
    n = size(A, 1)
    p = 2
    N = Grassmann(n, p)
    M = PowerManifold(N, ArrayPowerRepresentation(), 2)
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
        X = trust_regions(
            M, cost, rgrad, rhess, x; max_trust_region_radius=8.0, debug=[:Stop]
        )
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
                M,
                x,
                rgrad;
                steplength=2^(-9),
                vector_transport_method=ProjectionTransport(),
            ),
            x;
            max_trust_region_radius=8.0,
            stopping_criterion=StopAfterIteration(2000) | StopWhenGradientNormLess(1e-6),
        )
        XaH2 = deepcopy(x)
        trust_regions!(
            M,
            cost,
            rgrad,
            ApproxHessianFiniteDifference(
                M,
                x,
                rgrad;
                steplength=2^(-9),
                vector_transport_method=ProjectionTransport(),
            ),
            XaH2;
            stopping_criterion=StopAfterIteration(2000) | StopWhenGradientNormLess(1e-6),
            max_trust_region_radius=8.0,
        )
        @test isapprox(M, XaH, XaH2; atol=1e-6)
        @test cost(M, XaH) ≈ cost(M, X)

        ξ = random_tangent(M, x)
        @test_throws MethodError get_hessian(SubGradientProblem(M, cost, rgrad), x, ξ)

        η = truncated_conjugate_gradient_descent(
            M, cost, rgrad, x, ξ, rhess; trust_region_radius=0.5
        )
        ηOpt = truncated_conjugate_gradient_descent(
            M, cost, rgrad, x, ξ, rhess; trust_region_radius=0.5, return_options=true
        )
        @test get_solver_result(ηOpt) == η
    end
    @testset "Mutating" begin
        g = RGrad(M, A)
        h = RHess(M, A, p)
        x3 = deepcopy(x)
        trust_regions!(
            M,
            cost,
            g,
            h,
            x3;
            max_trust_region_radius=8.0,
            evaluation=MutatingEvaluation(),
            debug=[:Stop],
        )
        x4 = deepcopy(x)
        opt = trust_regions!(
            M,
            cost,
            g,
            h,
            x4;
            max_trust_region_radius=8.0,
            evaluation=MutatingEvaluation(),
            return_options=true,
        )
        @test isapprox(M, x3, x4)
        XaH = deepcopy(x)
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
                evaluation=MutatingEvaluation(),
            ),
            XaH;
            stopping_criterion=StopAfterIteration(2000) | StopWhenGradientNormLess(1e-6),
            max_trust_region_radius=8.0,
            evaluation=MutatingEvaluation(),
        )
        @test cost(M, XaH) ≈ cost(M, x3)
    end

    @testset "Stopping criteria" begin
        x5 = deepcopy(x)
        o = TrustRegionsOptions(M, x5; max_trust_region_radius=8.0)

        precon = (M, x, ξ) -> ξ
        p = HessianProblem(M, cost, rgrad, rhess, precon)

        initialize_solver!(p, o)

        # o.tcg_options.stop = StopWhenAny(
        #     StopAfterIteration(manifold_dimension(M)),
        #     StopWhenAll(
        #         StopIfResidualIsReducedByPower(1.0), StopIfResidualIsReducedByFactor(0.1)
        #     ),
        #     StopWhenTrustRegionIsExceeded(),
        #     StopWhenCurvatureIsNegative(),
        #     StopWhenModelIncreased(),
        # )

        o.tcg_options = TruncatedConjugateGradientOptions(
            p.M,
            o.x,
            o.η;
            trust_region_radius=o.trust_region_radius,
            randomize=o.randomize,
            (project!)=o.project!,
            stopping_criterion=StopAfterIteration(manifold_dimension(M)) |
                               (
                                   StopIfResidualIsReducedByPower(1.0) &
                                   StopIfResidualIsReducedByFactor(0.1)
                               ) |
                               StopWhenTrustRegionIsExceeded() |
                               StopWhenCurvatureIsNegative() |
                               StopWhenModelIncreased(),
        )

        result = solve(p, o)
        x_result = get_solver_result(result)

        x_sol = trust_regions(M, cost, rgrad, rhess, x; max_trust_region_radius=8.0)

        @test cost(M, x_result) ≈ cost(M, x_sol)
    end
end
