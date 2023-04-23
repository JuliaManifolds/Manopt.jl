using Manifolds, Manopt, Test, LinearAlgebra

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
            p;
            max_trust_region_radius=8.0,
            stopping_criterion=StopAfterIteration(2000) | StopWhenGradientNormLess(1e-6),
        )
        q2 = copy(M, p)
        trust_regions!(
            M,
            cost,
            rgrad,
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
    @testset "...with different Hessian updates" begin
        n = 4

        A = [
            1.0 2.0 3.0 0.0
            2.0 4.0 5.0 6.0
            3.0 5.0 7.0 8.0
            0.0 6.0 8.0 9.0
        ]
        cost(::Sphere, p) = p' * A * p
        grad(::Sphere, p) = 2 * (A * p - p * p' * A * p)
        function hess(::Sphere, p, X)
            return 2 * (A * X - p * p' * A * X - X * p' * A * p - p * p' * X * p' * A * p)
        end
        grad!(::Sphere, X, p) = (X .= 2 * (A * p - p * p' * A * p))
        function hess!(::Sphere, X, p, Y)
            X .= 2 * (A * Y - p * p' * A * Y - Y * p' * A * p - p * p' * Y * p' * A * p)
            return X
        end
        M = Sphere(n - 1)
        p = zeros(n)
        p[1] = 1.0

        p_star = eigvecs(A)[:, 1]

        @testset "Allocating Variant" begin
            q = trust_regions(M, cost, grad, hess, p)
            @test isapprox(M, q, p_star)
            q2 = copy(M, p)
            trust_regions!(M, cost, grad, hess, q2)
            @test isapprox(M, q2, p_star)
            # random start point
            q3 = trust_regions(M, cost, grad, hess)
            # remove ambiguity
            q3 = (sign(q3[1]) == sign(p_star[1])) ? q3 : -q3
            @test isapprox(M, q3, p_star)

            qaHSR1 = trust_regions(
                M,
                cost,
                grad,
                ApproxHessianSymmetricRankOne(M, p, grad; nu=eps(Float64)^2),
                p;
                stopping_criterion=StopWhenAny(
                    StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
                ),
                trust_region_radius=1.0,
                θ=0.1,
                κ=0.9,
                retraction_method=ProjectionRetraction(),
            )
            @test isapprox(M, qaHSR1, p_star)

            qaHSR1_2 = copy(M, p)
            trust_regions!(
                M,
                cost,
                grad,
                ApproxHessianSymmetricRankOne(M, qaHSR1_2, grad; nu=eps(Float64)^2),
                qaHSR1_2;
                stopping_criterion=StopWhenAny(
                    StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
                ),
                trust_region_radius=1.0,
                θ=0.1,
                κ=0.9,
                retraction_method=ProjectionRetraction(),
            )
            @test isapprox(M, qaHSR1_2, p_star)

            qaHBFGS = trust_regions(
                M,
                cost,
                grad,
                ApproxHessianBFGS(M, p, grad),
                p;
                stopping_criterion=StopWhenAny(
                    StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
                ),
                trust_region_radius=1.0,
                θ=0.1,
                κ=0.9,
                retraction_method=ProjectionRetraction(),
            )
            @test isapprox(M, qaHBFGS, p_star)

            qaHBFGS_2 = copy(M, p)
            trust_regions!(
                M,
                cost,
                grad,
                ApproxHessianBFGS(M, qaHBFGS_2, grad),
                qaHBFGS_2;
                stopping_criterion=StopWhenAny(
                    StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
                ),
                trust_region_radius=1.0,
                θ=0.1,
                κ=0.9,
                retraction_method=ProjectionRetraction(),
            )
            @test isapprox(M, qaHBFGS_2, p_star)
        end
        @testset "Mutating" begin
            q3 = copy(M, p)
            trust_regions!(
                M,
                cost,
                grad!,
                hess!,
                q3;
                trust_region_radius=1.0,
                evaluation=InplaceEvaluation(),
            )
            @test isapprox(M, q3, p_star)

            q4 = copy(M, p)
            trust_regions!(
                M,
                cost,
                grad!,
                hess!,
                q4;
                trust_region_radius=1.0,
                evaluation=InplaceEvaluation(),
            )
            @test isapprox(M, q4, p_star)

            qaHSR1_3 = copy(M, p)

            trust_regions!(
                M,
                cost,
                grad!,
                ApproxHessianSymmetricRankOne(
                    M, qaHSR1_3, grad!; nu=eps(Float64)^2, evaluation=InplaceEvaluation()
                ),
                qaHSR1_3;
                stopping_criterion=StopWhenAny(
                    StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
                ),
                trust_region_radius=1.0,
                θ=0.1,
                κ=0.9,
                retraction_method=ProjectionRetraction(),
                evaluation=InplaceEvaluation(),
            )
            @test isapprox(M, qaHSR1_3, p_star)

            qaHBFGS_3 = copy(M, p)
            trust_regions!(
                M,
                cost,
                grad!,
                ApproxHessianBFGS(M, qaHBFGS_3, grad!; evaluation=InplaceEvaluation()),
                qaHBFGS_3;
                stopping_criterion=StopWhenAny(
                    StopAfterIteration(10000), StopWhenGradientNormLess(10^(-6))
                ),
                trust_region_radius=1.0,
                θ=0.1,
                κ=0.9,
                retraction_method=ProjectionRetraction(),
                evaluation=InplaceEvaluation(),
            )
            @test isapprox(M, qaHBFGS_3, p_star)
        end
    end
    @testset "on the Circle" begin
        M = Circle()
        data = [-π / 2, π / 4, 0.0, π / 4]
        p_star = 0.0
        f(M, p) = 1 / 10 * sum(distance.(Ref(M), data, Ref(p)) .^ 2)
        grad_f(M, p) = 1 / 5 * sum(-log.(Ref(M), Ref(p), data))
        hess_f(M, p, X) = 1.0
        s = trust_regions(M, f, grad_f, hess_f; return_state=true)
        q = get_solver_result(s)
        @test distance(M, p_star, q[]) < 1e-2
        q2 = trust_regions(M, f, grad_f, hess_f, 0.1)
        @test distance(M, p_star, q[]) < 1e-2
    end
end
