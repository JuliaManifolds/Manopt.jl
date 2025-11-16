s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using LinearAlgebra, Manifolds, Manopt, ManoptTestSuite, Random, Test

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

    @test_throws ErrorException trust_regions(
        M, f, rgrad, rhess, p; max_trust_region_radius = -0.1
    )
    @test_throws ErrorException trust_regions(
        M, f, rgrad, rhess, p; trust_region_radius = -0.1
    )
    @test_throws ErrorException trust_regions(
        M, f, rgrad, rhess, p; max_trust_region_radius = 0.1, trust_region_radius = 0.11
    )
    @testset "State Constructors" begin
        X = rgrad(M, p)
        TpM = TangentSpace(M, copy(M, p))
        mho = ManifoldHessianObjective(f, rgrad, rhess)
        sub_objective = TrustRegionModelObjective(mho)
        sub_problem = DefaultManoptProblem(TpM, sub_objective)
        sub_state = TruncatedConjugateGradientState(TpM; X = get_gradient(M, mho, p))
        trs1 = TrustRegionsState(M, sub_problem)
        trs2 = TrustRegionsState(M, sub_problem, sub_state)
        trs3 = TrustRegionsState(M, sub_problem; p = p)
        @test Manopt.get_gradient_function(sub_objective)(M, p) == X
    end
    @testset "Objective accessors" begin
        mho = ManifoldHessianObjective(f, rgrad, rhess)
        X = rgrad(M, p)
        TpM = TangentSpace(M, copy(M, p))
        trmo = TrustRegionModelObjective(mho)
        c = f(M, p)
        g = inner(M, p, X, X)
        H = rhess(M, p, X)
        @test get_cost(TpM, trmo, X) == c + g + 1 / 2 * inner(M, p, H, X)
        @test get_gradient(TpM, trmo, X) == X + H
        Y = similar(X)
        get_gradient!(TpM, Y, trmo, X)
        @test Y == X + H
        @test get_hessian(TpM, trmo, Y, X) == H
        get_hessian!(TpM, Y, trmo, Y, X)
        @test Y == H
    end
    @testset "Allocating Variant" begin
        s = trust_regions(
            M, f, rgrad, rhess, p; max_trust_region_radius = 8.0, return_state = true
        )
        @test startswith(repr(s), "# Solver state for `Manopt.jl`s Trust Region Method\n")
        p1 = get_solver_result(s)
        q = copy(M, p)
        set_gradient!(s, M, p, zero_vector(M, p))
        @test norm(M, p, get_gradient(s)) ≈ 0.0
        trust_regions!(M, f, rgrad, rhess, q; max_trust_region_radius = 8.0)
        @test isapprox(M, p1, q)
        Random.seed!(42)
        p2 = trust_regions(
            M, f, rgrad, rhess, p; max_trust_region_radius = 8.0, randomize = true
        )

        @test f(M, p2) ≈ f(M, p1)

        p3 = trust_regions(
            M,
            f,
            rgrad,
            p;
            max_trust_region_radius = 8.0,
            stopping_criterion = StopAfterIteration(2000) | StopWhenGradientNormLess(1.0e-6),
        )
        q2 = copy(M, p)
        trust_regions!(
            M,
            f,
            rgrad,
            q2;
            stopping_criterion = StopAfterIteration(2000) | StopWhenGradientNormLess(1.0e-6),
            max_trust_region_radius = 8.0,
        )
        @test isapprox(M, p3, q2; atol = 1.0e-6)
        @test f(M, p3) ≈ f(M, p1)
    end
    @testset "TCG" begin
        X = zero_vector(M, p)
        Y = truncated_conjugate_gradient_descent(
            M, f, rgrad, rhess, p, X; trust_region_radius = 0.5
        )
        @test Y != X
        # random point -> different result
        Y2 = truncated_conjugate_gradient_descent( #random point and vector
            M,
            f,
            rgrad,
            rhess;
            trust_region_radius = 0.5,
        )
        @test Y2 != X
        Y3 = truncated_conjugate_gradient_descent(
            M, f, rgrad, rhess, p, X; trust_region_radius = 0.5
        )
        @test Y3 != X
        Y4 = copy(M, p, X)
        truncated_conjugate_gradient_descent!(
            M, f, rgrad, rhess, p, Y4; trust_region_radius = 0.5
        )
        @test Y4 != X
    end
    @testset "Mutating" begin
        g = RGrad(M, A)
        h = RHess(M, A, m)
        p1 = copy(M, p)
        trust_regions!(
            M, f, g, h, p1; max_trust_region_radius = 8.0, evaluation = InplaceEvaluation()
        )
        p2 = copy(M, p)
        trust_regions!(
            M,
            f,
            g,
            ApproxHessianFiniteDifference(
                M,
                copy(M, p),
                g;
                steplength = 2^(-9),
                vector_transport_method = ProjectionTransport(),
                evaluation = InplaceEvaluation(),
            ),
            p2;
            stopping_criterion = StopAfterIteration(2000) | StopWhenGradientNormLess(1.0e-6),
            max_trust_region_radius = 8.0,
            evaluation = InplaceEvaluation(),
        )
        @test f(M, p2) ≈ f(M, p1)
        X = zero_vector(M, p)
        Y6 = truncated_conjugate_gradient_descent( # in-place -> other precondition default
            M,
            f,
            g,
            h,
            p,
            X;
            evaluation = InplaceEvaluation(),
            trust_region_radius = 0.5,
        )
        @test Y6 != X
        Y9 = copy(M, p, X)
        truncated_conjugate_gradient_descent!(
            M, f, g, h, p, Y6; evaluation = InplaceEvaluation(), trust_region_radius = 0.5
        )
        @test Y6 != X
    end
    @testset "...with different Hessian updates" begin
        n = 4

        A = [
            1.0 2.0 3.0 0.0
            2.0 4.0 5.0 6.0
            3.0 5.0 7.0 8.0
            0.0 6.0 8.0 9.0
        ]
        f(::Sphere, p) = p' * A * p
        grad_f(::Sphere, p) = 2 * (A * p - p * p' * A * p)
        function Hess_f(::Sphere, p, X)
            return 2 * (A * X - p * p' * A * X - X * p' * A * p - p * p' * X * p' * A * p)
        end
        grad_f!(::Sphere, X, p) = (X .= 2 * (A * p - p * p' * A * p))
        function Hess_f!(::Sphere, X, p, Y)
            X .= 2 * (A * Y - p * p' * A * Y - Y * p' * A * p - p * p' * Y * p' * A * p)
            return X
        end
        M = Sphere(n - 1)
        p = zeros(n)
        p[1] = 1.0

        p_star = eigvecs(A)[:, 1]

        @testset "Allocating Variant" begin
            q = trust_regions(M, f, grad_f, Hess_f, p)
            @test isapprox(M, q, p_star) || isapprox(M, q, -p_star)
            q2 = copy(M, p)
            trust_regions!(M, f, grad_f, Hess_f, q2)
            @test isapprox(M, q2, p_star) || isapprox(M, q2, -p_star)
            # random start point
            q3 = trust_regions(M, f, grad_f, Hess_f)
            # remove ambiguity
            q3 = (sign(q3[1]) == sign(p_star[1])) ? q3 : -q3
            @test isapprox(M, q3, p_star) || isapprox(M, q3, -p_star)

            # a Default
            qaAoor = trust_regions(M, f, grad_f)
            #
            qaHSR1 = trust_regions(
                M,
                f,
                grad_f!,
                ApproxHessianSymmetricRankOne(
                    M, p, grad_f!; nu = eps(Float64)^2, evaluation = InplaceEvaluation()
                ),
                p;
                stopping_criterion = StopAfterIteration(100) | StopWhenGradientNormLess(1.0e-8),
                trust_region_radius = 1.0,
                θ = 0.1,
                κ = 0.9,
                retraction_method = ProjectionRetraction(),
                evaluation = InplaceEvaluation(),
            )
            @test isapprox(M, qaHSR1, p_star) || isapprox(M, qaHSR1, -p_star)

            qaHSR1_2 = copy(M, p)
            trust_regions!(
                M,
                f,
                grad_f,
                ApproxHessianSymmetricRankOne(M, qaHSR1_2, grad_f; nu = eps(Float64)^2),
                qaHSR1_2;
                stopping_criterion = StopAfterIteration(10000) |
                    StopWhenGradientNormLess(1.0e-6),
                trust_region_radius = 1.0,
                θ = 0.1,
                κ = 0.9,
                retraction_method = ProjectionRetraction(),
            )
            @test isapprox(M, qaHSR1_2, p_star) || isapprox(M, qaHSR1_2, -p_star)

            qaHBFGS = trust_regions(
                M,
                f,
                grad_f,
                ApproxHessianBFGS(M, p, grad_f),
                p;
                stopping_criterion = StopAfterIteration(10000) |
                    StopWhenGradientNormLess(1.0e-6),
                trust_region_radius = 1.0,
                θ = 0.1,
                κ = 0.9,
                retraction_method = ProjectionRetraction(),
            )
            @test isapprox(M, qaHBFGS, p_star) || isapprox(M, qaHBFGS, -p_star)

            qaHBFGS_2 = copy(M, p)
            trust_regions!(
                M,
                f,
                grad_f,
                ApproxHessianBFGS(M, qaHBFGS_2, grad_f),
                qaHBFGS_2;
                stopping_criterion = StopAfterIteration(10000) |
                    StopWhenGradientNormLess(1.0e-6),
                trust_region_radius = 1.0,
                θ = 0.1,
                κ = 0.9,
                retraction_method = ProjectionRetraction(),
            )
            @test isapprox(M, qaHBFGS_2, p_star) || isapprox(M, qaHBFGS_2, -p_star)
        end
        @testset "Mutating" begin
            q3 = copy(M, p)
            trust_regions!(
                M,
                f,
                grad_f!,
                Hess_f!,
                q3;
                trust_region_radius = 1.0,
                evaluation = InplaceEvaluation(),
            )
            @test isapprox(M, q3, p_star) || isapprox(M, q3, -p_star)

            q4 = copy(M, p)
            trust_regions!(
                M,
                f,
                grad_f!,
                Hess_f!,
                q4;
                trust_region_radius = 1.0,
                evaluation = InplaceEvaluation(),
            )
            @test isapprox(M, q4, p_star) || isapprox(M, q4, -p_star)

            qaHSR1_3 = copy(M, p)

            trust_regions!(
                M,
                f,
                grad_f!,
                ApproxHessianSymmetricRankOne(
                    M, qaHSR1_3, grad_f!; nu = eps(Float64)^2, evaluation = InplaceEvaluation()
                ),
                qaHSR1_3;
                stopping_criterion = StopAfterIteration(10000) |
                    StopWhenGradientNormLess(1.0e-6),
                trust_region_radius = 1.0,
                θ = 0.1,
                κ = 0.9,
                retraction_method = ProjectionRetraction(),
                evaluation = InplaceEvaluation(),
            )
            @test isapprox(M, qaHSR1_3, p_star) || isapprox(M, qaHSR1_3, -p_star)

            qaHBFGS_3 = copy(M, p)
            trust_regions!(
                M,
                f,
                grad_f!,
                ApproxHessianBFGS(M, qaHBFGS_3, grad_f!; evaluation = InplaceEvaluation()),
                qaHBFGS_3;
                stopping_criterion = StopAfterIteration(10000) |
                    StopWhenGradientNormLess(1.0e-6),
                trust_region_radius = 1.0,
                θ = 0.1,
                κ = 0.9,
                retraction_method = ProjectionRetraction(),
                evaluation = InplaceEvaluation(),
            )
            @test isapprox(M, qaHBFGS_3, p_star) || isapprox(M, qaHBFGS_3, -p_star)
        end
    end
    @testset "on the Circle" begin
        Mc, fc, grad_fc, pc0, pc_star = ManoptTestSuite.Circle_mean_task()
        hess_fc(Mc, p, X) = 1.0
        s = trust_regions(Mc, fc, grad_fc, hess_fc; return_state = true)
        q = get_solver_result(s)
        @test distance(Mc, pc_star, q[]) < 1.0e-2
        q2 = trust_regions(Mc, fc, grad_fc, hess_fc, 0.1)
        @test distance(Mc, pc_star, q[]) < 1.0e-2
        q2 = trust_regions(Mc, fc, grad_fc, hess_fc)
        @test distance(Mc, pc_star, q[]) < 1.0e-2
        Y1 = truncated_conjugate_gradient_descent(
            Mc, fc, grad_fc, hess_fc, 0.1, 0.0; trust_region_radius = 0.5
        )
        @test abs(Y1) ≈ 0.5
    end
    @testset "Euclidean Embedding" begin
        Random.seed!(42)
        n = 2
        A = Symmetric(randn(n + 1, n + 1))
        # Euclidean variant with conversion
        M = Sphere(n)
        p0 = [1.0, zeros(n)...]
        f(E, p) = p' * A * p
        ∇f(E, p) = A * p
        ∇²f(M, p, X) = A * X
        λ = min(eigvals(A)...)
        q = trust_regions(M, f, ∇f, p0; objective_type = :Euclidean, (project!) = (project!))
        @test f(M, q) ≈ λ atol = 1 * 1.0e-1 # a bit imprecise?
        grad_f(M, p) = A * p - (p' * A * p) * p
        Hess_f(M, p, X) = A * X - (p' * A * X) .* p - (p' * A * p) .* X
        q3 = trust_regions(M, f, grad_f, p0)
        q4 = trust_regions(M, f, grad_f, Hess_f, p0)
        @test f(M, q3) ≈ λ atol = 5 * 1.0e-8
        @test f(M, q4) ≈ λ atol = 5 * 1.0e-10
    end
end
