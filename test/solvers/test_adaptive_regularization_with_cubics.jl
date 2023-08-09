using Manopt, Manifolds, Test, Random
using LinearAlgebra: I, tr, Symmetric

@testset "Adaptive Reguilarization with Cubics" begin
    @testset "A few solver runs" begin
        Random.seed!(42)
        n = 8
        k = 3
        A = Symmetric(randn(n, n))

        M = Grassmann(n, k)

        f(M, p) = -0.5 * tr(p' * A * p)
        grad_f(M, p) = -A * p + p * (p' * A * p)
        Hess_f(M, p, X) = -A * X + p * p' * A * X + X * p' * A * p

        p0 = Matrix{Float64}(I, n, n)[:, 1:k]
        p1 = adaptive_regularization_with_cubics(
            M, f, grad_f, Hess_f, p0; θ=0.5, σ=100.0, retraction_method=PolarRetraction()
        )
        p2 = copy(M, p0)
        adaptive_regularization_with_cubics!(
            M, f, grad_f, Hess_f, p2; θ=0.5, σ=100.0, retraction_method=PolarRetraction()
        )

        @test isapprox(M, p1, p2)

        mho = ManifoldHessianObjective(f, grad_f, Hess_f)
        M2 = TangentSpaceAtPoint(M, p0)
        g = AdaptiveRegularizationCubicCost(M2, mho)
        grad_g = AdaptiveRegularizationCubicGrad(M2, mho)
        sub_problem = DefaultManoptProblem(M2, ManifoldGradientObjective(g, grad_g))
        sub_state = GradientDescentState(
            M2,
            zero_vector(M, p0);
            stopping_criterion=StopAfterIteration(500) |
                               StopWhenGradientNormLess(1e-11) |
                               StopWhenFirstOrderProgress(0.5),
        )
        p3 = copy(M, p0) # we compute in-place of this variable
        r3 = adaptive_regularization_with_cubics!(
            M,
            mho,
            p3;
            θ=0.5,
            σ=100.0,
            retraction_method=PolarRetraction(),
            sub_problem=sub_problem,
            sub_state=sub_state,
            return_objective=true,
            return_state=true,
        )

        @test isapprox(M, p1, p3)
    end
end
