using Manopt, Manifolds, ManifoldsBase, Test, Random, LinearAlgebra
using LinearAlgebra: Diagonal, dot, eigvals, eigvecs

@testset "Conjugate Gradient Descent" begin
    @testset "Conjugate Gradient coefficient rules" begin
        M = Euclidean(2)
        f(M, x) = norm(x)^2
        grad_f(::Euclidean, x) = 2 * x
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        x0 = [0.0, 1.0]
        sC = StopAfterIteration(1)
        s = Manopt.ConstantStepsize(M)
        retr = ExponentialRetraction()
        vtm = ParallelTransport()

        grad_1 = [1.0, 1.0]
        δ1 = [0.0, 2.0]
        grad_2 = [1.0, 1.5]
        δ2 = [0.5, 2.0]
        diff = grad_2 - grad_1

        dU = SteepestDescentCoefficient()
        s1 = ConjugateGradientDescentState(
            M;
            p = x0,
            stopping_criterion = sC,
            stepsize = s,
            coefficient = dU,
            retraction_method = retr,
            vector_transport_method = vtm,
            initial_gradient = zero_vector(M, x0),
        )
        @test s1.coefficient(dmp, s1, 1) == 0
        @test default_stepsize(M, typeof(s1)) isa Manopt.ArmijoLinesearchStepsize
        @test Manopt.get_message(s1) == ""

        dU = Manopt.ConjugateDescentCoefficient()
        s2 = ConjugateGradientDescentState(
            M;
            p = x0,
            stopping_criterion = sC,
            stepsize = s,
            coefficient = dU,
            retraction_method = retr,
            vector_transport_method = vtm,
            initial_gradient = zero_vector(M, x0),
        )
        s2.X = grad_1
        s2.δ = δ1
        # the first case is zero
        @test s2.coefficient(dmp, s2, 1) == 0.0
        s2.X = grad_2
        s2.δ = δ2
        @test s2.coefficient(dmp, s2, 2) == dot(grad_2, grad_2) / dot(-δ2, grad_1)

        dU = DaiYuanCoefficient()
        s3 = ConjugateGradientDescentState(
            M;
            p = x0,
            stopping_criterion = sC,
            stepsize = s,
            coefficient = dU,
            retraction_method = retr,
            vector_transport_method = vtm,
        )
        s3.X = grad_1
        s3.δ = δ1
        # the first case is zero
        @test s3.coefficient(dmp, s3, 1) == 0.0
        s3.X = grad_2
        s3.δ = δ2
        @test s3.coefficient(dmp, s3, 2) == dot(grad_2, grad_2) / dot(δ2, grad_2 - grad_1)

        dU = FletcherReevesCoefficient()
        s4 = ConjugateGradientDescentState(
            M;
            p = x0,
            stopping_criterion = sC,
            stepsize = s,
            coefficient = dU,
            retraction_method = retr,
            vector_transport_method = vtm,
        )
        s4.X = grad_1
        s4.δ = δ1
        # the first case is zero
        @test s4.coefficient(dmp, s4, 1) == 1.0
        s4.X = grad_2
        s4.δ = δ2
        @test s4.coefficient(dmp, s4, 2) == dot(grad_2, grad_2) / dot(grad_1, grad_1)

        dU = HagerZhangCoefficient()
        s5 = ConjugateGradientDescentState(
            M;
            p = x0,
            stopping_criterion = sC,
            stepsize = s,
            coefficient = dU,
            retraction_method = retr,
            vector_transport_method = vtm,
        )
        s5.X = grad_1
        s5.δ = δ1
        # the first case is zero
        @test s5.coefficient(dmp, s5, 1) == 0.0
        s5.X = grad_2
        s5.δ = δ2
        denom = dot(δ1, diff)
        ndiffsq = dot(diff, diff)
        @test s5.coefficient(dmp, s5, 2) ==
            dot(diff, grad_2) / denom - 2 * ndiffsq * dot(δ1, grad_2) / denom^2

        dU = HestenesStiefelCoefficient()
        s6 = ConjugateGradientDescentState(
            M;
            p = x0,
            stopping_criterion = sC,
            stepsize = s,
            coefficient = dU,
            retraction_method = retr,
            vector_transport_method = vtm,
        )
        s6.X = grad_1
        s6.δ = δ1
        @test s6.coefficient(dmp, s6, 1) == 0.0
        s6.X = grad_2
        s6.δ = δ2
        @test s6.coefficient(dmp, s6, 2) == dot(diff, grad_2) / dot(δ1, diff)

        dU = LiuStoreyCoefficient()
        s7 = ConjugateGradientDescentState(
            M;
            p = x0,
            stopping_criterion = sC,
            stepsize = s,
            coefficient = dU,
            retraction_method = retr,
            vector_transport_method = vtm,
        )
        s7.X = grad_1
        s7.δ = δ1
        @test s7.coefficient(dmp, s7, 1) == 0.0
        s7.X = grad_2
        s7.δ = δ2
        @test s7.coefficient(dmp, s7, 2) == -dot(grad_2, diff) / dot(δ1, grad_1)

        dU = PolakRibiereCoefficient()
        s8 = ConjugateGradientDescentState(
            M;
            p = x0,
            stopping_criterion = sC,
            stepsize = s,
            coefficient = dU,
            retraction_method = retr,
            vector_transport_method = vtm,
        )
        s8.X = grad_1
        s8.δ = δ1
        @test s8.coefficient(dmp, s8, 1) == 0.0
        s8.X = grad_2
        s8.δ = δ2
        @test s8.coefficient(dmp, s8, 2) == dot(grad_2, diff) / dot(grad_1, grad_1)

        dU = HybridCoefficient(PolakRibiereCoefficient(), FletcherReevesCoefficient())
        s9 = ConjugateGradientDescentState(
            M;
            p = x0,
            stopping_criterion = sC,
            stepsize = s,
            coefficient = dU,
            retraction_method = retr,
            vector_transport_method = vtm,
        )
        s9.X = grad_1
        s9.δ = δ1
        @test s9.coefficient(dmp, s9, 1) == 0.0 #max(0.0, min(0.0, 1.0))
        s9.X = grad_2
        s9.δ = δ2
        @test s9.coefficient(dmp, s9, 2) == max(
            0.0,
            min(dot(grad_2, diff) / dot(grad_1, grad_1), dot(grad_2, grad_2) / dot(grad_1, grad_1))
        )

    end
    @testset "Conjugate Gradient runs – Low Rank matrix approx" begin
        A = Diagonal([2.0, 1.1, 1.0])
        M = Sphere(size(A, 1) - 1)
        f(::Sphere, p) = p' * A * p
        grad_f(M, p) = project(M, p, 2 * A * p) # project the Euclidean gradient

        p0 = [2.0, 0.0, 2.0] / sqrt(8.0)
        x_opt = conjugate_gradient_descent(
            M,
            f,
            grad_f,
            p0;
            stepsize = ArmijoLinesearch(),
            coefficient = FletcherReevesCoefficient(),
            stopping_criterion = StopAfterIteration(15),
        )
        @test isapprox(f(M, x_opt), minimum(eigvals(A)); atol = 2.0 * 1.0e-2)
        @test isapprox(x_opt, eigvecs(A)[:, size(A, 1)]; atol = 3.0 * 1.0e-1)
        x_opt2 = conjugate_gradient_descent(
            M,
            f,
            grad_f,
            p0;
            stepsize = ArmijoLinesearch(),
            coefficient = FletcherReevesCoefficient(),
            stopping_criterion = StopAfterIteration(15),
            return_state = true,
        )
        @test get_solver_result(x_opt2) == x_opt
        @test startswith(
            repr(x_opt2),
            "# Solver state for `Manopt.jl`s Conjugate Gradient Descent Solver",
        )
        Random.seed!(23)
        x_opt3 = conjugate_gradient_descent(
            M,
            f,
            grad_f;
            stepsize = ArmijoLinesearch(),
            coefficient = FletcherReevesCoefficient(),
            stopping_criterion = StopAfterIteration(15),
        )
        @test isapprox(f(M, x_opt3), minimum(eigvals(A)); atol = 2.0 * 1.0e-2)
    end

    @testset "CG on complex manifolds" begin
        M = Euclidean(2; field = ℂ)
        A = [2 im; -im 2]
        fc(::Euclidean, p) = real(p' * A * p)
        grad_fc(::Euclidean, p) = 2 * A * p
        p0 = [2.0, 1 + im]
        # just one step as a test
        p1 = conjugate_gradient_descent(
            M,
            fc,
            grad_fc,
            p0;
            coefficient = FletcherReevesCoefficient(),
            stopping_criterion = StopAfterIteration(1),
        )
        @test fc(M, p1) ≤ fc(M, p0)
    end

    @testset "Euclidean Quadratic function test" begin
        M = Euclidean(2)
        A = [1.0 0; 0 0.1]
        f(M, p) = p' * A * p
        grad_f(M, p) = 2 * A * p
        p0 = [0.1; 1]
        struct CGStepsize <: Stepsize end
        function (cs::CGStepsize)(
                amp::AbstractManoptProblem,
                cgds::ConjugateGradientDescentState,
                i,
                args...;
                kwargs...,
            ) # for this example a closed form solution is known for the best step size
            M = get_manifold(amp)
            p = get_iterate(cgds)
            X = -get_gradient(amp, p)
            α = 0.5 * inner(M, p, X, cgds.δ) / inner(M, p, cgds.δ, A * cgds.δ)
            return α
        end
        # should be zero after 2 steps
        p1 = conjugate_gradient_descent(
            M,
            f,
            grad_f,
            p0;
            stepsize = CGStepsize(),
            stopping_criterion = StopAfterIteration(2),
        )
        p2 = copy(M, p0)
        conjugate_gradient_descent!(
            M,
            f,
            grad_f,
            p2;
            stepsize = CGStepsize(),
            stopping_criterion = StopAfterIteration(2),
        )
        @test norm(p1) ≈ 0 atol = 4 * 1.0e-16
        @test p1 == p2

        # Cubic stepsize should yield the exact stepsize for quadratic problems
        p3 = copy(M, p0)
        conjugate_gradient_descent!(
            M,
            f,
            grad_f,
            p3;
            stepsize = CubicBracketingLinesearch(; sufficient_curvature = 1.0e-4),
            stopping_criterion = StopAfterIteration(2),
        )
        @test norm(p1) ≈ 0 atol = 4 * 1.0e-16
        @test isapprox(M, p1, p3; atol = 5.0e-8)
    end

    @testset "CG on the Circle" begin
        M, f, grad_f, p0, p_star = Manopt.Test.Circle_mean_task()
        s = conjugate_gradient_descent(
            M, f, grad_f, p0; evaluation = InplaceEvaluation(), return_state = true
        )
        p = get_solver_result(s)[]
        @test f(M, p) < f(M, p0)
        @test isapprox(M, p, p_star; atol = 5.0e-8)
    end

    @testset "Restart Condition test" begin
        n = 4
        A = diagm(1:n)
        M = Sphere(n - 1)

        e_func(p) = dot(p, A * p)

        obj = ManifoldFirstOrderObjective(;
            cost = (M, p) -> e_func(p), gradient = (M, p) -> A * p - dot(p, A * p) * p
        )
        p0 = 0.5 * [1.0, 1.0, 1.0, 1.0]

        stopping_criterion = StopAfterIteration(30)
        get_stepsize() = Manopt.ArmijoLinesearchStepsize(
            M; initial_stepsize = 1.0, initial_guess = Manopt.ConstantInitialGuess(1.0)
        )

        p1 = conjugate_gradient_descent(
            M,
            obj,
            p0;
            restart_condition = NeverRestart(),
            stopping_criterion,
            stepsize = get_stepsize(),
        )

        p2 = conjugate_gradient_descent(
            M,
            obj,
            p0;
            restart_condition = RestartOnNonDescent(),
            stopping_criterion,
            stepsize = get_stepsize(),
        )

        p3 = conjugate_gradient_descent(
            M,
            obj,
            p0;
            restart_condition = RestartOnNonSufficientDescent(0.5),
            stopping_criterion,
            stepsize = get_stepsize(),
        )
        # sufficient descent should perform best, descent better than no restart
        @test e_func(p3) < e_func(p2) && e_func(p2) < e_func(p1)
        @test e_func(p3) ≈ 1 atol = 1.0e-3
    end

    @testset "Minimal eigenvalue calculation" begin

        n = 3
        A = diagm(1:3)
        M = Sphere(n - 1)
        obj = ManifoldFirstOrderObjective(;
            cost = (M, p) -> dot(p, A * p),
            gradient = (M, p) -> A * p - dot(p, A * p) * p
        )
        retraction_method = ProjectionRetraction()
        vector_transport_method = ProjectionTransport()
        stopping_criterion = StopAfterIteration(300) | StopWhenGradientNormLess(1.0e-10)
        p0 = 1 / sqrt(3) * [1.0, 1, 1]
        is_converged = false
        q1 = copy(p0)
        conjugate_gradient_descent!(
            M, obj, q1;
            retraction_method, vector_transport_method,
            stopping_criterion,
            coefficient = HybridCoefficient(M, FletcherReevesCoefficient(), PolakRibiereCoefficient(M; vector_transport_method)),
            stepsize = CubicBracketingLinesearch(
                M; retraction_method, vector_transport_method
            )
        )
        @test q1 ≈ [1, 0, 0] rtol = 1.0e-8

        q2 = copy(p0)
        conjugate_gradient_descent!(
            M, obj, q2;
            retraction_method, vector_transport_method,
            stopping_criterion,
            coefficient = HybridCoefficient(M, FletcherReevesCoefficient(), PolakRibiereCoefficient(M; vector_transport_method)),
            stepsize = ArmijoLinesearch(M)
        )
        @test q2 ≈ [1, 0, 0] rtol = 1.0e-7
    end
end
