using Manifolds, ManifoldsBase, Manopt, Test, Random
using LinearAlgebra: I, tr, Symmetric

include("../utils/example_tasks.jl")

@testset "Adaptive Regularization with Cubics" begin
    Random.seed!(42)
    n = 8
    k = 3
    A = Symmetric(randn(n, n))
    M = Grassmann(n, k)

    f(M, p) = -0.5 * tr(p' * A * p)
    grad_f(M, p) = -A * p + p * (p' * A * p)
    Hess_f(M, p, X) = -A * X + p * p' * A * X + X * p' * A * p

    p0 = Matrix{Float64}(I, n, n)[:, 1:k]
    M2 = TangentSpace(M, copy(M, p0))
    mho = ManifoldHessianObjective(f, grad_f, Hess_f)
    arcmo = AdaptiveRagularizationWithCubicsModelObjective(mho)

    @testset "State and repr" begin
        # if we neither provide a problem nor an objective, we expect an error
        @test_throws ErrorException AdaptiveRegularizationState(ℝ^2)

        arcs = AdaptiveRegularizationState(
            M,
            p0;
            sub_problem=DefaultManoptProblem(M2, arcmo),
            sub_state=GradientDescentState(M2, zero_vector(M, p0)),
        )
        @test startswith(
            repr(arcs),
            "# Solver state for `Manopt.jl`s Adaptive Regularization with Cubics (ARC)",
        )
        p1 = rand(M)
        X1 = rand(M; vector_at=p1)
        set_iterate!(arcs, p1)
        @test arcs.p == p1
        set_gradient!(arcs, X1)
        @test arcs.X == X1
        arcs2 = AdaptiveRegularizationState(
            M,
            p0;
            sub_objective=arcmo,
            sub_state=LanczosState(M2; maxIterLanczos=1),
            stopping_criterion=StopWhenAllLanczosVectorsUsed(1),
        )
        #add a fake Lanczos
        push!(arcs2.sub_state.Lanczos_vectors, X1)
        # we reached 1 Lanczos
        @test stop_solver!(arcs2.sub_problem, arcs2.sub_state, 1)
        @test stop_solver!(arcs2.sub_problem, arcs2, 1)

        arcs3 = AdaptiveRegularizationState(
            M, p0; sub_objective=arcmo, sub_state=LanczosState(M2; maxIterLanczos=2)
        )
        #add a fake Lanczos
        initialize_solver!(arcs3.sub_problem, arcs3.sub_state)
        push!(arcs3.sub_state.Lanczos_vectors, copy(M, p1, X1))
        step_solver!(arcs3.sub_problem, arcs3.sub_state, 2) # to introduce a random new one
        # test orthogonality of the new 2 ones
        @test isapprox(
            inner(
                M,
                p1,
                arcs3.sub_state.Lanczos_vectors[1],
                arcs3.sub_state.Lanczos_vectors[2],
            ),
            0.0,
            atol=1e-14,
        )
        # a second that copies
        arcs4 = AdaptiveRegularizationState(
            M, p0; sub_objective=arcmo, sub_state=LanczosState(M2; maxIterLanczos=2)
        )
        #add a fake Lanczos
        push!(arcs4.sub_state.Lanczos_vectors, copy(M, p1, X1))
        push!(arcs4.sub_state.Lanczos_vectors, copy(M, p1, X1))
        step_solver!(arcs4.sub_problem, arcs4.sub_state, 2) # to introduce a random new one but cupy to 2
        # test orthogonality of the new 2 ones
        @test isapprox(
            inner(
                M,
                p1,
                arcs4.sub_state.Lanczos_vectors[1],
                arcs4.sub_state.Lanczos_vectors[2],
            ),
            0.0,
            atol=1e-14,
        )

        st1 = StopWhenFirstOrderProgress(0.5)
        @test startswith(repr(st1), "StopWhenFirstOrderProgress(0.5)\n")
        @test Manopt.indicates_convergence(st1)
        st2 = StopWhenAllLanczosVectorsUsed(2)
        @test startswith(repr(st2), "StopWhenAllLanczosVectorsUsed(2)\n")
        @test !Manopt.indicates_convergence(st2)
        @test startswith(
            repr(arcs2.sub_state), "# Solver state for `Manopt.jl`s Lanczos Iteration\n"
        )

        f1(M, p) = p
        f1!(M, q, p) = copyto!(M, q, p)
        r = copy(M, p1)
        Manopt.solve_arc_subproblem!(M, r, f1, AllocatingEvaluation(), p0)
        @test r == p0
        r = copy(M, p1)
        Manopt.solve_arc_subproblem!(M, r, f1!, InplaceEvaluation(), p0)
        @test r == p0
    end

    @testset "A few solver runs" begin
        p1 = adaptive_regularization_with_cubics(
            M, f, grad_f, Hess_f, p0; θ=0.5, σ=100.0, retraction_method=PolarRetraction()
        )
        # Second run with random p0
        Random.seed!(42)
        p2 = adaptive_regularization_with_cubics(
            M, f, grad_f, Hess_f; θ=0.5, σ=100.0, retraction_method=PolarRetraction()
        )
        @test isapprox(M, p1, p2)
        # Third with approximate Hessian
        p3 = adaptive_regularization_with_cubics(
            M,
            f,
            grad_f,
            p0;
            θ=0.5,
            σ=100.0,
            retraction_method=PolarRetraction(),
            debug=[:Iteration, :Cost, :σ, :ρ_denonimator, "\n"],
        )
        @test isapprox(M, p1, p3)
        # Fourth with approximate Hessian _and_ random point
        Random.seed!(36)
        p4 = adaptive_regularization_with_cubics(
            M, f, grad_f; θ=0.5, σ=100.0, retraction_method=PolarRetraction()
        )
        @test isapprox(M, p1, p4)
        # with a large η1 to trigger the bad model case once
        p5 = adaptive_regularization_with_cubics(
            M,
            f,
            grad_f,
            Hess_f;
            θ=0.5,
            σ=100.0,
            η1=0.89,
            retraction_method=PolarRetraction(),
        )
        @test isapprox(M, p1, p5)

        # in place
        q1 = copy(M, p0)
        adaptive_regularization_with_cubics!(
            M, f, grad_f, Hess_f, q1; θ=0.5, σ=100.0, retraction_method=PolarRetraction()
        )
        @test isapprox(M, p1, q1)
        # in place with approx Hess
        q2 = copy(M, p0)
        adaptive_regularization_with_cubics!(
            M, f, grad_f, q2; θ=0.5, σ=100.0, retraction_method=PolarRetraction()
        )
        @test isapprox(M, p1, q2)

        # test both inplace and allocating variants of grad_g
        X0 = grad_f(M, p0)
        X1 = get_gradient(M2, arcmo, X0)
        X2 = zero_vector(M, p0)
        get_gradient!(M2, X2, arcmo, X0)
        @test isapprox(M, p0, X1, X2)

        sub_problem = DefaultManoptProblem(M2, arcmo)
        sub_state = GradientDescentState(
            M2,
            zero_vector(M, p0);
            stopping_criterion=StopAfterIteration(500) |
                               StopWhenGradientNormLess(1e-11) |
                               StopWhenFirstOrderProgress(0.1),
        )
        q3 = copy(M, p0)
        adaptive_regularization_with_cubics!(
            M,
            mho,
            q3;
            θ=0.5,
            σ=100.0,
            retraction_method=PolarRetraction(),
            sub_problem=sub_problem,
            sub_state=sub_state,
            return_objective=true,
            return_state=true,
        )
        @test isapprox(M, p1, q3)
    end
    @testset "A short solver run on the circle" begin
        Mc, fc, grad_fc, pc0, pc_star = Circle_mean_task()
        hess_fc(Mc, p, X) = 1.0
        p0 = 0.2
        p1 = adaptive_regularization_with_cubics(
            Mc, fc, grad_fc, hess_fc, p0; θ=0.5, σ=100.0
        )
        @test fc(Mc, p0) > fc(Mc, p1)
        p2 = adaptive_regularization_with_cubics(
            Mc, fc, grad_fc, hess_fc, p0; θ=0.5, σ=100.0, evaluation=InplaceEvaluation()
        )
        @test fc(Mc, p0) > fc(Mc, p2)
    end
end
