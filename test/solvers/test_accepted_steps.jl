using LinearAlgebra, Manifolds, Manopt, QuadraticModels, Random, RipQP, Test

# For every iteration, record whether the solver reports its step as accepted, see
# `has_accepted_step`, and whether the iterate actually changed
function accepted_step_recorder(M)
    accepted, moved = Bool[], Bool[]
    q = Ref{Any}(nothing)
    callbacks = [
        :BeforeStep => (mp, st, k) -> (q[] = copy(M, get_iterate(st)); nothing),
        :Step => function (mp, st, k)
            push!(accepted, Manopt.has_accepted_step(st))
            push!(moved, q[] != get_iterate(st))
            return nothing
        end,
    ]
    return accepted, moved, callbacks
end

@testset "has_accepted_step of solvers that reject steps" begin
    M = Euclidean(2)
    # Rosenbrock, from its usual starting point
    f(M, p) = 100 * (p[2] - p[1]^2)^2 + (1 - p[1])^2
    grad_f(M, p) = [-400 * p[1] * (p[2] - p[1]^2) - 2 * (1 - p[1]), 200 * (p[2] - p[1]^2)]
    function Hess_f(M, p, X)
        H = [1200 * p[1]^2 - 400 * p[2] + 2 -400 * p[1]; -400 * p[1] 200]
        return H * X
    end
    p0 = [-1.2, 1.0]
    # a step accepted with an unchanged iterate would break this test, but every rejected
    # step has to keep the iterate
    function check(accepted, moved)
        @test accepted == moved
        @test any(accepted) && !all(accepted)
        return nothing
    end
    @testset "default" begin
        # solvers that do not reject steps accept them all, also for decorated states
        s = GradientDescentState(M; p = copy(p0))
        @test Manopt.has_accepted_step(s)
        @test Manopt.has_accepted_step(DebugSolverState(s, DebugIteration()))
    end
    @testset "trust regions" begin
        accepted, moved, callbacks = accepted_step_recorder(M)
        trust_regions(
            M, f, grad_f, Hess_f, p0;
            trust_region_radius = 10.0, max_trust_region_radius = 10.0,
            stopping_criterion = StopAfterIteration(30), callbacks = callbacks,
        )
        check(accepted, moved)
    end
    @testset "adaptive regularization with cubics" begin
        accepted, moved, callbacks = accepted_step_recorder(M)
        adaptive_regularization_with_cubics(
            M, f, grad_f, Hess_f, p0;
            σ = 1.0e-3, stopping_criterion = StopAfterIteration(30), callbacks = callbacks,
        )
        check(accepted, moved)
    end
    @testset "Levenberg-Marquardt" begin
        accepted, moved, callbacks = accepted_step_recorder(M)
        F(M, p) = [10 * (p[2] - p[1]^2), 1 - p[1]]
        JF(M, p) = [-20 * p[1] 10.0; -1.0 0.0]
        LevenbergMarquardt(
            M, F, JF, p0;
            damping_term_min = 1.0e-6, stopping_criterion = StopAfterIteration(30), callbacks = callbacks,
        )
        check(accepted, moved)
    end
    H = Hyperbolic(4)
    q = [0.0, 0.0, 0.0, 0.0, 1.0]
    q0 = exp(H, q, [1.0, 0.0, 0.0, 0.0, 0.0])
    fd, ∂fd, _ = Manopt.Test.distance_task(H, q)
    @testset "convex bundle method" begin
        accepted, moved, callbacks = accepted_step_recorder(H)
        convex_bundle_method(
            H, fd, ∂fd, copy(H, q0);
            k_max = 0.0, k_min = -1.0, diameter = 10.0,
            stopping_criterion = StopAfterIteration(30), callbacks = callbacks,
        )
        check(accepted, moved)
    end
    @testset "proximal bundle method" begin
        accepted, moved, callbacks = accepted_step_recorder(H)
        proximal_bundle_method(
            H, fd, ∂fd, copy(H, q0); stopping_criterion = StopAfterIteration(30), callbacks = callbacks,
        )
        check(accepted, moved)
    end
    @testset "gradient sampling" begin
        Random.seed!(42)
        accepted, moved, callbacks = accepted_step_recorder(H)
        gradient_sampling(
            H, fd, ∂fd, copy(H, q0); stopping_criterion = StopAfterIteration(30), callbacks = callbacks,
        )
        check(accepted, moved)
    end
    @testset "mesh adaptive direct search" begin
        Random.seed!(42)
        accepted, moved, callbacks = accepted_step_recorder(M)
        mesh_adaptive_direct_search(
            M, f, p0; stopping_criterion = StopAfterIteration(30), callbacks = callbacks,
        )
        check(accepted, moved)
    end
end
