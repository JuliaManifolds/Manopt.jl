s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using Manopt, Manifolds, Test, ManifoldDiff, ManoptTestSuite

@testset "The Proximal Gradient Method" begin
    M = Hyperbolic(2)
    p = [0.0, 0.0, 1.0]
    p0 = [0.0, 0.0, -1.0]
    pgms = ProximalGradientMethodState(M; p=p0, stopping_criterion=StopAfterIteration(200))
    @test get_iterate(pgms) == p0

    pgms.X = [1.0, 0.0, 0.0]
    @testset "Special Stopping Criterion" begin
        sc1 = StopWhenGradientMappingNormLess(1e-8)
        @test startswith(repr(sc1), "StopWhenGradientMappingNormLess(1.0e-8)\n")
        @test get_reason(sc1) == ""
        # Trigger manually
        sc1.at_iteration = 2
        @test length(get_reason(sc1)) > 0
    end
    @testset "Allocating Evaluation" begin
        g(M, q) = distance(M, q, p)^2
        grad_g(M, q) = -2 * log(M, q, p)
        h(M, q) = distance(M, q, p)
        prox_h(M, λ, q) = ManifoldDiff.prox_distance(M, λ, p, q, 1)
        f(M, q) = g(M, q) + h(M, q)
        ob = ManifoldProximalGradientObjective(f, g, grad_g, prox_h)
        mp = DefaultManoptProblem(M, ob)
        X = zero_vector(M, p)
        Y = get_gradient(mp, p)
        get_gradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        oR = solve!(mp, pgms)
        xHat = get_solver_result(oR)
        # Check Fallbacks of Problem
        @test get_cost(mp, p) == 0.0
        @test get_cost_smooth(M, ob, p) == g(M, p)
        @test norm(M, p, get_gradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, 1.0, pgms.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, pgms.p, 1)
        pgms2 = proximal_gradient_method(
            M,
            f,
            g,
            grad_g,
            prox_h,
            p0;
            stopping_criterion=StopAfterIteration(10),
            return_state=true,
            debug=[],
            stepsize=ProximalGradientMethodBacktracking(;
                initial_stepsize=1.0, strategy=:convex
            ),
        )
        p_star2 = get_solver_result(pgms2)
        @test f(M, p_star2) <= f(M, p0)
        set_iterate!(pgms2, M, p)
        @test get_iterate(pgms2) == p
        @test pgms2.last_stepsize ≤ 1.0
        st = Manopt.ProximalGradientMethodBacktrackingStepsize(
            M; initial_stepsize=1.0, strategy=:convex, stop_when_stepsize_less=2.0
        )
        @test_logs (:warn,) st(mp, pgms, 1)

        # prox pass through with dummy objective deco
        dob = ManoptTestSuite.DummyDecoratedObjective(ob)
        @test get_proximal_map(M, ob, 0.1, p) == get_proximal_map(M, dob, 0.1, p)
        q1 = copy(M, p)
        q2 = copy(M, p)
        get_proximal_map!(M, q1, ob, 0.1, p)
        get_proximal_map!(M, q2, dob, 0.1, p)
        @test q1 == q2
    end
    @testset "Mutating Subgradient" begin
        g(M, q) = distance(M, q, p)^2
        function grad_g!(M, X, q)
            X .= -2 * log(M, q, p)
            return X
        end
        h(M, q) = distance(M, q, p)
        prox_h!(M, a, λ, q) = ManifoldDiff.prox_distance!(M, a, λ, p, q, 1)
        f(M, q) = g(M, q) + h(M, q)
        bmom = ManifoldProximalGradientObjective(
            f, g, grad_g!, prox_h!; evaluation=InplaceEvaluation()
        )
        mp = DefaultManoptProblem(M, bmom)
        X = zero_vector(M, p)
        Y = get_gradient(mp, p)
        get_gradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        sr = solve!(mp, pgms)
        xHat = get_solver_result(sr)
        # Test Fallbacks of Problem
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_gradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, 1.0, pgms.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, pgms.p, 1)
        s2 = proximal_gradient_method(
            M,
            f,
            g,
            grad_g!,
            prox_h!,
            copy(p0);
            stopping_criterion=StopAfterIteration(200),
            evaluation=InplaceEvaluation(),
            return_state=true,
            debug=[],
        )
        p_star2 = get_solver_result(s2)
        @test f(M, p_star2) <= f(M, p0)
    end
    @testset "A mean run" begin
        M = Sphere(2)
        p1 = [1.0, 0.0, 0.0]
        p2 = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
        p3 = 1 / sqrt(2) .* [1.0, 0.0, 1.0]
        data = [p1, p2, p3]
        f(M, p) = sum(1 / 2length(data) * distance(M, p, di)^2 for di in data)
        g(M, p) = f(M, p)
        grad_g(M, p) = sum(
            1 / length(data) *
            ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 2; atol=1e-8),
        )
        h(M, p) = 0
        prox_h(M, λ, p) = p
        p0 = p1
        pbm_s = proximal_gradient_method(M, f, g, grad_g, prox_h; return_state=true)
        @test startswith(
            repr(pbm_s), "# Solver state for `Manopt.jl`s Proximal Gradient Method\n"
        )
        q = get_solver_result(pbm_s)
        # with default parameters for both median and proximal gradient, this is not very precise
        m = mean(M, data)
        @test distance(M, q, m) < 2 * 1e-2
        # test access functions
        # @test get_iterate(pbm_s) == q
        # @test norm(M, q, get_gradient(pbm_s)) < 1e-4
        # Test gradient size and in-place
        p_size = copy(p0)
        function grad_g!(M, X, p)
            X = sum(
                1 / length(data) *
                ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 2; atol=1e-8),
            )
            return X
        end
        function prox_h!(M, a, λ, p)
            copyto!(M, a, p)
            return a
        end
        proximal_gradient_method!(
            M,
            f,
            g,
            grad_g!,
            prox_h!,
            p_size;
            evaluation=InplaceEvaluation(),
            stopping_criterion=StopAfterIteration(200),
        )
    end
end
