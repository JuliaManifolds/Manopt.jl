using Manopt, Manifolds, Test, ManifoldDiff

@testset "The Proximal Gradient Method" begin
    M = Hyperbolic(2)
    p = [0.0, 0.0, 1.0]
    p0 = [1.0, 0.0, √2]
    pgms = ProximalGradientMethodState(
        M;
        p = p0,
        stepsize = Manopt.ProximalGradientMethodBacktrackingStepsize(
            M; initial_stepsize = 1.0, strategy = :convex
        ),
        stopping_criterion = StopAfterIteration(200),
    )
    @test get_iterate(pgms) == p0

    pgms.X = [1.0, 0.0, 0.0]
    @testset "Special Stopping Criterion" begin
        sc1 = StopWhenGradientMappingNormLess(1.0e-8)
        @test startswith(repr(sc1), "StopWhenGradientMappingNormLess(1.0e-8)")
        @test get_reason(sc1) == ""
        # Trigger manually
        sc1.at_iteration = 2
        @test length(get_reason(sc1)) > 0
    end
    @testset "Proximal Gradient Backtracking" begin
        pgb = Manopt.ProximalGradientMethodBacktrackingStepsize(M)
        @test get_initial_stepsize(pgb) == 1.0
        @test get_last_stepsize(pgb) == 1.0
        @test startswith(repr(pgb), "ProximalGradientMethodBacktrackingStepsize(;")
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
        # Check Fallbacks of Problem
        @test get_cost(mp, p) == 0.0
        @test get_cost_smooth(M, ob, p) == g(M, p)
        @test norm(M, p, get_gradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, 1.0, pgms.p)
        @test_throws MethodError get_proximal_map(mp, 1.0, pgms.p, 1)
        pgm = proximal_gradient_method(
            M,
            f,
            g,
            grad_g,
            p0;
            prox_nonsmooth = prox_h,
            stopping_criterion = StopAfterIteration(10),
            return_state = true,
            debug = [],
            stepsize = ProximalGradientMethodBacktracking(;
                initial_stepsize = 1.0, strategy = :convex
            ),
            sub_state = AllocatingEvaluation(),
        )
        p_star2 = get_solver_result(pgm)
        @test f(M, p_star2) <= f(M, p0)
        set_iterate!(pgm, M, p)
        @test get_iterate(pgm) == p
        @test pgm.last_stepsize ≤ 1.0
        st = Manopt.ProximalGradientMethodBacktrackingStepsize(
            M; initial_stepsize = 1.0, strategy = :convex, stop_when_stepsize_less = 2.0
        )

        @test st.warm_start_factor == 1.0
        @test st.last_stepsize == 1.0
        @test get_initial_stepsize(st) == 1.0
        @test st(mp, pgms, 1) == 1.0
        pr = prox_h(M, 1.0, p0)
        @test get_proximal_map(M, ob, 1.0, p0) == pr
        @test_throws DomainError Manopt.ProximalGradientMethodBacktrackingStepsize(
            M; strategy = :neither
        )
        @test_throws DomainError Manopt.ProximalGradientMethodBacktrackingStepsize(
            M; warm_start_factor = -1.0
        )

        @testset "Backtracking Warnings" begin
            dw1 = DebugWarnIfStepsizeCollapsed(:Once)
            @test repr(dw1) == "DebugWarnIfStepsizeCollapsed(Once, :Once)"
            pgms_warn = ProximalGradientMethodState(
                M;
                p = p0,
                stepsize = Manopt.ProximalGradientMethodBacktrackingStepsize(
                    M; initial_stepsize = 1.0, strategy = :convex, stop_when_stepsize_less = 10.0
                ),
                stopping_criterion = StopAfterIteration(200),
            )
            @test_logs (:warn,) (:warn,) dw1(mp, pgms_warn, 1)
            dw2 = DebugWarnIfStepsizeCollapsed(1.0, :Once)
            pgms_const = ProximalGradientMethodState(
                M;
                p = p0,
                stepsize = Manopt.ConstantStepsize(M, 1.0),
                stopping_criterion = StopAfterIteration(2),
            )
            # works normally does nothing on init and normally
            @test isnothing(dw2(mp, pgms_const, 0))
            @test isnothing(dw2(mp, pgms_const, 0))
            # but if we force a small step we warn
            pgms_const.stepsize.length = 0.5
            @test_logs (:warn,) (:warn,) dw2(mp, pgms_const, 1)
            # but also only once
            @test_nowarn dw2(mp, pgms_const, 2)
        end

        # Test subsolver with subgradient
        ∂h(M, q) = ManifoldDiff.subgrad_distance(M, p, q, 1; atol = 1.0e-8)
        sub_pgm = proximal_gradient_method(
            M,
            f,
            g,
            grad_g,
            p0;
            cost_nonsmooth = h,
            subgradient_nonsmooth = ∂h,
            stopping_criterion = StopAfterIteration(10),
        )
        @test_throws ErrorException proximal_gradient_method(M, f, g, grad_g, p0)
        pgnc = ProximalGradientNonsmoothCost(h, 0.1, p)
        pgng = ProximalGradientNonsmoothSubgradient(∂h, 0.1, p)
        @test Manopt.get_parameter(pgnc, :λ) == 0.1
        @test Manopt.get_parameter(pgnc, :proximity_point) == p
        @test Manopt.get_parameter(pgng, :λ) == 0.1
        @test Manopt.get_parameter(pgng, :proximity_point) == p

        # prox pass through with dummy objective deco
        dob = Manopt.Test.DummyDecoratedObjective(ob)
        @test get_proximal_map(M, ob, 0.1, p) == get_proximal_map(M, dob, 0.1, p)
        q1 = copy(M, p)
        q2 = copy(M, p)
        get_proximal_map!(M, q1, ob, 0.1, p)
        get_proximal_map!(M, q2, dob, 0.1, p)
        @test q1 == q2

        # Acceleration
        pgma = Manopt.ProximalGradientMethodAcceleration(M; p = copy(M, p0))
        # Since this is experimental, we for now just check that it does not error,
        # but we can not yet verify the result
        pgma(mp, pgms, 1)
        @test startswith(repr(pgma), "ProximalGradientMethodAcceleration with parameters\n")
    end
    @testset "Inplace Evaluation" begin
        g(M, q) = distance(M, q, p)^2
        function grad_g!(M, X, q)
            X .= -2 * log(M, q, p)
            return X
        end
        h(M, q) = distance(M, q, p)
        prox_h!(M, a, λ, q) = ManifoldDiff.prox_distance!(M, a, λ, p, q, 1)
        f(M, q) = g(M, q) + h(M, q)
        ieob = ManifoldProximalGradientObjective(
            f, g, grad_g!, prox_h!; evaluation = InplaceEvaluation()
        )
        mp = DefaultManoptProblem(M, ieob)
        X = zero_vector(M, p)
        Y = get_gradient(mp, p)
        get_gradient!(mp, X, p)
        @test isapprox(M, p, X, Y)
        # Test Fallbacks of Problem
        @test get_cost(mp, p) == 0.0
        @test norm(M, p, get_gradient(mp, p)) == 0
        @test_throws MethodError get_gradient(mp, 1.0, pgms.p)
        @test isapprox(M, get_proximal_map(mp, 1.0, pgms.p), pgms.p)
        sr = solve!(mp, pgms)
        xHat = get_solver_result(sr)
        s2 = proximal_gradient_method(
            M,
            f,
            g,
            grad_g!,
            copy(p0);
            prox_nonsmooth = prox_h!,
            stepsize = ProximalGradientMethodBacktracking(;
                initial_stepsize = 1.0, strategy = :convex
            ),
            stopping_criterion = StopAfterIteration(200),
            evaluation = InplaceEvaluation(),
            return_state = true,
            debug = [],
        )
        p_star2 = get_solver_result(s2)
        @test f(M, p_star2) <= f(M, p0)
        a = copy(p0)
        prox_h!(M, a, 1.0, p)
        @test get_proximal_map(M, ieob, 1.0, p) == a
        p2 = copy(M, p0)
        proximal_gradient_method!(
            M,
            f,
            g,
            grad_g!,
            p2;
            prox_nonsmooth = prox_h!,
            stepsize = ProximalGradientMethodBacktracking(;
                initial_stepsize = 1.0, strategy = :convex
            ),
            stopping_criterion = StopAfterIteration(200),
            evaluation = InplaceEvaluation(),
            return_state = true,
            debug = [],
        )
        @test isapprox(M, p2, p_star2)
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
                ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 2; atol = 1.0e-8),
        )
        h(M, p) = 0
        prox_h(M, λ, p) = p
        p0 = p1
        pbm_s = proximal_gradient_method(
            M, f, g, grad_g;
            prox_nonsmooth = prox_h,
            inverse_retraction_method = ProjectionInverseRetraction(),
            stepsize = ProximalGradientMethodBacktracking(;
                initial_stepsize = 1.0,
                strategy = :convex
            ),
            return_state = true
        )
        @test startswith(
            Manopt.status_summary(pbm_s; inline = false),
            "# Solver state for `Manopt.jl`s Proximal Gradient Method\n"
        )
        q = get_solver_result(pbm_s)
        # with default parameters for both median and proximal gradient, this is not very precise
        m = mean(M, data)
        @test distance(M, q, m) < 2 * 1.0e-2
        p_size = copy(p0)
        function grad_g!(M, X, p)
            X = sum(
                1 / length(data) *
                    ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 2; atol = 1.0e-8),
            )
            return X
        end
        function prox_h!(M, a, λ, p)
            copyto!(M, a, p)
            return a
        end
    end
end
