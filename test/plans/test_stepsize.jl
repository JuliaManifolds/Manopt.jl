using ManifoldsBase, Manopt, Manifolds, Test

@testset "Initial stepsize" begin
    @testset "Hager-Zhang" begin
        M = ManifoldsBase.DefaultManifold(2)
        hzi = Manopt.HagerZhangInitialGuess()
        hzi_nq = Manopt.HagerZhangInitialGuess{Float64}(; constant_guess = 12.0, quadstep = false)

        f(M, p) = sum((p .- 1) .^ 2)
        grad_f(M, p) = 2 .* (p .- 1)
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))

        # case I0 (a)
        p1 = [2.0, 2.0]
        X1 = grad_f(M, p1)

        gds1 = GradientDescentState(M; p = p1)
        η1 = -X1
        @test hzi(dmp, gds1, 1, NaN, η1) ≈ hzi.ψ0 * Manopt.default_point_distance(M, p1) / Manopt.default_vector_norm(M, p1, η1)
        # case I0 (b)
        p2 = [0.0, 0.0]
        X2 = grad_f(M, p2)

        gds2 = GradientDescentState(M; p = p2)
        η2 = -X2
        @test hzi(dmp, gds2, 1, NaN, η2) ≈ hzi.ψ0 * abs(f(M, p2)) / norm(M, p2, η2)^2

        # case I0 (c)
        f2(M, p) = sum((p .- 1) .^ 2) - 2
        grad_f2(M, p) = 2 .* (p .- 1)
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f2, grad_f2))

        p3 = [0.0, 0.0]
        X3 = grad_f2(M, p3)

        gds3 = GradientDescentState(M; p = p3)
        η3 = -X3
        @test hzi(dmp, gds3, 1, NaN, η3) ≈ 1.0

        # case I0, explicit guess
        @test hzi_nq(dmp, gds3, 1, NaN, η3) ≈ hzi_nq.constant_guess

        # case I1
        @test hzi(dmp, gds3, 2, 1.0, η3) ≈ 0.5

        # case I2
        @test hzi_nq(dmp, gds3, 2, 41.0, η3) ≈ hzi.ψ2 * 41.0

        # sphere
        MS = Sphere(1)
        f3(M, p) = 100 * sum((p .- 1) .^ 2)
        grad_f3(M, p) = project(M, p, 200 .* (p .- 1))
        dmp = DefaultManoptProblem(MS, ManifoldGradientObjective(f3, grad_f3))
        p4 = [0.0, 1.0]
        X4 = grad_f3(MS, p4)

        gds = GradientDescentState(MS; p = p4)
        η4 = -X4
        @test hzi(dmp, gds, 1, NaN, η4) ≈ 2.5e-5

        # some defaults
        @test Manopt.default_point_distance(Euclidean(2), p1) == 2.0
        @test Manopt.default_vector_norm(Euclidean(2), p1, X1) == 2.0
    end
end


@testset "Stepsize" begin
    M = ManifoldsBase.DefaultManifold(2)
    @test Manopt.get_message(Manopt.ConstantStepsize(M, 1.0)) == ""
    s = Manopt.ArmijoLinesearchStepsize(Euclidean())
    @test startswith(repr(s), "ArmijoLinesearch(;")
    s_stat = Manopt.status_summary(s)
    @test startswith(s_stat, "ArmijoLinesearch(;")
    @test endswith(s_stat, "of 1.0")
    @test Manopt.get_message(s) == ""

    s2 = NonmonotoneLinesearch()(M)
    @test startswith(repr(s2), "NonmonotoneLinesearch(;")
    @test Manopt.get_message(s2) == ""

    s3 = WolfePowellBinaryLinesearch()(M)
    @test Manopt.get_message(s3) == ""
    @test startswith(repr(s3), "WolfePowellBinaryLinesearch(;")
    @test get_last_stepsize(s3) == 0.0
    # no stepsize yet so `repr` and summary are the same
    @test repr(s3) == Manopt.status_summary(s3)
    s4 = WolfePowellLinesearch()(M)
    @test startswith(repr(s4), "WolfePowellLinesearch(;")
    # no stepsize yet so `repr` and summary are the same
    @test repr(s4) == Manopt.status_summary(s4)
    @test Manopt.get_message(s4) == ""
    @testset "Armijo setter / getters" begin
        # Check that the passdowns work, though; since the defaults are functions, they return nothing
        @test isnothing(Manopt.get_parameter(s, :IncreaseCondition, :Dummy))
        @test isnothing(Manopt.get_parameter(s, :DecreaseCondition, :Dummy))
        @test Manopt.set_parameter!(s, :IncreaseCondition, :Dummy, 1) == s #setters return s
        @test Manopt.set_parameter!(s, :DecreaseCondition, :Dummy, 1) == s
    end
    @testset "Linesearch safeguards" begin
        M = Euclidean(2)
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = sum(2 .* p)
        p = [2.0, 2.0]
        msgs = (;
            non_descent_direction = Manopt.StepsizeMessage{Float64, Float64}(),
            stop_decreasing = Manopt.StepsizeMessage{Int, Float64}(),
            stop_increasing = Manopt.StepsizeMessage{Int, Float64}(),
            stepsize_less = Manopt.StepsizeMessage{Float64, Float64}(),
            stepsize_exceeds = Manopt.StepsizeMessage{Float64, Float64}(),
        )
        s1 = Manopt.linesearch_backtrack(
            M, f, p, 1.0, 1.0, 0.99, -grad_f(M, p); gradient = grad_f(M, p), stop_decreasing_at_step = 10,
            report_messages_in = msgs,
        )
        @test msgs[:stop_decreasing].at_iteration == 10
        s2 = Manopt.linesearch_backtrack(
            M, f, p, 1.0, 1.0, 0.5, grad_f(M, p); gradient = grad_f(M, p), retraction_method = ExponentialRetraction(),
            report_messages_in = msgs,
        )
        @test msgs[:non_descent_direction].at_iteration == 0
        s3 = Manopt.linesearch_backtrack(
            M, f, p, 1.0, 1.0, 0.5, -grad_f(M, p); gradient = grad_f(M, p), stop_when_stepsize_less = 0.75, report_messages_in = msgs
        )
        @test msgs[:stepsize_less].at_iteration == 1
        # cheating for increase
        s4 = Manopt.linesearch_backtrack(
            M, f, p, 1.0e-12, 0, 0.5, -grad_f(M, p); gradient = grad_f(M, p), stop_when_stepsize_exceeds = 0.1, report_messages_in = msgs
        )
        @test msgs[:stepsize_exceeds].at_iteration > 0 # or 37
        s5 = Manopt.linesearch_backtrack(
            M, f, p, 1.0e-12, 0, 0.5, -grad_f(M, p); gradient = grad_f(M, p), stop_increasing_at_step = 1, report_messages_in = msgs
        )
        @test msgs[:stop_increasing].at_iteration == 1
    end
    @testset "Adaptive WN Gradient" begin
        # Build a dummy function and gradient
        f(M, p) = 0
        grad_f(M, p) = [0.0, 0.75, 0.0] # valid, since only north pole used
        M = Sphere(2)
        p = [1.0, 0.0, 0.0]
        mgo = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, mgo)
        s = AdaptiveWNGradient(; gradient_reduction = 0.5, count_threshold = 2)(M)
        gds = GradientDescentState(M; p = p)
        @test get_initial_stepsize(s) == 1.0
        @test get_last_stepsize(s) == 1.0
        @test s(mp, gds, 0) == 1.0
        @test s(mp, gds, 1) == 0.64 # running into the last case
        @test s.count == 0 # unchanged
        @test s.weight == 0.75 # unchanged
        # tweak bound
        s.gradient_reduction = 10.0
        @test s(mp, gds, 2) ≈ 0.5201560468140443 # running into the last case
        @test s.count == 1 # running into case 2
        @test s(mp, gds, 3) ≈ 3.1209362808842656
        @test s.count == 0 # was reset
        @test s.weight == 0.75 # also reset to orig
        @test startswith(repr(s), "AdaptiveWNGradient(;\n  ")
    end
    @testset "Absolute stepsizes" begin
        M = ManifoldsBase.DefaultManifold(2)
        # Build a dummy function and gradient
        f(M, p) = 0
        grad_f(M, p) = [0.0, 0.75, 0.0] # valid, since only north pole used
        M = Sphere(2)
        p = [1.0, 0.0, 0.0]
        mgo = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, mgo)
        gds = GradientDescentState(M; p = p)
        abs_dec_step = Manopt.DecreasingStepsize(
            M; length = 10.0, factor = 1.0, subtrahend = 0.0, exponent = 1.0, type = :absolute
        )
        solve!(mp, gds)
        @test abs_dec_step(mp, gds, 1) ==
            10.0 / norm(get_manifold(mp), get_iterate(gds), get_gradient(gds))
        abs_const_step = Manopt.ConstantStepsize(M, 1.0; type = :absolute)
        @test abs_const_step(mp, gds, 1) ==
            1.0 / norm(get_manifold(mp), get_iterate(gds), get_gradient(gds))
    end
    @testset "Polyak Stepsize" begin
        M = Euclidean(2)
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 .* p
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        p = [2.0, 2.0]
        X = grad_f(M, p)
        sgs = SubGradientMethodState(M; p = p)
        ps = Polyak()()
        @test repr(ps) ==
            "Polyak()\nA stepsize with keyword parameters\n   * initial_cost_estimate = 0.0\n"
        @test ps(dmp, sgs, 1) == (f(M, p) - 0 + 1) / (norm(M, p, X)^2)
    end
    @testset "CubicBracketing Stepsize" begin
        M = Euclidean(2)
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 .* p
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        p = [1.0, 2.0]
        X = grad_f(M, p)
        gs = GradientDescentState(M; p = p, X = grad_f(M, p))
        clbs = CubicBracketingLinesearch()(M)
        @test startswith(repr(clbs), "CubicBracketingLinesearch(;")
        @test startswith(Manopt.status_summary(clbs), repr(clbs))
        @test clbs(dmp, gs, 1) ≈ 0.5 atol = 4 * 1.0e-8

        #edge cases of interval bracketing
        a, b, τ = 0, 1, 0.25
        @test Manopt.cubic_stepsize_update_step(a, b, 0.5, τ) == 0.5
        @test Manopt.cubic_stepsize_update_step(a, b, a, τ) == a + τ
        @test Manopt.cubic_stepsize_update_step(a, b, b, τ) == b - τ


        # check x^3 - 3x; local min at x = 1
        a = Manopt.UnivariateTriple(0.0, 0.0, -3.0)
        b = Manopt.UnivariateTriple(2.0, 2.0, 9.0)
        @test Manopt.cubic_polynomial_argmin(a, b) ≈ 1.0 rtol = 1.0e-12

        # test if DomainError is thrown
        c = Manopt.UnivariateTriple(3.0, 0.0, 0.0)
        @test_throws DomainError Manopt.update_bracket(a, b, c)

        # test (R3)
        c = Manopt.UnivariateTriple(1.0, 1.0, 1.0)
        @test Manopt.update_bracket(a, b, c) == (a, c)

        # test (R4)
        c = Manopt.UnivariateTriple(1.0, -2.0, -1.0)
        @test Manopt.update_bracket(a, b, c) == (c, b)

        c = Manopt.UnivariateTriple(1.0, -2.0, 1.0)
        @test Manopt.update_bracket(a, b, c) == (c, a)

        #test (R5)
        c = Manopt.UnivariateTriple(1.0, 0.0, 1.0)
        @test Manopt.update_bracket(a, b, c) == (c, a)

        c = Manopt.UnivariateTriple(1.0, 0.0, -1.0)
        @test Manopt.update_bracket(a, b, c) == (a, c)

        a = Manopt.UnivariateTriple(0.0, 0.0, 0.0)
        @test Manopt.update_bracket(a, b, c) == (c, b)

        # test secant
        @test Manopt.secant(a, b) == (a.t * b.df - b.t * a.df) / (b.df - a.df)
    end
    @testset "CubicBracketingStepsize force Hybrid" begin
        # test hybrid intervention for edge case
        M = Euclidean(1)
        f(M, p) = sum(p .^ 6)
        grad_f(M, p) = 6 * p .^ 5
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        p = [-1.0]
        X = grad_f(M, p)
        gs = GradientDescentState(M; p = p, X = grad_f(M, p))
        clbs = CubicBracketingLinesearch(; sufficient_curvature = 1.0e-16, min_bracket_width = 0.0, initial_stepsize = 0.5)(M)
        @test clbs(dmp, gs, 1) ≈ 1 / 6 atol = 5.0e-4
    end
    @testset "secant numerical stability" begin
        # Large offset, small interval
        a = 1.0e7
        b = a + 1.0e-6

        # Choose derivatives that differ slightly
        ga = 1.0
        gb = nextfloat(ga)   # smallest representable difference

        # minimizer using affine formula
        x_ref = a - ga * (b - a) / (gb - ga)

        err_secant = abs(
            Manopt.secant(
                Manopt.UnivariateTriple(a, 0.0, ga),
                Manopt.UnivariateTriple(b, 0.0, gb)
            ) - x_ref
        )
        @test err_secant < 1.0e-6
    end
    @testset "HagerZhang Linesearch Stepsize" begin
        M = Euclidean(2)
        f_sum_sq(M, p) = sum(p .^ 2)
        grad_f_sum_sq(M, p) = 2 .* p
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f_sum_sq, grad_f_sum_sq))
        p = [1.0, 2.0]
        η = -grad_f_sum_sq(M, p)
        gs = GradientDescentState(M; p = p)

        hzls = HagerZhangLinesearch()(M)
        @test startswith(repr(hzls), "HagerZhangLinesearch(;")
        @test startswith(Manopt.status_summary(hzls), "HagerZhangLinesearch(;")
        @test Manopt.get_message(hzls) == ""

        α = hzls(dmp, gs, 1, η)
        @test isfinite(α)
        @test α > 0
        α2 = hzls(dmp, gs, 1, η; gradient = grad_f_sum_sq(M, p))
        @test α2 ≈ α
        @test hzls.last_stepsize == α
        @test hzls.last_cost <= f_sum_sq(M, p) + 1.0e-12

        hzls_limit = Manopt.HagerZhangLinesearchStepsize(M; stepsize_limit = 0.05)
        α_limit = hzls_limit(dmp, gs, 1, η)
        @test α_limit <= 0.05 + eps(0.05)
        @test hzls_limit.last_stepsize == α_limit
        α_limit_kw = hzls_limit(dmp, gs, 2, η; stop_when_stepsize_exceeds = 0.01)
        @test α_limit_kw <= 0.01 + eps(0.01)
        @testset "Running out of evaluations in _hz_evaluate_next_step" begin
            N = length(hzls_limit.triples) - hzls_limit.last_evaluation_index
            for i in 1:N
                Manopt._hz_evaluate_next_step(hzls_limit, M, dmp, p, η, 0.1)
            end
            @test_throws ErrorException Manopt._hz_evaluate_next_step(hzls_limit, M, dmp, p, η, 0.1)
        end
        @testset "Wolfe condition modes" begin
            hzls_default = Manopt.HagerZhangLinesearchStepsize(M)
            hzls.current_mode = :invalid_mode
            @test_throws ErrorException hzls(dmp, gs, 1, η)
        end


        hzls_approx = Manopt.HagerZhangLinesearchStepsize(
            M; wolfe_condition_mode = :approximate, stepsize_limit = 0.2
        )
        α_approx = hzls_approx(dmp, gs, 1, η)
        @test α_approx > 0

        @testset "termination modes" begin
            hzls_std = Manopt.HagerZhangLinesearchStepsize(
                M;
                wolfe_condition_mode = :standard,
                initial_guess = Manopt.ConstantInitialGuess(0.5),
                max_function_evaluations = 5,
            )
            α_std = hzls_std(dmp, gs, 1, η)
            @test isapprox(α_std, 0.5; rtol = 1.0e-12, atol = 0.0)
            @test hzls_std.current_mode == :standard

            hzls_adapt = Manopt.HagerZhangLinesearchStepsize(
                M;
                wolfe_condition_mode = :adaptive,
                initial_guess = Manopt.ConstantInitialGuess(0.5),
                initial_last_cost = f_sum_sq(M, p),
                ω = 1.0,
                max_function_evaluations = 5,
            )
            α_adapt = hzls_adapt(dmp, gs, 1, η)
            @test α_adapt > 0
            @test hzls_adapt.current_mode == :approximate

            hzls_eval = Manopt.HagerZhangLinesearchStepsize(
                M;
                wolfe_condition_mode = :standard,
                initial_guess = Manopt.ConstantInitialGuess(1.0),
                max_function_evaluations = 2,
            )
            α_eval = hzls_eval(dmp, gs, 1, η)
            @test α_eval > 0
            @test hzls_eval.last_evaluation_index == length(hzls_eval.triples)
        end
        @testset "B1 bracketing test" begin
            M = Euclidean(1)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = 2 .* p
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [1.0]
            η = -grad_f(M, p)
            gs = GradientDescentState(M; p = p)
            hzls_b1 = Manopt.HagerZhangLinesearchStepsize(
                M;
                initial_guess = Manopt.ConstantInitialGuess(0.75),
                start_enforcing_wolfe_conditions_at_bracketing_iteration = 2,
                max_bracket_iterations = 1,
            )
            α_b1 = hzls_b1(dmp, gs, 1, η)
            @test α_b1 > 0
        end
        @testset "B2 bracketing test" begin
            M = Euclidean(1)
            # f(x) = -22 x^3 + 33 x^2 - x
            # grad_f(x) = -66 x^2 + 66 x - 1
            f(M, p) = -22 * p[1]^3 + 33 * p[1]^2 - p[1]
            grad_f(M, p) = [-66 * p[1]^2 + 66 * p[1] - 1]
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [0.0]
            η = [1.0] # Descent direction
            gs = GradientDescentState(M; p = p)
            hzls_b2 = Manopt.HagerZhangLinesearchStepsize(
                M;
                initial_guess = Manopt.ConstantInitialGuess(1.0),
                start_enforcing_wolfe_conditions_at_bracketing_iteration = 2,
                max_bracket_iterations = 2,
            )
            α = hzls_b2(dmp, gs, 1, η)
            @test α > 0
        end
        @testset "B3 bracketing test" begin
            M = Euclidean(1)
            # f(x) = -x
            # grad_f(x) = -1
            f(M, p) = -p[1]
            grad_f(M, p) = [-1.0]
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [0.0]
            η = [1.0] # Descent direction
            gs = GradientDescentState(M; p = p)
            hzls_b3 = Manopt.HagerZhangLinesearchStepsize(
                M;
                initial_guess = Manopt.ConstantInitialGuess(1.0),
                stepsize_limit = 2.0,
                max_bracket_iterations = 2,
            )
            α = hzls_b3(dmp, gs, 1, η)
            @test α > 0
        end
        @testset "U1 trigger test" begin
            M = Euclidean(1)
            # f(x) = x^2 / 2
            # grad_f(x) = x
            f(M, p) = p[1]^2 / 2
            grad_f(M, p) = [p[1]]
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [1.0]
            η = [-1.0] # Descent direction
            gs = GradientDescentState(M; p = p)
            hzls_u1 = Manopt.HagerZhangLinesearchStepsize(
                M;
                initial_guess = Manopt.ConstantInitialGuess(2.0),
            )
            # We expect U1 to be triggered during the update (secant is exact, slope 0 >= 0)
            α = hzls_u1(dmp, gs, 1, η)
            @test α > 0
        end
        @testset "U2 trigger test" begin
            M = Euclidean(1)
            # We mock f and grad_f to trigger U2 termination
            # We need:
            # 1. Starting at p=0 with descent direction (df < 0)
            # 2. Bracketing finds a point with df > 0 (to finish bracketing) -> p=1.0, df=1.0
            # 3. Refinement hits max evaluations at a point with df < 0 and f > f(0)+eps -> p=0.5, f=10.0, df=-0.1

            function f_u2(M, q)
                v = q[1]
                if isapprox(v, 0.0; atol = 1.0e-9)
                    return 0.0
                elseif isapprox(v, 1.0; atol = 1.0e-9)
                    return 0.0
                elseif isapprox(v, 0.5; atol = 1.0e-9)
                    return 10.0
                end
                return 0.0
            end

            function grad_f_u2(M, q)
                v = q[1]
                if isapprox(v, 0.0; atol = 1.0e-9)
                    return [-1.0]
                elseif isapprox(v, 1.0; atol = 1.0e-9)
                    return [1.0]
                elseif isapprox(v, 0.5; atol = 1.0e-9)
                    return [-0.1]
                end
                return [0.0]
            end

            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f_u2, grad_f_u2))
            p = [0.0]
            η = [1.0]
            gs = GradientDescentState(M; p = p)
            hzls_u2 = Manopt.HagerZhangLinesearchStepsize(
                M; initial_guess = Manopt.ConstantInitialGuess(1.0), max_function_evaluations = 3
            )
            α = hzls_u2(dmp, gs, 1, η)
            @test α > 0
        end
        @testset "U3 trigger test" begin
            M = Euclidean(1)
            # Trigger U3 by having a point that satisfies conditions for U2 but f_eval is false.
            # Same landscape as U2:
            # p=0, df=-1 (start)
            # p=1, df=1 (end of bracket)
            # p=0.5, f=10, df=-0.1 (high function value, negative slope)

            function f_u3(M, q)
                v = q[1]
                if isapprox(v, 0.0; atol = 1.0e-9)
                    return 0.0
                elseif isapprox(v, 1.0; atol = 1.0e-9)
                    return 0.0
                elseif isapprox(v, 0.5; atol = 1.0e-9)
                    return 10.0
                end
                return 0.0
            end

            function grad_f_u3(M, q)
                v = q[1]
                if isapprox(v, 0.0; atol = 1.0e-9)
                    return [-1.0]
                elseif isapprox(v, 1.0; atol = 1.0e-9)
                    return [1.0]
                elseif isapprox(v, 0.5; atol = 1.0e-9)
                    return [-0.1]
                end
                return [0.0]
            end

            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f_u3, grad_f_u3))
            p = [0.0]
            η = [1.0] # Descent direction
            gs = GradientDescentState(M; p = p)
            # Set max_function_evaluations > 3 so we don't hit U2 termination (f_eval=true)
            hzls_u3 = Manopt.HagerZhangLinesearchStepsize(
                M; initial_guess = Manopt.ConstantInitialGuess(1.0), max_function_evaluations = 5
            )
            α = hzls_u3(dmp, gs, 1, η)
            @test α > 0
        end
        @testset "U3 (b) trigger test" begin
            M = Euclidean(1)
            # Force U3 (b) in _hz_u3:
            # 1) At d=0.5 we need df < 0 and f(d) <= f(0) + ϵₖ with no termination,
            #    so i_a_bar gets updated to i_d.
            # 2) On the next U3 iteration we return from the loop.
            function f_u3b(M, q)
                return 0.0
            end

            function grad_f_u3b(M, q)
                v = q[1]
                if isapprox(v, 0.0; atol = 1.0e-12)
                    return [-1.0]
                elseif isapprox(v, 1.0; atol = 1.0e-12)
                    return [1.0]
                elseif isapprox(v, 0.5; atol = 1.0e-12)
                    return [-1.0]
                elseif isapprox(v, 0.75; atol = 1.0e-12)
                    return [1.0]
                end
                return [0.0]
            end

            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f_u3b, grad_f_u3b))
            p = [0.0]
            η = [1.0]
            hzls_u3b = Manopt.HagerZhangLinesearchStepsize(M; max_function_evaluations = 4)
            Manopt.initialize_stepsize!(hzls_u3b)
            Manopt._hz_evaluate_next_step(hzls_u3b, M, dmp, p, η, 0.0)
            Manopt._hz_evaluate_next_step(hzls_u3b, M, dmp, p, η, 1.0)

            (i_a, i_b, f_eval, f_wolfe) = Manopt._hz_u3(hzls_u3b, M, dmp, p, η, 1, 2)
            @test (i_a, i_b) == (3, 4)
            @test f_eval
            @test !f_wolfe
        end
        @testset "U3 (c) info trigger test" begin
            M = Euclidean(1)
            # Force U3 (c) inside _hz_u3 by making the mid-point have
            # negative slope but too large function value.
            function f_u3c(M, q)
                v = q[1]
                if isapprox(v, 0.0; atol = 1.0e-12)
                    return 0.0
                elseif isapprox(v, 1.0; atol = 1.0e-12)
                    return 0.0
                elseif isapprox(v, 0.5; atol = 1.0e-12)
                    return 1.0
                elseif isapprox(v, 0.25; atol = 1.0e-12)
                    return 0.0
                end
                return 0.0
            end

            function grad_f_u3c(M, q)
                v = q[1]
                if isapprox(v, 0.0; atol = 1.0e-12)
                    return [-1.0]
                elseif isapprox(v, 1.0; atol = 1.0e-12)
                    return [1.0]
                elseif isapprox(v, 0.5; atol = 1.0e-12)
                    return [-0.1]
                elseif isapprox(v, 0.25; atol = 1.0e-12)
                    return [0.1]
                end
                return [0.0]
            end

            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f_u3c, grad_f_u3c))
            p = [0.0]
            η = [1.0]
            hzls_u3c = Manopt.HagerZhangLinesearchStepsize(M; max_function_evaluations = 4)
            Manopt.initialize_stepsize!(hzls_u3c)
            Manopt._hz_evaluate_next_step(hzls_u3c, M, dmp, p, η, 0.0)
            Manopt._hz_evaluate_next_step(hzls_u3c, M, dmp, p, η, 1.0)
            @test (1, 4, true, false) == Manopt._hz_u3(hzls_u3c, M, dmp, p, η, 1, 2)
        end
        @testset "U3 max evaluations termination" begin
            M = Euclidean(1)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = 2 .* p
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [0.0]
            η = [1.0]

            hzls_u3_max = Manopt.HagerZhangLinesearchStepsize(M; max_function_evaluations = 2)
            Manopt.initialize_stepsize!(hzls_u3_max)
            Manopt._hz_evaluate_next_step(hzls_u3_max, M, dmp, p, η, 0.0)
            Manopt._hz_evaluate_next_step(hzls_u3_max, M, dmp, p, η, 1.0)
            @test hzls_u3_max.last_evaluation_index == length(hzls_u3_max.triples)

            (i_a, i_b, f_eval, f_wolfe) = Manopt._hz_u3(hzls_u3_max, M, dmp, p, η, 1, 2)
            @test (i_a, i_b) == (1, 2)
            @test !f_eval
            @test !f_wolfe
        end
        @testset "U0 out-of-bracket early return" begin
            M = Euclidean(1)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = 2 .* p
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [0.0]
            η = [1.0]

            hzls_u0 = Manopt.HagerZhangLinesearchStepsize(M; max_function_evaluations = 5)
            Manopt.initialize_stepsize!(hzls_u0)
            Manopt._hz_evaluate_next_step(hzls_u0, M, dmp, p, η, 0.0)
            Manopt._hz_evaluate_next_step(hzls_u0, M, dmp, p, η, 1.0)

            last_eval_before = hzls_u0.last_evaluation_index

            # c is left of bracket [0, 1] -> U0 early return
            @test (1, 2, -1, false, false) == Manopt._hz_update(hzls_u0, M, dmp, p, η, 1, 2, -0.1)
            @test hzls_u0.last_evaluation_index == last_eval_before

            # c is right of bracket [0, 1] -> U0 early return
            @test (1, 2, -1, false, false) == Manopt._hz_update(hzls_u0, M, dmp, p, η, 1, 2, 1.1)
            @test hzls_u0.last_evaluation_index == last_eval_before
        end

        @testset "S2 trigger test" begin
            M = Euclidean(1)
            # S2 is triggered within _hz_secant2 when the updated bracket point i_c is the new upper bound i_B
            # This happens if slope at c is positive (U1 case in _hz_update).
            # Sequence:
            # 1. Start p=0, df=-1.
            # 2. Initial bracket p=1, df=4 (df > 0 -> bracket found).
            # 3. _hz_secant2 calls secant(0, 1) -> c = (0*4 - 1*(-1))/(4 - (-1)) = 0.2.
            # 4. At c=0.2, we set df=0.1 (positive slope -> U1 -> i_c = i_B).
            # 5. We also need f(0.2) high enough to fail Armijo so we don't return early with f_wolfe=true.
            #    f(0)=0. f(0.2)=0.5. Armijo check: 0.5 <= 0 + 0.1*0.2*(-1) = -0.02 (False).

            function f_s2(M, q)
                v = q[1]
                if isapprox(v, 0.0; atol = 1.0e-9)
                    return 0.0
                elseif isapprox(v, 1.0; atol = 1.0e-9)
                    return 2.0 # Arbitrary high value
                elseif isapprox(v, 0.2; atol = 1.0e-9)
                    return 0.5 # Fail Armijo
                end
                return 0.0 # Fallback (e.g. for c_bar in S2)
            end

            function grad_f_s2(M, q)
                v = q[1]
                if isapprox(v, 0.0; atol = 1.0e-9)
                    return [-1.0]
                elseif isapprox(v, 1.0; atol = 1.0e-9)
                    return [4.0]
                elseif isapprox(v, 0.2; atol = 1.0e-9)
                    return [0.1] # Positive slope triggers U1 -> i_c = i_B
                end
                return [0.0]
            end

            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f_s2, grad_f_s2))
            p = [0.0]
            η = [1.0]
            gs = GradientDescentState(M; p = p)
            hzls_s2 = Manopt.HagerZhangLinesearchStepsize(
                M; initial_guess = Manopt.ConstantInitialGuess(1.0)
            )
            # We expect the S2 log
            α = hzls_s2(dmp, gs, 1, η)
            @test α > 0
        end

        @testset "S3 trigger test" begin
            M = Euclidean(1)
            # S3 is triggered within _hz_secant2 when the updated bracket point i_c is the new lower bound i_A
            # (U2 case in _hz_update). We set up:
            # 1. Start p=0, df=-1 (descent).
            # 2. Bracket at p=1, df=4 (positive slope).
            # 3. Secant gives c=0.2. At c, df=-0.1 and f=0 -> U2.

            function f_s3(M, q)
                return 0.0
            end

            function grad_f_s3(M, q)
                v = q[1]
                if isapprox(v, 0.0; atol = 1.0e-12)
                    return [-1.0]
                elseif isapprox(v, 1.0; atol = 1.0e-12)
                    return [4.0]
                elseif isapprox(v, 0.2; atol = 1.0e-12)
                    return [-0.1]
                end
                return [0.0]
            end

            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f_s3, grad_f_s3))
            p = [0.0]
            η = [1.0]
            hzls_s3 = Manopt.HagerZhangLinesearchStepsize(M; max_function_evaluations = 5)
            Manopt.initialize_stepsize!(hzls_s3)
            Manopt._hz_evaluate_next_step(hzls_s3, M, dmp, p, η, 0.0)
            Manopt._hz_evaluate_next_step(hzls_s3, M, dmp, p, η, 1.0)

            c = Manopt.secant(hzls_s3.triples[1], hzls_s3.triples[2])
            (i_A, i_B, i_c, f_eval, f_wolfe) = Manopt._hz_secant2(hzls_s3, M, dmp, p, η, 1, 2)
            @test !f_eval
            @test !f_wolfe
            @test hzls_s3.triples[i_A].t ≈ c atol = 1.0e-12

            c_bar = Manopt.secant(hzls_s3.triples[1], hzls_s3.triples[i_A])
            @test hzls_s3.triples[i_c].t ≈ c_bar atol = 1.0e-12
            @test i_A != i_B
        end

        @testset "Hager-Zhang infinite at b" begin
            # A function that is finite for small steps but infinite for larger ones
            # and has positive slope where it is infinite to trigger the bracket condition.

            M = Euclidean(1)

            # f(x) = x^2 - x for x < 1.0
            # f(x) = Inf for x >= 1.0
            # Min at x = 0.5, f(0.5) = -0.25
            function f_inf(M, p)
                x = p[1]
                if x < 1.0
                    return x^2 - x
                else
                    return Inf
                end
            end

            function grad_f_inf(M, p)
                x = p[1]
                if x < 1.0
                    return [2 * x - 1]
                else
                    # Return a positive slope to satisfy _hz_bracket exit condition
                    return [1.0]
                end
            end

            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f_inf, grad_f_inf))

            # Start at 0. f(0)=0. grad(0)=-1. Search direction +1.
            s = GradientDescentState(M; p = [0.0])

            # Force initial guess to be 2.0 (in the infinite region)
            hzls = HagerZhangLinesearch(; initial_guess = Manopt.ConstantInitialGuess(2.0))(M)

            # Because initial bracket will be [0, 2] with f(2)=Inf.
            # Then bisection will eventually find 0.5.

            step = hzls(dmp, s, 1, [1.0])
            @test abs(step - 0.5) < 1.0e-1
        end

        @testset "Hager-Zhang initialize_stepsize!" begin
            hzls = HagerZhangLinesearch()(M)
            hzls.last_evaluation_index = 5
            hzls.Qₖ = 2.0
            hzls.Cₖ = 2.0
            hzls.current_mode = :approximate
            Manopt.initialize_stepsize!(hzls)
            @test hzls.last_evaluation_index == 0
            @test hzls.Qₖ == 0.0
            @test hzls.Cₖ == 0.0
            @test hzls.current_mode == :standard
        end

    end
    @testset "Distance over Gradients Stepsize" begin
        @testset "does not use sectional cuvature (Eucludian)" begin
            M = Euclidean(2)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = sum(2 .* p)
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [2.0, 2.0]
            gds = GradientDescentState(M; p = p)
            ds = DistanceOverGradientsStepsize(
                M, p; initial_distance = 1.0, use_curvature = false
            )
            @test ds.gradient_sum == 0
            @test ds.max_distance == 1.0
            @test ds.initial_point == p
            @test ds.last_stepsize === get_initial_stepsize(ds)
            @test ds.last_stepsize === NaN
            @test ds.last_stepsize === get_last_stepsize(ds)
            # test printed representation before first step
            repr_ds = repr(ds)
            @test occursin("DistanceOverGradients(;", repr_ds)
            @test occursin("initial_distance = 1.0", repr_ds)
            @test occursin("use_curvature = false", repr_ds)
            @test occursin("sectional_curvature_bound = 0.0", repr_ds)
            @test occursin("Current state:", repr_ds)
            @test occursin("max_distance = 1.0", repr_ds)
            @test occursin("gradient_sum = 0.0", repr_ds)
            @test occursin("last_stepsize = NaN", repr_ds)
            lr = ds(dmp, gds, 0)
            @test lr == 0.125
            # after first step, last_stepsize should be reflected in repr
            repr_ds_after = repr(ds)
            @test occursin("last_stepsize = 0.125", repr_ds_after)
        end
        @testset "use sectional cuvature (Euclidian)" begin
            M = Euclidean(2)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = sum(2 .* p)
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [2.0, 2.0]
            gds = GradientDescentState(M; p = p)
            ds = DistanceOverGradientsStepsize(
                M, p;
                initial_distance = 1.0,
                use_curvature = true,
                sectional_curvature_bound = 0.0,
            )
            @test ds.gradient_sum == 0
            @test ds.max_distance == 1.0
            @test ds.initial_point == p
            @test ds.last_stepsize === get_initial_stepsize(ds)
            @test ds.last_stepsize === NaN
            @test ds.last_stepsize === get_last_stepsize(ds)
            lr = ds(dmp, gds, 0)
            @test lr == 0.125
        end
        @testset "do not use sectional cuvature (Sphere)" begin
            M = Sphere(1)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = sum(2 .* p)
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [1, 0]
            gds = GradientDescentState(M; p = p)
            ds = DistanceOverGradientsStepsize(
                M, p; initial_distance = 1.0, use_curvature = false
            )
            @test ds.gradient_sum == 0
            @test ds.max_distance == 1.0
            @test ds.initial_point == p
            @test ds.last_stepsize === get_initial_stepsize(ds)
            @test ds.last_stepsize === NaN
            @test ds.last_stepsize === get_last_stepsize(ds)
            lr = ds(dmp, gds, 0)
            @test lr == 0.5
        end
        @testset "use sectional cuvature (Sphere)" begin
            M = Sphere(1)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = sum(2 .* p)
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [1, 0]
            gds = GradientDescentState(M; p = p)
            ds = DistanceOverGradientsStepsize(
                M, p;
                initial_distance = 1.0,
                use_curvature = true,
                sectional_curvature_bound = 1.0,
            )
            @test ds.gradient_sum == 0
            @test ds.max_distance == 1.0
            @test ds.initial_point == p
            @test ds.last_stepsize === get_initial_stepsize(ds)
            @test ds.last_stepsize === NaN
            @test ds.last_stepsize === get_last_stepsize(ds)
            lr = ds(dmp, gds, 0)
            @test lr == 0.5
        end
        @testset "use sectional curvature (Hyperbolic)" begin
            M = Hyperbolic(2)              # Lorentz model in R^3

            t = 0.5
            p = [cosh(t), sinh(t), 0.0]

            v_ambient = [0.0, 1.0, 0.0]
            g = project(M, p, v_ambient)
            gnorm = norm(M, p, g)
            f(M, q) = 0.0
            grad_f(M, q) = g

            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            gds = GradientDescentState(M; p = p)
            ds = DistanceOverGradientsStepsize(
                M, p;
                initial_distance = 1.0,
                use_curvature = true,
                sectional_curvature_bound = -1.0,
            )

            @test ds.gradient_sum == 0
            @test ds.max_distance == 1.0
            @test ds.initial_point == p
            @test ds.last_stepsize === get_initial_stepsize(ds)
            @test ds.last_stepsize === NaN
            @test ds.last_stepsize === get_last_stepsize(ds)

            # Expected initial step:
            # ζκ(1) = (√|κ| * 1) / tanh(√|κ| * 1) = 1 / tanh(1)
            # lr = initial_distance / (sqrt(ζκ(1)) * sqrt(gnorm^2)) = 1 / (sqrt(ζκ(1)) * gnorm)
            zeta = 1 / tanh(1.0)
            expected_lr = 1.0 / (sqrt(zeta) * gnorm)

            # Call with the known gradient to avoid any objective/gradient mismatches
            lr = ds(dmp, gds, 0; gradient = g)
            @test isapprox(lr, expected_lr; rtol = 1.0e-12, atol = 0)
        end
        @testset "Simple Rayleigh coefficient" begin
            # Minimize negative Rayleigh quotient on the sphere S^1
            M = Sphere(1)
            A = [1.0 0; 0 1.0]

            f(M, p) = -p' * A * p

            function grad_f(M, p)
                return project(M, p, -2 * A * p)
            end

            p0 = rand(M)

            x = gradient_descent(
                M, f, grad_f, p0;
                stepsize = DistanceOverGradients(initial_distance = Manifolds.injectivity_radius(M)),
                stopping_criterion = StopWhenGradientNormLess(1.0e-15),
            )

            # 1e-6 is the maximum rtol for the test to pass on 1.10; it works without specifying rtol on 1.11
            @test f(M, x) ≈ -1 rtol = 1.0e-6
        end
        @testset "Distance from Hyperbolic to origin" begin
            M = Hyperbolic(2)

            f(M, x) = exp(x[1]^2 + x[2]^2)

            function grad_f(M, p)
                val = exp(p[1]^2 + p[2]^2)
                grad_E = [2 * p[1] * val, 2 * p[2] * val, 0.0]
                return project(M, p, grad_E)
            end

            p0 = rand(M)

            x = gradient_descent(
                M, f, grad_f, p0;
                stepsize = DistanceOverGradients(use_curvature = true, sectional_curvature_bound = -1.0),
                stopping_criterion = StopWhenGradientNormLess(1.0e-15)
            )

            @test f(M, x) ≈ 1
        end
    end
    @testset "max_stepsize fallbacks" begin
        M = Manopt.Test.DummyManifold()
        @test isinf(Manopt.max_stepsize(M))
        @test isinf(Manopt.max_stepsize(M, :NoPoint))
    end
    @testset "stepsize messages" begin
        msgRR = Manopt.StepsizeMessage(; bound = 0.0, value = 0.1)
        msgRR.at_iteration = 1
        msgIR = Manopt.StepsizeMessage(; bound = 10, value = 0.2)
        msgIR.at_iteration = 1

        s1 = Manopt.get_message(:non_descent_direction, msgRR)
        @test contains(s1, "⟨η, grad_f(p)⟩ = 0.1 ≥ 0.0")
        s1b = Manopt.get_message(:non_descent_direction, 1, 0.1, 0.0)
        @test s1 == s1b
        s2 = Manopt.get_message(:stepsize_exceeds, msgRR)
        @test contains(s2, "bound (0.0) exceeded")
        @test contains(s2, "Reducing to 0.1")
        s3 = Manopt.get_message(:stop_decreasing, msgIR)
        @test contains(s3, "number of decrease steps (10) reached")
        @test contains(s3, "Continuing with a stepsize of 0.2")
        s4 = Manopt.get_message(:stop_increasing, msgIR)
        @test contains(s4, "number of increase steps (10) reached")
        @test contains(s4, "Continuing with a stepsize of 0.2")
        s5 = Manopt.get_message(:stepsize_less, msgRR)
        @test contains(s5, "bound (0.0) reached")
        @test contains(s5, "Falling back to a stepsize of 0.1")
    end
    @testset "Warnings within WolfePowellLinesearch" begin
        M = Euclidean(2)
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 .* p
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        p = [2.0, 2.0]
        gs = GradientDescentState(M; p = p)
        # large sufficient curvatuture to trigger stop inc.
        wpls = WolfePowellLinesearch(M; stop_increasing_at_step = 1, stop_decreasing_at_step = 1)()
        wpls(dmp, gs, 1)
        # This set the dec message
        @test wpls.messages[:stop_decreasing].at_iteration > 0
        # to hit the innc message we set the values to something surreal
        wpls.sufficient_decrease = 0.1
        wpls.sufficient_curvature = 0.2
        wpls(dmp, gs, 2, -0.0001 * grad_f(M, p))
        @test wpls.messages[:stop_increasing].at_iteration > 0
    end
end
