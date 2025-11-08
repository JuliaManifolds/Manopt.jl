using ManifoldsBase, Manopt, Manifolds, Test

s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))
using ManoptTestSuite

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
        s1 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1.0, 1.0, 0.99; stop_decreasing_at_step = 10
        )
        @test :stop_decreasing in keys(s1[2])
        s2 = Manopt.linesearch_backtrack(
            M,
            f,
            p,
            grad_f(M, p),
            1.0,
            1.0,
            0.5,
            grad_f(M, p);
            retraction_method = ExponentialRetraction(),
        )
        @test :non_descent_direction in keys(s2[2])
        s3 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1.0, 1.0, 0.5; stop_when_stepsize_less = 0.75
        )
        @test :stepsize_exceeds in keys(s3[2])
        # cheating for increase
        s4 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1.0e-12, 0, 0.5; stop_when_stepsize_exceeds = 0.1
        )
        @test :stepsize_exceeds in keys(s4[2])
        @test startswith(s4[2], "Max step size (0.1)")
        s5 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1.0e-12, 0, 0.5; stop_increasing_at_step = 1
        )
        @test :stop_increasing in keys(s5[2])
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
    @testset "Distance over Gradients Stepsize" begin
        @testset "does not use sectional cuvature (Eucludian)" begin
            M = Euclidean(2)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = sum(2 .* p)
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [2.0, 2.0]
            gds = GradientDescentState(M; p = p)
            ds = DistanceOverGradientsStepsize(
                M; p = p, initial_distance = 1.0, use_curvature = false
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
                M;
                p = p,
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
                M; p = p, initial_distance = 1.0, use_curvature = false
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
                M;
                p = p,
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
                M;
                p = p,
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
        M = ManoptTestSuite.DummyManifold()
        @test isinf(Manopt.max_stepsize(M))
        @test isinf(Manopt.max_stepsize(M, :NoPoint))
    end
end
