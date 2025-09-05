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
            M, f, p, grad_f(M, p), 1.0, 1.0, 0.99; stop_decreasing_at_step=10
        )
        @test startswith(s1[2], "Max decrease")
        s2 = Manopt.linesearch_backtrack(
            M,
            f,
            p,
            grad_f(M, p),
            1.0,
            1.0,
            0.5,
            grad_f(M, p);
            retraction_method=ExponentialRetraction(),
        )
        @test startswith(s2[2], "The search direction")
        s3 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1.0, 1.0, 0.5; stop_when_stepsize_less=0.75
        )
        @test startswith(s3[2], "Min step size (0.75)")
        # cheating for increase
        s4 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1e-12, 0, 0.5; stop_when_stepsize_exceeds=0.1
        )
        @test startswith(s4[2], "Max step size (0.1)")
        s5 = Manopt.linesearch_backtrack(
            M, f, p, grad_f(M, p), 1e-12, 0, 0.5; stop_increasing_at_step=1
        )
        @test startswith(s5[2], "Max increase steps (1)")
    end
    @testset "Adaptive WN Gradient" begin
        # Build a dummy function and gradient
        f(M, p) = 0
        grad_f(M, p) = [0.0, 0.75, 0.0] # valid, since only north pole used
        M = Sphere(2)
        p = [1.0, 0.0, 0.0]
        mgo = ManifoldGradientObjective(f, grad_f)
        mp = DefaultManoptProblem(M, mgo)
        s = AdaptiveWNGradient(; gradient_reduction=0.5, count_threshold=2)(M)
        gds = GradientDescentState(M; p=p)
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
        gds = GradientDescentState(M; p=p)
        abs_dec_step = Manopt.DecreasingStepsize(
            M; length=10.0, factor=1.0, subtrahend=0.0, exponent=1.0, type=:absolute
        )
        solve!(mp, gds)
        @test abs_dec_step(mp, gds, 1) ==
            10.0 / norm(get_manifold(mp), get_iterate(gds), get_gradient(gds))
        abs_const_step = Manopt.ConstantStepsize(M, 1.0; type=:absolute)
        @test abs_const_step(mp, gds, 1) ==
            1.0 / norm(get_manifold(mp), get_iterate(gds), get_gradient(gds))
    end
    @testset "Polyak Stepsize" begin
        M = Euclidean(2)
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = sum(2 .* p)
        dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        p = [2.0, 2.0]
        X = grad_f(M, p)
        sgs = SubGradientMethodState(M; p=p)
        ps = Polyak()()
        @test repr(ps) ==
            "Polyak()\nA stepsize with keyword parameters\n   * initial_cost_estimate = 0.0\n"
        @test ps(dmp, sgs, 1) == (f(M, p) - 0 + 1) / (norm(M, p, X)^2)
    end
    @testset "Distance over Gradients Stepsize" begin
        @testset "does not use sectional cuvature (Eucludian)" begin
            M = Euclidean(2)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = sum(2 .* p)
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [2.0, 2.0]
            gds = GradientDescentState(M; p=p)
            ds = DistanceOverGradientsStepsize(M, p=p, initial_distance=1.0, use_curvature=false)
            @test ds.gradient_sum == 0
            @test ds.max_distance == 1.0
            @test ds.initial_point == p
            @test ds.last_stepsize  === NaN
            lr = ds(dmp, gds, 0)
            @test lr == 0.125
        end
        @testset "use sectional cuvature (Euclidian)" begin
            M = Euclidean(2)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = sum(2 .* p)
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [2.0, 2.0]
            gds = GradientDescentState(M; p=p)
            ds = DistanceOverGradientsStepsize(M, p=p, initial_distance=1.0, use_curvature=true, sectional_curvature_bound=0.0)
            @test ds.gradient_sum == 0
            @test ds.max_distance == 1.0
            @test ds.initial_point == p
            @test ds.last_stepsize  === NaN
            lr = ds(dmp, gds, 0)
            @test lr == 0.125
        end
        @testset "do not use sectional cuvature (Sphere)" begin
            M = Sphere(1)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = sum(2 .* p)
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [1, 0] 
            gds = GradientDescentState(M; p=p)
            ds = DistanceOverGradientsStepsize(M, p=p, initial_distance=1.0, use_curvature=false)
            @test ds.gradient_sum == 0
            @test ds.max_distance == 1.0
            @test ds.initial_point == p
            @test ds.last_stepsize  === NaN
            lr = ds(dmp, gds, 0)
            @test lr == 0.5
        end
        @testset "use sectional cuvature (Sphere)" begin
            M = Sphere(1)
            f(M, p) = sum(p .^ 2)
            grad_f(M, p) = sum(2 .* p)
            dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            p = [1, 0]
            gds = GradientDescentState(M; p=p)
            ds = DistanceOverGradientsStepsize(M, p=p, initial_distance=1.0, use_curvature=true, sectional_curvature_bound=1.0)
            @test ds.gradient_sum == 0
            @test ds.max_distance == 1.0
            @test ds.initial_point == p
            @test ds.last_stepsize  === NaN
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
            gds = GradientDescentState(M; p=p)
            ds = DistanceOverGradientsStepsize(M; p=p, initial_distance=1.0, use_curvature=true, sectional_curvature_bound=-1.0)
        
            @test ds.gradient_sum == 0
            @test ds.max_distance == 1.0
            @test ds.initial_point == p
            @test ds.last_stepsize === NaN
        
            # Expected initial step:
            # ζκ(1) = (√|κ| * 1) / tanh(√|κ| * 1) = 1 / tanh(1)
            # lr = initial_distance / (sqrt(ζκ(1)) * sqrt(gnorm^2)) = 1 / (sqrt(ζκ(1)) * gnorm)
            zeta = 1 / tanh(1.0)
            expected_lr = 1.0 / (sqrt(zeta) * gnorm)
        
            # Call with the known gradient to avoid any objective/gradient mismatches
            lr = ds(dmp, gds, 0; gradient=g)
            @test isapprox(lr, expected_lr; rtol=1e-12, atol=0)
        end
    end
    @testset "max_stepsize fallbacks" begin
        M = ManoptTestSuite.DummyManifold()
        @test isinf(Manopt.max_stepsize(M))
        @test isinf(Manopt.max_stepsize(M, :NoPoint))
    end
end
