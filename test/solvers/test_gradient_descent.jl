using Manopt, Manifolds, Test

@testset "Gradient Descent" begin
    @testset "allocating Circle" begin
        # Test the gradient descent with
        # the distance function squared
        # on S1, such that we can easily also verify exp and log
        M = Circle()
        r = [-π / 2, π / 4, 0.0, π / 4]
        s = r
        F(M, x) = 1 / 10 * sum(distance.(Ref(M), s, Ref(x)) .^ 2)
        gradF(M, x) = 1 / 5 * sum(-log.(Ref(M), Ref(x), s))
        o = gradient_descent!(
            M,
            F,
            gradF,
            s[1];
            stopping_criterion=StopWhenAny(
                StopAfterIteration(200), StopWhenChangeLess(10^-16)
            ),
            stepsize=ArmijoLinesearch(1.0, ExponentialRetraction(), 0.99, 0.1),
            debug=[:Iteration, " ", :Cost, :Stop, 100, "\n"],
            record=[:Iteration, :Cost, 1],
            return_options=true,
        )
        x = get_solver_result(o)
        x2 = gradient_descent!(
            M,
            F,
            gradF,
            s[1];
            stopping_criterion=StopWhenAny(
                StopAfterIteration(200), StopWhenChangeLess(10^-16)
            ),
            stepsize=ArmijoLinesearch(1.0, ExponentialRetraction(), 0.99, 0.1),
        )
        @test x == x2
        x3 = gradient_descent!(
            M,
            F,
            gradF,
            s[1];
            stopping_criterion=StopWhenAny(
                StopAfterIteration(1000), StopWhenChangeLess(10^-16)
            ),
            stepsize=NonmonotoneLinesearch(
                1.0,
                ExponentialRetraction(),
                ParallelTransport(),
                0.99,
                0.1,
                2,
                1e-7,
                π / 2,
                :direct,
            ),
            debug=[:Stop],
        )
        @test isapprox(x, x3; atol=1e-13)
        x4 = gradient_descent!(
            M,
            F,
            gradF,
            s[1];
            stopping_criterion=StopWhenAny(
                StopAfterIteration(1000), StopWhenChangeLess(10^-16)
            ),
            stepsize=NonmonotoneLinesearch(
                1.0,
                ExponentialRetraction(),
                ParallelTransport(),
                0.99,
                0.1,
                2,
                1e-7,
                π / 2,
                :inverse,
            ),
            debug=[:Stop],
        )
        @test isapprox(x, x4; atol=1e-13)
        x5 = gradient_descent!(
            M,
            F,
            gradF,
            s[1];
            stopping_criterion=StopWhenAny(
                StopAfterIteration(1000), StopWhenChangeLess(10^-16)
            ),
            stepsize=NonmonotoneLinesearch(
                1.0,
                ExponentialRetraction(),
                ParallelTransport(),
                0.99,
                0.1,
                2,
                1e-7,
                π / 2,
                :alternating,
            ),
            debug=[:Stop],
        )
        @test isapprox(x, x5; atol=1e-13)
        x6 = gradient_descent!(
            M,
            F,
            gradF,
            s[1];
            stopping_criterion=StopWhenAny(
                StopAfterIteration(1000), StopWhenChangeLess(10^-16)
            ),
            direction=Nesterov(s[1]),
            debug=[:Stop],
        )
        @test isapprox(x, x6; atol=1e-13)

        @test_logs (
            :warn,
            string(
                "The strategy 'indirect' is not defined. The 'direct' strategy is used instead.",
            ),
        ) NonmonotoneLinesearch(
            1.0,
            ExponentialRetraction(),
            ParallelTransport(),
            0.99,
            0.1,
            2,
            1e-7,
            π / 2,
            :indirect,
        )
        @test_throws DomainError NonmonotoneLinesearch(
            1.0,
            ExponentialRetraction(),
            ParallelTransport(),
            0.99,
            0.1,
            2,
            0.0,
            π / 2,
            :direct,
        )
        @test_throws DomainError NonmonotoneLinesearch(
            1.0,
            ExponentialRetraction(),
            ParallelTransport(),
            0.99,
            0.1,
            2,
            π / 2,
            π / 4,
            :direct,
        )
        @test_throws DomainError NonmonotoneLinesearch(
            1.0,
            ExponentialRetraction(),
            ParallelTransport(),
            0.99,
            0.1,
            0,
            π / 4,
            π / 2,
            :direct,
        )

        rec = get_record(o)
        # after one step for local enough data -> equal to real valued data
        @test abs(x - sum(r) / length(r)) ≈ 0 atol = 5 * 10.0^(-14)
        # Test Fallbacks -> we can't do steps with the wrong combination
        p = SubGradientProblem(M, F, gradF)
        o = GradientDescentOptions(
            s[1]; stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(1.0)
        )
        @test_throws MethodError initialize_solver!(p, o)
        @test_throws MethodError step_solver!(p, o, 1)
    end
    @testset "mutating Sphere" begin
        M = Sphere(2)
        north = [0.0, 0.0, 1.0]
        pre_pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        pts = exp.(Ref(M), Ref(north), pre_pts)
        F(M, x) = 1 / 8 * sum(distance.(Ref(M), pts, Ref(x)) .^ 2)
        gradF(M, x) = 1 / 4 * sum(-log.(Ref(M), Ref(x), pts))
        n2 = gradient_descent(M, F, gradF, pts[1])
        @test !isapprox(M, pts[1], n2) # n2 is newly allocated and not pts[1]
        @test isapprox(M, north, n2)
        n3 = gradient_descent(
            M, F, gradF, pts[1]; direction=MomentumGradient(M, pts[1]), debug=[]
        )
        @test isapprox(M, north, n3)
        n4 = gradient_descent(M, F, gradF, pts[1]; direction=AverageGradient(M, pts[1], 5))
        @test isapprox(M, north, n4; atol=1e-7)
    end
    @testset "Warning when cost increases" begin
        M = Sphere(2)
        q = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
        F(M, p) = distance(M, p, q) .^ 2
        # chosse a wrong gradient such that ConstantStepsize yields an increase
        gradF(M, p) = -grad_distance(M, q, p)
        # issues three warnings
        @test_logs (:warn,) (:warn,) gradient_descent(
            M, F, gradF, 1 / sqrt(2) .* [1.0, -1.0, 0.0]
        )
        @test_logs (:warn,) (:warn,) (warn,) gradient_descent(
            M,
            F,
            gradF,
            1 / sqrt(2) .* [1.0, -1.0, 0.0];
            debug=[DebugWarnIfCostIncreases(:Once)],
        )
    end
end
