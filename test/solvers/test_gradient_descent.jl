using Manopt, Manifolds, Test

@testset "Gradient Descent" begin
    @testset "allocating Circle" begin
        # Test the gradient descent with
        # the distance function squared
        # on S1, such that we can easily also verify exp and log
        M = Circle()
        #vecr(α) = [cos(α), sin(α)]
        r = [-π / 2, π / 4, 0.0, π / 4]
        r2 = [-π / 2, π / 4, 0.0, π / 4]
        apprpstar = sum([-π / 2, π / 4, 0.0, π / 4]) / length(r)
        F(M, p) = 1 / 10 * sum(distance.(Ref(M), r2, Ref(p)) .^ 2)
        gradF(M, p) = 1 / 5 * sum(-log.(Ref(M), Ref(p), r2))
        my_io = IOBuffer()
        d = DebugEvery(
            DebugGroup([
                DebugIterate(; io=my_io),
                DebugDivider(" "; io=my_io),
                DebugCost(; io=my_io),
                DebugStoppingCriterion(; io=my_io),
                DebugDivider("\n"; io=my_io),
            ]),
            500,
        )
        s = gradient_descent!(
            M,
            F,
            gradF,
            r2[1];
            stopping_criterion=StopAfterIteration(200) | StopWhenChangeLess(10^-16),
            stepsize=ArmijoLinesearch(M; contraction_factor=0.99),
            debug=d,
            record=[:Iteration, :Cost, 1],
            return_state=true,
        )
        x = get_solver_result(s)
        res_debug = String(take!(my_io))
        @test res_debug === " F(x): 1.357071\n"
        x2 = gradient_descent!(
            M,
            F,
            gradF,
            r2[1];
            stopping_criterion=StopAfterIteration(200) | StopWhenChangeLess(10^-16),
            stepsize=ArmijoLinesearch(M; contraction_factor=0.99),
        )
        @test x == x2
        step = NonmonotoneLinesearch(
            M;
            stepsize_reduction=0.99,
            sufficient_decrease=0.1,
            memory_size=2,
            min_stepsize=1e-7,
            max_stepsize=π / 2,
            strategy=:direct,
        )
        x3 = gradient_descent!(
            M,
            F,
            gradF,
            r2[1];
            stopping_criterion=StopAfterIteration(1000) | StopWhenChangeLess(10^-16),
            stepsize=step,
        )
        @test isapprox(M, x, x3; atol=1e-13)
        step.strategy = :inverse
        x4 = gradient_descent!(
            M,
            F,
            gradF,
            r2[1];
            stopping_criterion=StopAfterIteration(1000) | StopWhenChangeLess(10^-16),
            stepsize=step,
        )
        @test isapprox(M, x, x4; atol=1e-13)
        step.strategy = :alternating
        x5 = gradient_descent!(
            M,
            F,
            gradF,
            r2[1];
            stopping_criterion=StopWhenAny(
                StopAfterIteration(1000), StopWhenChangeLess(10^-16)
            ),
            stepsize=step,
        )
        @test isapprox(M, x, x5; atol=1e-13)
        x6 = gradient_descent!(
            M,
            F,
            gradF,
            r2[1];
            stopping_criterion=StopWhenAny(
                StopAfterIteration(1000), StopWhenChangeLess(10^-16)
            ),
            direction=Nesterov(M, r2[1]),
        )
        @test isapprox(M, x, x6; atol=1e-13)

        @test_logs (
            :warn,
            string(
                "The strategy 'indirect' is not defined. The 'direct' strategy is used instead.",
            ),
        ) NonmonotoneLinesearch(M; strategy=:indirect)
        @test_throws DomainError NonmonotoneLinesearch(M; min_stepsize=0.0)
        @test_throws DomainError NonmonotoneLinesearch(
            M; min_stepsize=π / 2, max_stepsize=π / 4
        )
        @test_throws DomainError NonmonotoneLinesearch(M; memory_size=0)

        rec = get_record(s)
        # after one step for local enough data -> equal to real valued data
        @test distance(M, x, apprpstar) ≈ 0 atol = 5 * 10.0^(-14)
        # Test Fallbacks -> we can't do steps with the wrong combination
        p = SubGradientProblem(M, F, gradF)
        o = GradientDescentState(
            M, s[1]; stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(M)
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
        @test_logs (:warn,) (:warn,) (:warn,) gradient_descent(
            M,
            F,
            gradF,
            1 / sqrt(2) .* [1.0, -1.0, 0.0];
            debug=[DebugWarnIfCostIncreases(:Once)],
        )
    end
end
