using Manopt, Manifolds, Test

@testset "Gradient Descent" begin
    @testset "allocating Circle" begin
        M = Circle()
        data = [-π / 2, π / 4, 0.0, π / 4]
        apprpstar = sum([-π / 2, π / 4, 0.0, π / 4]) / length(data)
        f(M, p) = 1 / 10 * sum(distance.(Ref(M), data, Ref(p)) .^ 2)
        grad_f(M, p) = 1 / 5 * sum(-log.(Ref(M), Ref(p), data))
        my_io = IOBuffer()
        d = DebugEvery(
            DebugGroup([
                DebugIterate(; io=my_io),
                DebugDivider(" "; io=my_io),
                DebugCost(; io=my_io),
                DebugStoppingCriterion(; io=my_io),
                DebugDivider("\n"; io=my_io),
                DebugMessages(),
            ]),
            500,
        )
        s = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion=StopAfterIteration(200) | StopWhenChangeLess(1e-16),
            stepsize=ArmijoLinesearch(M; contraction_factor=0.99),
            debug=d,
            record=[:Iteration, :Cost, 1],
            return_state=true,
        )
        p = get_solver_result(s)[]
        res_debug = String(take!(my_io))
        @test res_debug === " F(x): 1.357071\n"
        p2 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion=StopAfterIteration(200) | StopWhenChangeLess(1e-16),
            stepsize=ArmijoLinesearch(M; contraction_factor=0.99),
        )
        @test p == p2
        step = NonmonotoneLinesearch(
            M;
            stepsize_reduction=0.99,
            sufficient_decrease=0.1,
            memory_size=2,
            bb_min_stepsize=1e-7,
            bb_max_stepsize=π / 2,
            strategy=:direct,
            stop_when_stepsize_exceeds=0.9 * π,
        )
        p3 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion=StopAfterIteration(1000) | StopWhenChangeLess(1e-16),
            stepsize=step,
            debug=[], # do not warn about increasing step here
        )
        @test isapprox(M, p, p3; atol=1e-13)
        step.strategy = :inverse
        p4 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion=StopAfterIteration(1000) | StopWhenChangeLess(1e-16),
            stepsize=step,
            debug=[], # do not warn about increasing step here
        )
        @test isapprox(M, p, p4; atol=1e-13)
        step.strategy = :alternating
        p5 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion=StopAfterIteration(1000) | StopWhenChangeLess(1e-16),
            stepsize=step,
            debug=[], # do not warn about increasing step here
        )
        @test isapprox(M, p, p5; atol=1e-13)
        p6 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion=StopWhenAny(
                StopAfterIteration(1000), StopWhenChangeLess(1e-16)
            ),
            direction=Nesterov(M, data[1]),
        )
        @test isapprox(M, p, p6; atol=1e-13)

        @test_logs (
            :warn,
            string(
                "The strategy 'indirect' is not defined. The 'direct' strategy is used instead.",
            ),
        ) NonmonotoneLinesearch(Euclidean(); strategy=:indirect)
        @test_throws DomainError NonmonotoneLinesearch(Euclidean(); bb_min_stepsize=0.0)
        @test_throws DomainError NonmonotoneLinesearch(
            Euclidean(); bb_min_stepsize=π / 2, bb_max_stepsize=π / 4
        )
        @test_throws DomainError NonmonotoneLinesearch(Euclidean(); memory_size=0)

        rec = get_record(s)
        # after one step for local enough data -> equal to real valued data
        @test isapprox(M, p, apprpstar, atol=5e-10)
    end
    @testset "mutating Sphere" begin
        M = Sphere(2)
        north = [0.0, 0.0, 1.0]
        pre_pts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        pts = exp.(Ref(M), Ref(north), pre_pts)
        f(M, p) = 1 / 8 * sum(distance.(Ref(M), pts, Ref(p)) .^ 2)
        grad_f(M, p) = 1 / 4 * sum(-log.(Ref(M), Ref(p), pts))
        n2 = gradient_descent(M, f, grad_f, pts[1])
        n2a = gradient_descent(M, f, grad_f)
        # Since we called gradient_descent n2 is newly allocated
        @test !isapprox(M, pts[1], n2)
        @test isapprox(M, north, n2)
        n2a = gradient_descent(M, f, grad_f)
        # Since we called gradient_descent n2 is newly allocated
        @test isapprox(M, north, n2a)
        n3 = gradient_descent(
            M,
            f,
            grad_f,
            pts[1];
            direction=MomentumGradient(M, copy(M, pts[1])),
            stepsize=ConstantStepsize(M),
            debug=[], # do not warn about increasing step here
        )
        @test isapprox(M, north, n3)
        n4 = gradient_descent(
            M, f, grad_f, pts[1]; direction=AverageGradient(M, copy(M, pts[1]); n=5)
        )
        @test isapprox(M, north, n4; atol=1e-7)
        n5 = copy(M, pts[1])
        r = gradient_descent!(M, f, grad_f, n5; return_state=true)
        @test isapprox(M, n5, n2)
        @test startswith(repr(r), "# Solver state for `Manopt.jl`s Gradient Descent")
    end
    @testset "Warning when cost increases" begin
        M = Sphere(2)
        q = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
        f(M, p) = distance(M, p, q) .^ 2
        # chosse a wrong gradient such that ConstantStepsize yields an increase
        grad_f(M, p) = -grad_distance(M, q, p)
        # issues three warnings
        @test_logs (:warn,) (:warn,) (:warn,) gradient_descent(
            M,
            f,
            grad_f,
            1 / sqrt(2) .* [1.0, -1.0, 0.0];
            stepsize=ConstantStepsize(1.0),
            debug=[DebugWarnIfCostIncreases(:Once), :Messages],
        )
    end
end
