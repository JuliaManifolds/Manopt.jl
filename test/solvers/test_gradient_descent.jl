using Manopt, Manifolds, Test, Random

using ManifoldDiff: grad_distance

@testset "Gradient Descent" begin
    @testset "allocating Circle" begin
        M = Circle()
        data = [-π / 2, π / 4, 0.0, π / 4]
        apprpstar = sum([-π / 2, π / 4, 0.0, π / 4]) / length(data)
        f(M, p) = 1 / 10 * sum(distance.(Ref(M), data, Ref(p)) .^ 2)
        grad_f(M, p) = 1 / 5 * sum(-log.(Ref(M), Ref(p), data))
        my_io = IOBuffer()
        d = DebugEvery(
            DebugGroup(
                [
                    DebugIterate(; io = my_io),
                    DebugDivider(" "; io = my_io),
                    DebugCost(; io = my_io),
                    DebugStoppingCriterion(; io = my_io),
                    DebugDivider("\n"; io = my_io),
                    DebugMessages(),
                ]
            ),
            500,
        )
        s = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion = StopAfterIteration(200) | StopWhenChangeLess(M, 1.0e-16),
            stepsize = ArmijoLinesearch(; contraction_factor = 0.99),
            debug = d,
            record = [:Iteration, :Cost, 1],
            return_state = true,
        )
        p = get_solver_result(s)[]
        res_debug = String(take!(my_io))
        @test res_debug === "p: [-1.5707963267948966] f(x): 1.357071\n"
        p2 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion = StopAfterIteration(200) | StopWhenChangeLess(M, 1.0e-16),
            stepsize = ArmijoLinesearch(; contraction_factor = 0.99),
        )
        @test p == p2
        step = NonmonotoneLinesearch(;
            stepsize_reduction = 0.99,
            sufficient_decrease = 0.1,
            memory_size = 2,
            bb_min_stepsize = 1.0e-7,
            bb_max_stepsize = π / 2,
            strategy = :direct,
            stop_when_stepsize_exceeds = 0.9 * π,
        )
        p3 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion = StopAfterIteration(1000) | StopWhenChangeLess(M, 1.0e-16),
            stepsize = step,
            debug = [], # do not warn about increasing step here
        )
        @test isapprox(M, p, p3; atol = 1.0e-13)
        step2 = NonmonotoneLinesearch(;
            stepsize_reduction = 0.99,
            sufficient_decrease = 0.1,
            memory_size = 2,
            bb_min_stepsize = 1.0e-7,
            bb_max_stepsize = π / 2,
            strategy = :inverse,
            stop_when_stepsize_exceeds = 0.9 * π,
        )
        p4 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion = StopAfterIteration(1000) | StopWhenChangeLess(M, 1.0e-16),
            stepsize = step2,
            debug = [], # do not warn about increasing step here
        )
        @test isapprox(M, p, p4; atol = 1.0e-13)
        step3 = NonmonotoneLinesearch(;
            stepsize_reduction = 0.99,
            sufficient_decrease = 0.1,
            memory_size = 2,
            bb_min_stepsize = 1.0e-7,
            bb_max_stepsize = π / 2,
            strategy = :alternating,
            stop_when_stepsize_exceeds = 0.9 * π,
        )
        p5 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion = StopAfterIteration(1000) | StopWhenChangeLess(M, 1.0e-16),
            stepsize = step3,
            debug = [], # do not warn about increasing step here
        )
        @test isapprox(M, p, p5; atol = 1.0e-13)
        p6 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion = StopAfterIteration(1000) | StopWhenChangeLess(M, 1.0e-16),
            direction = Nesterov(; p = copy(M, data[1])),
        )
        @test isapprox(M, p, p6; atol = 1.0e-13)
        # Precon in simple scale down by 2
        p7 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion = StopAfterIteration(1000) | StopWhenChangeLess(M, 1.0e-16),
            direction = PreconditionedDirection((M, p, X) -> 0.5 .* X),
        )
        @test isapprox(M, p, p7; atol = 1.0e-13)
        # Precon in simple scale down by 2 – inplace
        p8 = gradient_descent(
            M,
            f,
            grad_f,
            data[1];
            stopping_criterion = StopAfterIteration(1000) | StopWhenChangeLess(M, 1.0e-16),
            direction = PreconditionedDirection(
                (M, Y, p, X) -> (Y .= 0.5 .* X); evaluation = InplaceEvaluation()
            ),
        )
        @test isapprox(M, p, p8; atol = 1.0e-13)
        M2 = Euclidean()
        @test_logs (
            :warn,
            string(
                "The strategy 'indirect' is not defined. The 'direct' strategy is used instead.",
            ),
        ) NonmonotoneLinesearch(; strategy = :indirect)(M2)
        @test_throws DomainError NonmonotoneLinesearch(Euclidean(); bb_min_stepsize = 0.0)(M2)
        @test_throws DomainError NonmonotoneLinesearch(;
            bb_min_stepsize = π / 2, bb_max_stepsize = π / 4
        )(
            M2
        )
        @test_throws DomainError NonmonotoneLinesearch(; memory_size = 0)(M2)

        rec = get_record(s)
        # after one step for local enough data -> equal to real valued data
        @test isapprox(M, p, apprpstar, atol = 5.0e-10)
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
        # `gradient_descent` allocated n2 newly
        @test !isapprox(M, pts[1], n2)
        @test isapprox(M, north, n2)
        Random.seed!(43)
        n2a = gradient_descent(M, f, grad_f)
        # `gradient_descent` allocated n2 newly
        @test isapprox(M, north, n2a)
        n3 = gradient_descent(
            M,
            f,
            grad_f,
            pts[1];
            direction = MomentumGradient(),
            stepsize = ConstantLength(),
            debug = [], # do not warn about increasing step here
        )
        @test isapprox(M, north, n3)
        n4 = gradient_descent(
            M,
            f,
            grad_f,
            pts[1];
            direction = AverageGradient(M; n = 5),
            stopping_criterion = StopAfterIteration(800),
        )
        @test isapprox(M, north, n4; atol = 1.0e-7)
        n5 = copy(M, pts[1])
        r = gradient_descent!(M, f, grad_f, n5; return_state = true)
        @test isapprox(M, n5, n2)
        @test startswith(repr(r), "# Solver state for `Manopt.jl`s Gradient Descent")
        # State and a count objective, putting stats behind print
        n6 = gradient_descent(
            M,
            f,
            grad_f,
            pts[1];
            count = [:Gradient],
            return_objective = true,
            return_state = true,
        )
        @test repr(n6) == "$(n6[2])\n\n$(n6[1])"
    end
    @testset "Tutorial mode" begin
        M = Sphere(2)
        q = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
        f(M, p) = distance(M, p, q) .^ 2
        # choose a wrong gradient such that ConstantStepsize yields an increase
        grad_f(M, p) = -grad_distance(M, q, p)
        @test_logs (:info,) Manopt.set_parameter!(:Mode, "Tutorial")
        @test_logs (:warn,) (:warn,) (:warn,) gradient_descent(
            M, f, grad_f, 1 / sqrt(2) .* [1.0, -1.0, 0.0]; stepsize = ConstantLength()
        )
        grad_f2(M, p) = 20 * grad_distance(M, q, p)
        @test_logs (:warn,) (:warn,) gradient_descent(
            M, f, grad_f2, 1 / sqrt(2) .* [1.0, -1.0, 0.0]
        )
        @test_logs (:info,) Manopt.set_parameter!(:Mode, "")
    end
end
