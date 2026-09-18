using Manifolds, ManifoldsBase, Manopt, Test
using Dates

struct NoIterateState <: AbstractManoptSolverState end

# a step size and a (sub) state that issue a message
struct MessageTestStepsize <: Manopt.Stepsize end
Manopt.get_message(::MessageTestStepsize) = "step message"
struct MessageTestState <: AbstractManoptSolverState end
Manopt.get_message(::MessageTestState) = "sub message"

@testset "Manopt Solver States" begin
    @testset "Generic State" begin
        M = Euclidean(3)
        s = Manopt.Test.DummyState()
        @test repr(Manopt.ReturnSolverState(s)) == "ReturnSolverState($s)"
        srst = "A Manopt Test state with storage Float64[]"
        @test Manopt.status_summary(Manopt.ReturnSolverState(s)) == srst
        io = IOBuffer()
        show(io, MIME"text/plain"(), Manopt.ReturnSolverState(s))
        @test startswith(String(take!(io)), srst)

        a = ArmijoLinesearch(; initial_stepsize = 1.0)(M)
        @test get_last_stepsize(a) == 1.0
        @test get_initial_stepsize(a) == 1.0
        @test Manopt.set_parameter!(s, :Dummy, 1) === s
    end

    @testset "Decreasing Stepsize" begin
        M = Euclidean(3)
        dec_step = DecreasingLength(; length = 10.0, factor = 1.0, subtrahend = 0.0, exponent = 1.0)(
            M
        )
        @test get_initial_stepsize(dec_step) == 10.0
        M = Euclidean(3)
        pr = Manopt.Test.DummyProblem{typeof(M)}()
        @test dec_step(pr, Manopt.Test.DummyState(), 1) == 10.0
        @test dec_step(pr, Manopt.Test.DummyState(), 2) == 5.0
    end

    @testset "Decorator State" begin
        s = Manopt.Test.DummyState(zeros(3))
        r = RecordSolverState(s, RecordIteration())
        d = DebugSolverState(s, DebugIteration())
        ret = Manopt.ReturnSolverState(s)
        dr = DebugSolverState(r, DebugIteration())
        M = Euclidean()

        @test has_record(s) == has_record(d)
        @test !has_record(s)
        @test has_record(r)
        @test has_record(r) == has_record(dr)

        @test is_state_decorator(r)
        @test is_state_decorator(dr)
        @test is_state_decorator(d)
        @test !is_state_decorator(s)

        @test dispatch_state_decorator(r) === Val(true)
        @test dispatch_state_decorator(dr) === Val(true)
        @test dispatch_state_decorator(ret) === Val(true)
        @test dispatch_state_decorator(d) === Val(true)
        @test dispatch_state_decorator(s) === Val(false)

        @test get_state(r) == s
        @test get_state(dr) == s
        @test get_state(d) == s
        @test get_state(s) == s
        @test Manopt._get_state(s, Val(false)) == s

        @test Manopt._extract_val(Val(true))
        @test !Manopt._extract_val(Val(false))

        @test_throws ErrorException get_gradient(s)
        @test_throws ErrorException get_gradient(r)
        @test isnan(get_iterate(s)) # dummy returns not-a-number
        @test isnan(get_iterate(r)) # dummy returns not-a-number
        @test_throws ErrorException set_iterate!(s, M, 0)
        @test_throws ErrorException set_iterate!(r, M, 0)
        s2 = NoIterateState()
        @test_throws ErrorException get_iterate(s2)
    end

    @testset "Iterate and Gradient setters" begin
        M = Euclidean(3)
        s1 = NelderMeadState(M)
        s2 = GradientDescentState(M)
        p = 3.0 * ones(3)
        X = ones(3)
        d1 = DebugSolverState(s1, DebugIteration())
        set_iterate!(d1, M, p)
        @test d1.state.p == 3 * ones(3)
        @test_throws ErrorException set_gradient!(d1, M, p, X)

        d2 = DebugSolverState(s2, DebugIteration())
        set_iterate!(d2, M, p)
        @test d2.state.p == 3 * ones(3)
        set_gradient!(d2, M, p, X)
        @test d2.state.X == ones(3)
        @test get_stopping_criterion(d2) === s2.stop
        @test has_converged(d2) === has_converged(s2)
    end

    @testset "Generic Objective and State solver returns" begin
        f(M, p) = 1
        o = ManifoldCostObjective(f)
        ro = Manopt.ReturnManifoldObjective(o)
        ddo = Manopt.Test.DummyDecoratedObjective(o)
        s = Manopt.Test.DummyState()
        rs = Manopt.ReturnSolverState(s)
        # parameters are passed through the decorator
        @test Manopt.set_parameter!(rs, :Dummy, 1) === rs
        @test isnothing(Manopt.get_parameter(rs, :Dummy))
        @test Manopt.get_solver_return(o, rs) == s #no ReturnManifoldObjective
        # Return O & S
        (a, b) = Manopt.get_solver_return(ro, rs)
        @test a == o
        @test b == s
        # Return just S
        @test Manopt.get_solver_return(ddo, rs) == s
        # both as tuples and they return the iterate
        @test isnan(get_solver_result((ro, rs)))
        @test isnan(get_solver_result((o, rs)))
        @test isnan(get_solver_result(ro, rs))
        @test isnan(get_solver_result(o, rs))
        # But also if the second is already some other type
        @test isnan(get_solver_result((ro, NaN)))
        @test isnan(get_solver_result((o, NaN)))
        @test isnan(get_solver_result(ro, NaN))
        @test isnan(get_solver_result(o, NaN))
        # unless overwritten, objectives to not display in these tuples.
        @test repr((o, s)) == repr(s)
        # test Pass down
        @test repr((ro, s)) == repr(s)
    end
    @testset "Decorator keywords accept concrete dictionaries" begin
        M = Euclidean(2)
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 .* p
        p = [1.0, 2.0]
        sc = StopAfterIteration(1)
        io = IOBuffer()
        # inferred as `Dict{Symbol, DebugStoppingCriterion}` / `Dict{Symbol, RecordIteration}`
        @test is_point(M, gradient_descent(M, f, grad_f, p; stopping_criterion = sc, debug = Dict(:Stop => DebugStoppingCriterion(; io = io))))
        @test is_point(M, gradient_descent(M, f, grad_f, p; stopping_criterion = sc, record = Dict(:Iteration => RecordIteration())))
        # and the explicitly typed form keeps working
        @test is_point(M, gradient_descent(M, f, grad_f, p; stopping_criterion = sc, debug = Dict{Symbol, DebugAction}(:Stop => DebugStoppingCriterion(; io = io))))
    end
    @testset "has_sub_problem" begin
        for S in (
                AdaptiveRegularizationState, AugmentedLagrangianMethodState, ConvexBundleMethodState,
                DifferenceOfConvexProximalState, DifferenceOfConvexState, ExactPenaltyMethodState,
                FrankWolfeState, GradientSamplingState, InteriorPointNewtonState, LevenbergMarquardtState,
                ProximalBundleMethodState, ProximalGradientMethodState, TrustRegionsState, VectorBundleNewtonState,
            )
            @test Manopt.has_sub_problem(S)
        end
        @test !Manopt.has_sub_problem(GradientDescentState)
        @test_throws ErrorException Manopt.get_sub_state(GradientDescentState(Euclidean(2)))
        @test_throws ErrorException Manopt.get_sub_problem(GradientDescentState(Euclidean(2)))
    end
    @testset "Decorator pass-through of solver and parameter functions" begin
        M = Euclidean(2)
        f(M, p) = sum(p .^ 2)
        grad_f(M, p) = 2 .* p
        p = [1.0, 2.0]
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        # `ReturnSolverState` forwards `initialize_solver!` and `step_solver!` to its state
        gds = GradientDescentState(M; p = copy(p))
        rets = Manopt.ReturnSolverState(gds)
        Manopt.initialize_solver!(mp, rets)
        @test get_gradient(gds) == grad_f(M, p)
        Manopt.step_solver!(mp, rets, 1)
        @test get_iterate(gds) != p
        # `:StoppingCriterion` is passed down from a state and through both decorators
        for (state, inner) in (
                (s0 = GradientDescentState(M; p = copy(p), stopping_criterion = StopAfterIteration(5)); (s0, s0)),
                (s1 = GradientDescentState(M; p = copy(p), stopping_criterion = StopAfterIteration(5)); (DebugSolverState(s1, DebugDivider("")), s1)),
                (s2 = GradientDescentState(M; p = copy(p), stopping_criterion = StopAfterIteration(5)); (RecordSolverState(s2, RecordIteration()), s2)),
                (s3 = GradientDescentState(M; p = copy(p), stopping_criterion = StopAfterIteration(5)); (Manopt.ReturnSolverState(s3), s3)),
            )
            Manopt.set_parameter!(state, Val(:StoppingCriterion), :MaxIteration, 7)
            @test inner.stop.max_iterations == 7
        end
        # and through nested decorators, where the return state is the outermost one
        s4 = GradientDescentState(M; p = copy(p), stopping_criterion = StopAfterIteration(5))
        r4 = Manopt.ReturnSolverState(DebugSolverState(s4, DebugDivider("")))
        Manopt.set_parameter!(r4, Val(:StoppingCriterion), :MaxIteration, 9)
        @test s4.stop.max_iterations == 9
    end
    @testset "Messages of step sizes and sub states are passed on" begin
        M = Euclidean(3)
        p = [1.0, 2.0, 3.0]
        @test Manopt.get_message(SubGradientMethodState(M; p = p, stepsize = MessageTestStepsize())) == "step message"
        pgms = ProjectedGradientMethodState(M; p = p, stepsize = MessageTestStepsize(), backtrack = MessageTestStepsize())
        @test Manopt.get_message(pgms) == "step message\nstep message"
        sub_problem = DefaultManoptProblem(TangentSpace(M, p), ManifoldCostObjective((TpM, X) -> 0.0))
        @test Manopt.get_message(AdaptiveRegularizationState(M, sub_problem, MessageTestState(); p = p)) == "sub message"
        @test Manopt.get_message(NoIterateState()) == ""
    end
    @testset "Display of the box constrained Levenberg-Marquardt sub solver state" begin
        M = Euclidean(3)
        p = [1.0, 2.0, 3.0]
        lmbs = Manopt.LevenbergMarquardtBoxSubsolver(M, GradientDescentState(M; p = copy(p)), p)
        @test startswith(repr(lmbs), "LevenbergMarquardtBoxSubsolver(GradientDescentState(")
        @test endswith(repr(lmbs), "last_gcd_result = :not_searched, last_gcd_stepsize = NaN)")
        ss = Manopt.status_summary(lmbs)
        @test startswith(ss, "# Solver state for a box constrained Levenberg-Marquardt subproblem\n")
        @test occursin("| ## Solver state for `Manopt.jl`s Gradient Descent", ss)
        @test Manopt.status_summary(lmbs; context = :inline) == repr(lmbs)
    end
    @testset "set_iterate! for states that store an iterate" begin
        M = Euclidean(3)
        p = [1.0, 2.0, 3.0]
        q = [4.0, 5.0, 6.0]
        sub_problem = DefaultManoptProblem(TangentSpace(M, p), ManifoldCostObjective((TpM, X) -> 0.0))
        states = [
            StochasticGradientDescentState(M; p = copy(p)),
            AlternatingGradientDescentState(M; p = copy(p)),
            GradientSamplingState(M; p = copy(p)),
            ProjectedGradientMethodState(M; p = copy(p)),
            MeshAdaptiveDirectSearchState(M; p = copy(p)),
            LevenbergMarquardtState(M, sub_problem, MessageTestState(), zeros(2); p = copy(p)),
            CMAESState(
                M, copy(p), 2, 5, 1.5, 0.1, 0.2, 0.3, 0.4, 1.0, 1.2, StopAfterIteration(1),
                [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0], 1.0, [0.6, 0.4, 0.0, -0.3, -0.7],
            ),
        ]
        for st in states
            stored = get_iterate(st)
            @test set_iterate!(st, M, q) === st
            @test get_iterate(st) == q
            @test get_iterate(st) === stored
        end
    end
end
