using Manopt, Test, ManifoldsBase, Dates, Manifolds

struct TestPolarManifold <: AbstractManifold{ℝ} end

function ManifoldsBase.default_inverse_retraction_method(::TestPolarManifold)
    return PolarInverseRetraction()
end

struct TestDebugAction <: DebugAction end

@testset "Debug State" begin
    # helper to get debug as string
    @testset "Basic Debug Output" begin
        io = IOBuffer()
        M = ManifoldsBase.DefaultManifold(2)
        p = [4.0, 2.0]
        st = GradientDescentState(
            M, p; stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(M)
        )
        f(M, q) = distance(M, q, p) .^ 2
        grad_f(M, q) = -2 * log(M, q, p)
        # summary fallback to show
        @test Manopt.status_summary(TestDebugAction()) === "TestDebugAction()"
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        a1 = DebugDivider("|"; io=io)
        @test Manopt.dispatch_state_decorator(DebugSolverState(st, a1)) === Val{true}()
        # constructors
        @test DebugSolverState(st, a1).debugDictionary[:All] == a1
        @test DebugSolverState(st, [a1]).debugDictionary[:All].group[1] == a1
        @test DebugSolverState(st, Dict(:A => a1)).debugDictionary[:A] == a1
        @test DebugSolverState(st, ["|"]).debugDictionary[:All].group[1].divider ==
            a1.divider
        @test endswith(repr(DebugSolverState(st, a1)), "\"|\"")
        @test repr(DebugSolverState(st, Dict{Symbol,DebugAction}())) == repr(st)
        # single AbstractStateActions
        # DebugDivider
        a1(mp, st, 0)
        s = @test String(take!(io)) == "|"
        DebugGroup([a1, a1])(mp, st, 0)
        @test String(take!(io)) == "||"
        DebugEvery(a1, 10, false)(mp, st, 9)
        @test String(take!(io)) == ""
        DebugEvery(a1, 10, true)(mp, st, 10)
        @test String(take!(io)) == "|"
        @test DebugEvery(a1, 10, true)(mp, st, -1) == nothing
        # Debug Cost
        @test DebugCost(; format="A %f").format == "A %f"
        DebugCost(; long=false, io=io)(mp, st, 0)
        @test String(take!(io)) == "F(x): 0.000000"
        DebugCost(; long=false, io=io)(mp, st, -1)
        @test String(take!(io)) == ""
        # entry
        DebugEntry(:p; prefix="x:", io=io)(mp, st, 0)
        @test String(take!(io)) == "x: $p"
        DebugEntry(:p; prefix="x:", io=io)(mp, st, -1)
        @test String(take!(io)) == ""
        # Change of Iterate and recording a custom field
        a2 = DebugChange(;
            storage=StoreStateAction(M; store_points=Tuple{:Iterate}, p_init=p),
            prefix="Last: ",
            io=io,
        )
        a2(mp, st, 0) # init
        st.p = [3.0, 2.0]
        a2(mp, st, 1)
        a2inv = DebugChange(;
            storage=StoreStateAction(M; store_fields=[:Iterate]),
            prefix="Last: ",
            io=io,
            inverse_retraction_method=PolarInverseRetraction(),
        )
        a2mani = DebugChange(
            TestPolarManifold();
            storage=StoreStateAction([:Iterate]),
            prefix="Last: ",
            io=io,
        )
        @test a2inv.inverse_retraction_method === PolarInverseRetraction()
        @test a2mani.inverse_retraction_method === PolarInverseRetraction()
        @test a2.inverse_retraction_method === LogarithmicInverseRetraction()
        @test String(take!(io)) == "Last: 1.000000"
        a3 = DebugGradientChange(;
            storage=StoreStateAction([:Gradient, :Iterate]), prefix="Last: ", io=io
        )
        a3(mp, st, 0) # init
        st.X = [1.0, 0.0]
        a3(mp, st, 1)
        @test String(take!(io)) == "Last: 1.000000"
        # Iterate
        DebugIterate(; io=io)(mp, st, 0)
        @test String(take!(io)) == ""
        DebugIterate(; io=io)(mp, st, 1)
        @test String(take!(io)) == "p: $(st.p)"
        # Iteration
        DebugIteration(; io=io)(mp, st, 0)
        @test String(take!(io)) == "Initial "
        DebugIteration(; io=io)(mp, st, 23)
        @test String(take!(io)) == "# 23    "
        @test repr(DebugIteration()) == "DebugIteration(; format=\"# %-6d\")"
        @test Manopt.status_summary(DebugIteration()) == "(:Iteration, \"# %-6d\")"
        # DebugEntryChange
        dec = DebugEntryChange(:p, x -> x)
        @test startswith(repr(dec), "DebugEntryChange(:p")
        # DEbugEntryChange - reset
        st.p = p
        a3 = DebugEntryChange(
            :p,
            (mp, o, x, y) -> distance(Manopt.get_manifold(mp), x, y);
            prefix="Last: ",
            io,
        )
        a4 = DebugEntryChange(
            :p,
            (mp, o, x, y) -> distance(Manopt.get_manifold(mp), x, y);
            initial_value=p,
            format="Last: %1.1f",
            io,
        )
        a3(mp, st, 0) # init
        @test String(take!(io)) == ""
        a4(mp, st, 0) # init
        @test String(take!(io)) == ""
        #change
        st.p = [3.0, 2.0]
        a3(mp, st, 1)
        @test String(take!(io)) == "Last: 1.0"
        a4(mp, st, 1)
        @test String(take!(io)) == "Last: 1.0"
        # StoppingCriterion
        DebugStoppingCriterion(; io=io)(mp, st, 1)
        @test String(take!(io)) == ""
        st.stop(mp, st, 19)
        DebugStoppingCriterion(; io=io)(mp, st, 19)
        @test String(take!(io)) == ""
        st.stop(mp, st, 20)
        DebugStoppingCriterion(; io=io)(mp, st, 20)
        @test String(take!(io)) ==
            "The algorithm reached its maximal number of iterations (20).\n"
        @test repr(DebugStoppingCriterion()) == "DebugStoppingCriterion()"
        @test Manopt.status_summary(DebugStoppingCriterion()) == ":Stop"

        # Factory
        df = DebugFactory([:Stop, "|"])
        @test isa(df[:Stop], DebugStoppingCriterion)
        @test isa(df[:All], DebugGroup)
        @test isa(df[:All].group[1], DebugDivider)
        @test length(df[:All].group) == 1
        df = DebugFactory([:Stop, "|", 20])
        @test isa(df[:All], DebugEvery)
        s = [
            :Change,
            :GradientChange,
            :Iteration,
            :Iterate,
            :Cost,
            :Stepsize,
            :p,
            :Time,
            :IterativeTime,
        ]
        @test all(
            isa.(
                DebugFactory(s)[:All].group,
                [
                    DebugChange,
                    DebugGradientChange,
                    DebugIteration,
                    DebugIterate,
                    DebugCost,
                    DebugStepsize,
                    DebugEntry,
                    DebugTime,
                    DebugTime,
                ],
            ),
        )
        @test DebugActionFactory((:IterativeTime)).mode == :Iterative
        @test all(
            isa.(
                DebugFactory([(t, "A") for t in s])[:All].group,
                [
                    DebugChange,
                    DebugGradientChange,
                    DebugIteration,
                    DebugIterate,
                    DebugCost,
                    DebugStepsize,
                    DebugEntry,
                    DebugTime,
                    DebugTime,
                ],
            ),
        )
        @test DebugActionFactory(a3) == a3
        @test DebugFactory([(:Iterate, "A")])[:All].group[1].format == "A"
        @test DebugActionFactory((:Iterate, "A")).format == "A"
        # Status for multiple dictionaries
        dss = DebugSolverState(st, DebugFactory([:Stop, 20, "|"]))
        @test contains(Manopt.status_summary(dss), ":Stop")
        @test Manopt.get_message(dss) == ""
        # DebugEvery summary
        de = DebugEvery(DebugGroup([DebugDivider("|"), DebugIteration()]), 10)
        @test Manopt.status_summary(de) == "[\"|\", (:Iteration, \"# %-6d\"), 10]"
        # DebugGradientChange
        dgc = DebugGradientChange()
        dgc_s = "DebugGradientChange(; format=\"Last Change: %f\", vector_transport_method=ParallelTransport())"
        @test repr(dgc) == dgc_s
        @test Manopt.status_summary(dgc) == "(:GradientChange, \"Last Change: %f\")"
        # Faster storage
        dgc2 = DebugGradientChange(Euclidean(2))
        @test repr(dgc2) == dgc_s
    end

    @testset "Debug Warnings" begin
        M = ManifoldsBase.DefaultManifold(2)
        p = [4.0, 2.0]
        st = GradientDescentState(
            M, p; stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(M)
        )
        f(M, y) = Inf
        grad_f(M, y) = Inf .* ones(2)
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))

        w1 = DebugWarnIfCostNotFinite()
        @test repr(w1) == "DebugWarnIfCostNotFinite()"
        @test Manopt.status_summary(w1) == ":WarnCost"
        @test_logs (:warn,) (
            :warn,
            "Further warnings will be supressed, use DebugWarnIfCostNotFinite(:Always) to get all warnings.",
        ) w1(mp, st, 0)
        w2 = DebugWarnIfCostNotFinite(:Always)
        @test_logs (
            :warn, "The cost is not finite.\nAt iteration #0 the cost evaluated to Inf."
        ) w2(mp, st, 0)

        st.X = grad_f(M, p)
        w3 = DebugWarnIfFieldNotFinite(:X)
        @test repr(w3) == "DebugWarnIfFieldNotFinite(:X)"
        @test_logs (:warn,) (
            :warn,
            "Further warnings will be supressed, use DebugWaranIfFieldNotFinite(:X, :Always) to get all warnings.",
        ) w3(mp, st, 0)
        w4 = DebugWarnIfFieldNotFinite(:X, :Always)
        @test_logs (
            :warn,
            "The field s.X is or contains values that are not finite.\nAt iteration #1 it evaluated to [Inf, Inf].",
        ) w4(mp, st, 1)
        w5 = DebugWarnIfFieldNotFinite(:Gradient, :Always)
        @test_logs (
            :warn,
            "The gradient is or contains values that are not finite.\nAt iteration #1 it evaluated to [Inf, Inf].",
        ) w5(mp, st, 1)

        st.p = Inf .* ones(2)
        w6 = DebugWarnIfFieldNotFinite(:Iterate, :Always)
        @test_logs (
            :warn,
            "The iterate is or contains values that are not finite.\nAt iteration #1 it evaluated to [Inf, Inf].",
        ) w6(mp, st, 1)

        df1 = DebugFactory([:WarnCost])
        @test isa(df1[:All].group[1], DebugWarnIfCostNotFinite)
        df2 = DebugFactory([:WarnGradient])
        @test isa(df2[:All].group[1], DebugWarnIfFieldNotFinite)
    end
    @testset "Debug Time" begin
        io = IOBuffer()
        M = ManifoldsBase.DefaultManifold(2)
        p = [4.0, 2.0]
        st = GradientDescentState(
            M, p; stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(M)
        )
        f(M, q) = distance(M, q, p) .^ 2
        grad_f(M, q) = -2 * log(M, q, p)
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        d1 = DebugTime(; start=true, io=io)
        @test d1.last_time != Nanosecond(0)
        d2 = DebugTime(; io=io)
        @test d2.last_time == Nanosecond(0)
        d2(mp, st, 1)
        @test d2.last_time != Nanosecond(0) # changes on first call
        t = d2.last_time
        sleep(0.002)
        d2(mp, st, 2)
        @test t == d2.last_time # but not afterwards
        @test endswith(String(take!(io)), "seconds")
        d3 = DebugTime(; start=true, mode=:iterative, io=io)
        @test d3.last_time != Nanosecond(0) # changes on first call
        t = d3.last_time
        d3(mp, st, 2)
        @test t != d3.last_time # and later as well
        t = d3.last_time
        sleep(0.002)
        Manopt.reset!(d3)
        @test t != d3.last_time
        Manopt.stop!(d3)
        @test d3.last_time == Nanosecond(0)
        drs = "DebugTime(; format=\"time spent: %s\", mode=:cumulative)"
        @test repr(DebugTime()) == drs
        drs2 = "(:IterativeTime, \"time spent: %s\")"
        @test Manopt.status_summary(DebugTime(; mode=:iterative)) == drs2
        drs3 = "(:Time, \"time spent: %s\")"
        @test Manopt.status_summary(DebugTime(; mode=:cumulative)) == drs3
    end
    @testset "Debug show/summaries" begin
        d1 = DebugDivider("|")
        d2 = DebugIterate()
        d3 = DebugGroup([d1, d2])
        @test repr(d3) == "DebugGroup([$(d1), $(d2)])"
        ts = "[ $(Manopt.status_summary(d1)), $(Manopt.status_summary(d2)) ]"
        @test Manopt.status_summary(d3) == ts

        d4 = DebugEvery(d1, 4)
        @test repr(d4) == "DebugEvery($(d1), 4, true)"
        @test Manopt.status_summary(d4) === "[$(d1), 4]"

        ts2 = "DebugChange(; format=\"Last Change: %f\", inverse_retraction=LogarithmicInverseRetraction())"
        @test repr(DebugChange()) == ts2
        @test Manopt.status_summary(DebugChange()) == "(:Change, \"Last Change: %f\")"
        # check that a nondefault manifold works as well - not sure how to test this then
        d = DebugChange(Euclidean(2))

        @test repr(DebugCost()) == "DebugCost(; format=\"F(x): %f\")"
        @test Manopt.status_summary(DebugCost()) == "(:Cost, \"F(x): %f\")"

        @test repr(DebugDivider("|")) == "DebugDivider(; divider=\"|\")"
        @test Manopt.status_summary(DebugDivider("a")) == "\"a\""

        @test repr(DebugEntry(:a)) == "DebugEntry(:a; format=\"a: %s\")"

        @test repr(DebugStepsize()) == "DebugStepsize(; format=\"s:%s\")"
        @test Manopt.status_summary(DebugStepsize()) == "(:Stepsize, \"s:%s\")"

        @test repr(DebugGradientNorm()) == "DebugGradientNorm(; format=\"|grad f(p)|:%s\")"
        dgn_s = "(:GradientNorm, \"|grad f(p)|:%s\")"
        @test Manopt.status_summary(DebugGradientNorm()) == dgn_s

        @test repr(DebugGradient()) == "DebugGradient(; format=\"grad f(p):%s\")"
        dg_s = "(:Gradient, \"grad f(p):%s\")"
        @test Manopt.status_summary(DebugGradient()) == dg_s
    end
end
