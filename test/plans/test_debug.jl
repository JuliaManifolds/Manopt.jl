using Manopt, Test, ManifoldsBase, Dates, Manifolds
using Manopt: DebugActionFactory, DebugFactory, DebugGroupFactory

struct TestPolarManifold <: AbstractManifold{ℝ} end

function ManifoldsBase.default_inverse_retraction_method(::TestPolarManifold)
    return PolarInverseRetraction()
end

struct TestDebugAction <: DebugAction end

struct TestMessageState <: AbstractManoptSolverState end
Manopt.get_message(::TestMessageState) = "DebugTest"

mutable struct TestDebugParameterState <: AbstractManoptSolverState
    value::Int
end
function Manopt.set_parameter!(d::TestDebugParameterState, ::Val{:value}, v)
    (d.value = v; return d)
end
Manopt.get_parameter(d::TestDebugParameterState, ::Val{:value}) = d.value

@testset "Debug State" begin
    # helper to get debug as string
    @testset "Basic Debug Output" begin
        io = IOBuffer()
        M = ManifoldsBase.DefaultManifold(2)
        p = [4.0, 2.0]
        st = GradientDescentState(
            M;
            p = p,
            stopping_criterion = StopAfterIteration(10),
            stepsize = Manopt.ConstantStepsize(M),
        )
        f(M, q) = distance(M, q, p) .^ 2
        grad_f(M, q) = -2 * log(M, q, p)
        # summary fallback to show
        @test Manopt.status_summary(TestDebugAction()) === "TestDebugAction()"
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        a1 = DebugDivider("|"; io = io)
        @test Manopt.dispatch_state_decorator(DebugSolverState(st, a1)) === Val{true}()
        # constructors
        @test DebugSolverState(st, a1).debugDictionary[:Iteration] == a1
        @test DebugSolverState(st, [a1]).debugDictionary[:Iteration].group[1] == a1
        @test DebugSolverState(st, Dict(:A => a1)).debugDictionary[:A] == a1
        @test DebugSolverState(st, ["|"]).debugDictionary[:Iteration].divider == a1.divider
        @test endswith(repr(DebugSolverState(st, a1)), "\"|\"")
        @test repr(DebugSolverState(st, Dict{Symbol, DebugAction}())) == repr(st)
        # Passthrough
        dss = DebugSolverState(st, a1)
        Manopt.set_parameter!(dss, :StoppingCriterion, :MaxIteration, 20)
        @test dss.state.stop.max_iterations == 20 #Maybe turn into a getter?
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
        @test DebugCost(; format = "A %f").format == "A %f"
        DebugCost(; long = false, io = io)(mp, st, 0)
        @test String(take!(io)) == "f(x): 0.000000"
        DebugCost(; long = false, io = io)(mp, st, -1)
        @test String(take!(io)) == ""
        # entry
        DebugEntry(:p; prefix = "x:", io = io)(mp, st, 0)
        @test String(take!(io)) == "x: $p"
        DebugEntry(:p; prefix = "x:", io = io)(mp, st, -1)
        @test String(take!(io)) == ""
        # Change of Iterate and recording a custom field
        a2 = DebugChange(;
            storage = StoreStateAction(M; store_points = Tuple{:Iterate}, p_init = p),
            prefix = "Last: ",
            io = io,
        )
        a2(mp, st, 0) # init
        st.p = [3.0, 2.0]
        a2(mp, st, 1)
        a2inv = DebugChange(;
            storage = StoreStateAction(M; store_fields = [:Iterate]),
            prefix = "Last: ",
            io = io,
            inverse_retraction_method = PolarInverseRetraction(),
        )
        a2mani = DebugChange(
            TestPolarManifold();
            storage = StoreStateAction([:Iterate]),
            prefix = "Last: ",
            io = io,
        )
        @test a2inv.inverse_retraction_method === PolarInverseRetraction()
        @test a2mani.inverse_retraction_method === PolarInverseRetraction()
        @test a2.inverse_retraction_method === LogarithmicInverseRetraction()
        @test String(take!(io)) == "Last: 1.000000"
        a3 = DebugGradientChange(;
            storage = StoreStateAction([:Gradient, :Iterate]), prefix = "Last: ", io = io
        )
        a3(mp, st, 0) # init
        st.X = [1.0, 0.0]
        a3(mp, st, 1)
        @test String(take!(io)) == "Last: 1.000000"
        # Iterate
        DebugIterate(; io = io)(mp, st, 0)
        @test String(take!(io)) == ""
        DebugIterate(; io = io)(mp, st, 1)
        @test String(take!(io)) == "p: $(st.p)"
        # Iteration
        DebugIteration(; io = io)(mp, st, 0)
        @test String(take!(io)) == "Initial "
        DebugIteration(; io = io)(mp, st, 23)
        @test String(take!(io)) == "# 23    "
        @test repr(DebugIteration()) == "DebugIteration(; format=\"# %-6d\")"
        @test Manopt.status_summary(DebugIteration()) == "(:Iteration, \"# %-6d\")"
        # `DebugEntryChange`
        dec = DebugEntryChange(:p, x -> x)
        @test startswith(repr(dec), "DebugEntryChange(:p")
        # DEbugEntryChange - reset
        st.p = p
        a3 = DebugEntryChange(
            :p,
            (mp, o, x, y) -> distance(Manopt.get_manifold(mp), x, y);
            prefix = "Last: ",
            io,
        )
        a4 = DebugEntryChange(
            :p,
            (mp, o, x, y) -> distance(Manopt.get_manifold(mp), x, y);
            initial_value = p,
            format = "Last: %1.1f",
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
        DebugStoppingCriterion(; io = io)(mp, st, 1)
        @test String(take!(io)) == ""
        st.stop(mp, st, 19)
        DebugStoppingCriterion(; io = io)(mp, st, 19)
        @test String(take!(io)) == ""
        st.stop(mp, st, 20)
        DebugStoppingCriterion(; io = io)(mp, st, 20)
        @test String(take!(io)) ==
            "At iteration 20 the algorithm reached its maximal number of iterations (20).\n"
        @test repr(DebugStoppingCriterion()) == "DebugStoppingCriterion()"
        @test Manopt.status_summary(DebugStoppingCriterion()) == ":Stop"
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
    @testset "Debug Factory" begin
        # Factory
        df = DebugFactory([:Stop, "|"])
        @test isa(df[:Stop], DebugStoppingCriterion)
        @test isa(df[:Iteration], DebugDivider)
        df = DebugFactory([:Stop, "|", 20])
        @test isa(df[:Iteration], DebugEvery)
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
                DebugFactory(s)[:Iteration].group,
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
                DebugFactory([(t, "A") for t in s])[:Iteration].group,
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
        a1 = DebugDivider("|")
        @test DebugActionFactory(a1) == a1
        @test DebugGroupFactory(a1) == a1 #when trying to build a one-element group, this is still just a1
        @test DebugFactory([(:Iterate, "A")])[:Iteration].format == "A"
        @test DebugActionFactory((:Iterate, "A")).format == "A"
        # Merge iteration and simple entries to Iteration
        df2 = DebugFactory([:Iteration, :Iteration => [:Cost]])
        @test length(df2[:Iteration].group) == 2
        # appended in the end
        @test df2[:Iteration].group[1] isa DebugCost
        @test df2[:Iteration].group[2] isa DebugIteration
        df3 = DebugFactory([:Stop, :Stop => [:Iteration]])
        @test length(df3[:Stop].group) == 2
        # appended in the end
        @test df3[:Stop].group[1] isa DebugIteration
        @test df3[:Stop].group[2] isa DebugStoppingCriterion
        # Group with every
        dgf1 = Manopt.DebugGroupFactory([" ", :Cost, 20])
        @test dgf1 isa DebugEvery
        @test dgf1.debug isa DebugGroup
    end
    @testset "Debug and parameter passthrough" begin
        s = TestDebugParameterState(0)
        d = DebugSolverState(s, DebugDivider(" | "))
        Manopt.set_parameter!(d, :value, 1)
        @test Manopt.get_parameter(d, :value) == 1
    end
    @testset "Debug Warnings" begin
        M = ManifoldsBase.DefaultManifold(2)
        p = [4.0, 2.0]
        st = GradientDescentState(
            M;
            p = p,
            stopping_criterion = StopAfterIteration(20),
            stepsize = Manopt.ConstantStepsize(M),
        )
        f(M, y) = Inf
        grad_f(M, y) = Inf .* ones(2)
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))

        w1 = DebugWarnIfCostNotFinite()
        @test repr(w1) == "DebugWarnIfCostNotFinite()"
        @test Manopt.status_summary(w1) == ":WarnCost"
        @test_logs (:warn,) (:warn,) w1(mp, st, 0)
        w2 = DebugWarnIfCostNotFinite(:Always)
        @test_logs (:warn,) w2(mp, st, 0)

        st.X = grad_f(M, p)
        w3 = DebugWarnIfFieldNotFinite(:X)
        @test repr(w3) == "DebugWarnIfFieldNotFinite(:X, :Once)"
        @test_logs (:warn,) (:warn,) w3(mp, st, 0)
        w4 = DebugWarnIfFieldNotFinite(:X, :Always)
        @test_logs (:warn,) w4(mp, st, 1)
        w5 = DebugWarnIfFieldNotFinite(:Gradient, :Always)
        @test_logs (:warn,) w5(mp, st, 1)

        M2 = Sphere(2)
        mp2 = DefaultManoptProblem(M2, ManifoldGradientObjective(f, grad_f))
        w6 = DebugWarnIfGradientNormTooLarge(1.0, :Once)
        @test repr(w6) == "DebugWarnIfGradientNormTooLarge(1.0, :Once)"
        st.X .= [4.0, 0.0] # > π in norm
        @test_logs (:warn,) (:warn,) w6(mp2, st, 1)

        st.p = Inf .* ones(2)
        w7 = DebugWarnIfFieldNotFinite(:Iterate, :Always)
        @test_logs (:warn,) w7(mp, st, 1)

        w8 = DebugWarnIfStepsizeCollapsed(1.0, :Once)
        @test repr(w8) == "DebugWarnIfStepsizeCollapsed(1.0, :Once)"
        @test_logs (:warn,) (:warn,) w8(mp2, st, 1)

        df1 = DebugFactory([:WarnCost])
        @test isa(df1[:Iteration], DebugWarnIfCostNotFinite)
        df2 = DebugFactory([:WarnGradient])
        @test isa(df2[:Iteration], DebugWarnIfFieldNotFinite)
        df3 = DebugFactory([:WarnBundle])
        @test isa(df3[:Iteration], DebugWarnIfLagrangeMultiplierIncreases)
        df4 = DebugFactory([:WarnStepsize])
        @test isa(df4[:Iteration], DebugWarnIfStepsizeCollapsed)
    end
    @testset "Debug Time" begin
        io = IOBuffer()
        M = ManifoldsBase.DefaultManifold(2)
        p = [4.0, 2.0]
        st = GradientDescentState(
            M;
            p = p,
            stopping_criterion = StopAfterIteration(20),
            stepsize = Manopt.ConstantStepsize(M),
        )
        f(M, q) = distance(M, q, p) .^ 2
        grad_f(M, q) = -2 * log(M, q, p)
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        d1 = DebugTime(; start = true, io = io)
        @test d1.last_time != Nanosecond(0)
        d2 = DebugTime(; io = io)
        @test d2.last_time == Nanosecond(0)
        d2(mp, st, 1)
        @test d2.last_time != Nanosecond(0) # changes on first call
        t = d2.last_time
        sleep(0.002)
        d2(mp, st, 2)
        @test t == d2.last_time # but not afterwards
        @test endswith(String(take!(io)), "seconds")
        d3 = DebugTime(; start = true, mode = :iterative, io = io)
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
        @test Manopt.status_summary(DebugTime(; mode = :iterative)) == drs2
        drs3 = "(:Time, \"time spent: %s\")"
        @test Manopt.status_summary(DebugTime(; mode = :cumulative)) == drs3
    end
    @testset "Debug show/summaries" begin
        d1 = DebugDivider("|")
        d2 = DebugIterate()
        d3 = DebugGroup([d1, d2])
        @test repr(d3) == "DebugGroup([$(d1), $(d2)])"
        ts = "[ $(Manopt.status_summary(d1)), $(Manopt.status_summary(d2)) ]"
        @test Manopt.status_summary(d3) == ts

        d4 = DebugEvery(d1, 4)
        @test repr(d4) == "DebugEvery($(d1), 4, true; activation_offset=1)"
        @test Manopt.status_summary(d4) === "[$(d1), 4]"

        ts2 = "DebugChange(; format=\"Last Change: %f\", inverse_retraction=LogarithmicInverseRetraction())"
        @test repr(DebugChange()) == ts2
        @test Manopt.status_summary(DebugChange()) == "(:Change, \"Last Change: %f\")"
        # verify that a non-default manifold works as well - not sure how to test this then
        d = DebugChange(Euclidean(2))

        @test repr(DebugCost()) == "DebugCost(; format=\"f(x): %f\", at_init=true)"
        @test Manopt.status_summary(DebugCost()) == "(:Cost, \"f(x): %f\")"

        @test repr(DebugDivider("|")) == "DebugDivider(; divider=\"|\", at_init=true)"
        @test Manopt.status_summary(DebugDivider("a")) == "\"a\""

        @test repr(DebugEntry(:a)) == "DebugEntry(:a; format=\"a: %s\", at_init=true)"

        @test repr(DebugStepsize()) == "DebugStepsize(; format=\"s:%s\", at_init=true)"
        @test Manopt.status_summary(DebugStepsize()) == "(:Stepsize, \"s:%s\")"

        @test repr(DebugGradientNorm()) == "DebugGradientNorm(; format=\"|grad f(p)|:%s\", at_init=true)"
        dgn_s = "(:GradientNorm, \"|grad f(p)|:%s\")"
        @test Manopt.status_summary(DebugGradientNorm()) == dgn_s

        @test repr(DebugGradient()) == "DebugGradient(; format=\"grad f(p):%s\", at_init=false)"
        dg_s = "(:Gradient, \"grad f(p):%s\")"
        @test Manopt.status_summary(DebugGradient()) == dg_s
    end
    @testset "Debug Messages" begin
        s = TestMessageState()
        mp = DefaultManoptProblem(Euclidean(2), ManifoldCostObjective(x -> x))
        d = DebugMessages(:Info, :Always)
        @test repr(d) == "DebugMessages(:Info, :Always)"
        @test Manopt.status_summary(d) == "(:Messages, :Always)"
        @test_logs (:info, "DebugTest") d(mp, s, 0)
    end
    @testset "DebugIfEntry" begin
        io = IOBuffer()
        M = ManifoldsBase.DefaultManifold(2)
        p = [-4.0, 2.0]
        st = GradientDescentState(
            M;
            p = p,
            stopping_criterion = StopAfterIteration(20),
            stepsize = Manopt.ConstantStepsize(M),
        )
        f(M, y) = Inf
        grad_f(M, y) = Inf .* ones(2)
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))

        die1 = DebugIfEntry(:p, p -> p[1] > 0.0; type = :warn, message = "test1")
        @test startswith(repr(die1), "DebugIfEntry(:p, ")
        @test_logs (:warn, "test1") die1(mp, st, 1)
        die2 = DebugIfEntry(:p, p -> p[1] > 0.0; type = :info, message = "test2")
        @test_logs (:info, "test2") die2(mp, st, 1)
        die3 = DebugIfEntry(:p, p -> p[1] > 0.0; type = :error, message = "test3")
        @test_throws ErrorException die3(mp, st, 1)
        die4 = DebugIfEntry(:p, p -> p[1] > 0.0; type = :print, message = "test4", io = io)
        die4(mp, st, 1)
        @test String(take!(io)) == "test4"
    end
    @testset "DebugWhenActive" begin
        io = IOBuffer()
        M = ManifoldsBase.DefaultManifold(2)
        p = [4.0, 2.0]
        st = GradientDescentState(
            M;
            p = p,
            stopping_criterion = StopAfterIteration(20),
            stepsize = Manopt.ConstantStepsize(M),
        )
        f(M, q) = distance(M, q, p) .^ 2
        grad_f(M, q) = -2 * log(M, q, p)
        mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        dD = DebugDivider(" | "; io = io)
        dA = DebugWhenActive(dD, false)
        @test !dA.active
        Manopt.set_parameter!(dA, :Dummy, true) # pass down
        Manopt.set_parameter!(dA, :Activity, true) # activate
        @test dA.active
        @test repr(dA) == "DebugWhenActive($(repr(dD)), true, true)"
        @test Manopt.status_summary(dA) == repr(dA)
        #issue active
        dA(mp, st, 1)
        @test endswith(String(take!(io)), " | ")
        dE = DebugEvery(dA, 2)
        dE(mp, st, 2)
        @test endswith(String(take!(io)), " | ")
        Manopt.set_parameter!(dE, :Activity, false) # deactivate
        dE(mp, st, -1) # test that reset is still working
        dE(mp, st, 2)
        @test endswith(String(take!(io)), "")
        @test !dA.active
        dG = DebugGroup([dA])
        Manopt.set_parameter!(dG, :Activity, true) # activate in group
        dG(mp, st, 2)
        @test endswith(String(take!(io)), " | ")
        # test its usage in the factory independent of position
        @test DebugFactory([" | ", :WhenActive])[:Iteration] isa DebugWhenActive
        @test DebugFactory([:WhenActive, " | "])[:Iteration] isa DebugWhenActive

        dst = DebugSolverState(st, dA)
        Manopt.set_parameter!(dst, :Debug, :Activity, true)
        @test dA.active
    end
    @testset "decorate_state! and callbacks" begin
        # Wrap this in a function so the callback uses right scope for n
        function test_simple_callback()
            M = ManifoldsBase.DefaultManifold(2)
            p = [4.0, 2.0]
            st = GradientDescentState(
                M;
                p = p,
                stopping_criterion = StopAfterIteration(20),
                stepsize = Manopt.ConstantStepsize(M),
            )
            f(M, q) = distance(M, q, p) .^ 2
            grad_f(M, q) = -2 * log(M, q, p)
            mp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
            n = 0
            cb() = (n += 1)
            dst = decorate_state!(st; callback = cb)
            step_solver!(mp, dst, 1)
            @test n == 1
            dst = decorate_state!(st; callback = cb, debug = DebugDivider(""))
            step_solver!(mp, dst, 1)
            @test n == 2
            cb2(p, s, k) = ((k > 1) && (n += 1))
            # Advanced 2, pass to debug
            dst2 = decorate_state!(st; debug = cb2)
            step_solver!(mp, dst2, 1)
            step_solver!(mp, dst2, 2)
            @test n == 3
            #Equivalent to 2.
            dst3 = decorate_state!(st; debug = Manopt.DebugCallback(cb2))
            step_solver!(mp, dst3, 1)
            step_solver!(mp, dst3, 2)
            @test n == 4
            return nothing
        end
        test_simple_callback()
        dbc = Manopt.DebugCallback(() -> nothing; simple = true)
        @test startswith(repr(dbc), "DebugCallback containing")
        @test startswith(Manopt.status_summary(dbc), "#")
    end
end
