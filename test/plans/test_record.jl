using Manifolds, Manopt, Test, ManifoldsBase, Dates
using Manopt: RecordFactory, RecordGroupFactory, RecordActionFactory
mutable struct TestRecordParameterState <: AbstractManoptSolverState
    value::Int
end
function Manopt.set_parameter!(d::TestRecordParameterState, ::Val{:value}, v)
    (d.value = v; return d)
end
Manopt.get_parameter(d::TestRecordParameterState, ::Val{:value}) = d.value

@testset "Record State" begin
    # helper to get debug as string
    io = IOBuffer()
    M = ManifoldsBase.DefaultManifold(2)
    p = [4.0, 2.0]
    gds = GradientDescentState(
        M;
        p = copy(p),
        stopping_criterion = StopAfterIteration(10),
        stepsize = Manopt.ConstantStepsize(M),
    )
    f(M, q) = distance(M, q, p) .^ 2
    grad_f(M, q) = -2 * log(M, q, p)
    dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
    a = RecordIteration()
    @test repr(a) == "RecordIteration()"
    @test Manopt.status_summary(a) == ":Iteration"
    # constructors
    rs = RecordSolverState(gds, a)
    Manopt.set_parameter!(rs, :Record, RecordCost())
    @test Manopt.dispatch_state_decorator(rs) === Val{true}()
    @test get_state(gds) == gds
    @test get_state(rs) == gds
    @test_throws MethodError get_state(dmp)
    Manopt.set_parameter!(rs, :StoppingCriterion, :MaxIteration, 20)
    @test rs.state.stop.max_iterations == 20 #Maybe turn into a getter?
    #
    @test get_initial_stepsize(dmp, rs) == 1.0
    @test get_stepsize(dmp, rs, 1) == 1.0
    @test get_last_stepsize(dmp, rs, 1) == 1.0
    #
    @test rs.recordDictionary[:Iteration] == a
    @test RecordSolverState(gds, [a]).recordDictionary[:Iteration] == a
    @test RecordSolverState(gds, a).recordDictionary[:Iteration] == a
    @test RecordSolverState(gds, Dict(:A => a)).recordDictionary[:A] == a
    @test isa(
        RecordSolverState(gds, :Iteration).recordDictionary[:Iteration], RecordIteration
    )
    @test isa(
        RecordSolverState(gds, [:Iteration]).recordDictionary[:Iteration], RecordIteration
    )
    @test isa(
        RecordSolverState(gds, [:It => RecordIteration()]).recordDictionary[:It],
        RecordIteration,
    )
    @test isa(RecordFactory(gds, :Iteration)[:Iteration], RecordIteration)
    sa = RecordIteration() => :It3
    @test RecordActionFactory(gds, sa) === sa
    @test !has_record(gds)
    @test_throws ErrorException get_record(gds)
    @test get_state(gds) == gds
    @test !has_record(DebugSolverState(gds, []))
    @test has_record(rs)
    @test_throws ErrorException get_record(gds)
    @test length(get_record(rs, :Iteration)) == 0
    @test length(rs[:Iteration]) == 0
    @test length(get_record(rs)) == 0
    @test length(get_record(DebugSolverState(rs, []))) == 0
    @test length(get_record(RecordSolverState(gds, [:Iteration]), :Iteration, 1)) == 0
    @test length(get_record(RecordIteration(), 1)) == 0
    @test_throws ErrorException get_record(
        RecordSolverState(gds, Dict{Symbol, RecordAction}())
    )
    @test_throws ErrorException get_record(gds)
    @test get_record(rs) == Array{Int64, 1}()
    # RecordIteration
    @test a(dmp, gds, 0) == nothing # inactive
    @test a(dmp, gds, 1) == [1]
    @test a(dmp, gds, 2) == [1, 2]
    @test a(dmp, gds, 9) == [1, 2, 9]
    @test a(dmp, gds, -1) == []
    # RecordGroup
    @test length(RecordGroup().group) == 0
    @test_throws ErrorException RecordGroup([:a]) #no valid action
    @test_throws ErrorException RecordGroup(RecordAction[], Dict(:a => 1))
    @test_throws ErrorException RecordGroup(RecordAction[], Dict(:a => 0))
    b = RecordGroup([RecordIteration(), RecordIteration()], Dict(:It1 => 1, :It2 => 2))
    @test Manopt.status_summary(b) == "[ :Iteration, :Iteration ]"
    @test repr(b) == "RecordGroup([RecordIteration(), RecordIteration()])"
    b(dmp, gds, 1)
    b(dmp, gds, 2)
    @test b.group[1].recorded_values == [1, 2]
    @test b.group[2].recorded_values == [1, 2]
    @test get_record(b) == [(1, 1), (2, 2)]
    @test get_record(b, 1) == [1, 2]
    @test b[1] == [1, 2]
    @test get_record(b, :It1) == [1, 2]
    @test b[:It1] == [1, 2]
    @test get_record(b, (:It1, :It2)) == [(1, 1), (2, 2)]
    @test b[(:It1, :It2)] == [(1, 1), (2, 2)]
    @test RecordSolverState(gds, b)[:Iteration, 1] == [1, 2]
    @testset "RecordEvery" begin
        c = RecordEvery(a, 10, true)
        @test repr(c) == "RecordEvery(RecordIteration(), 10, true)"
        @test Manopt.status_summary(c) == "[RecordIteration(), 10]"
        c(dmp, gds, 0)
        @test length(get_record(c)) === 0
        c(dmp, gds, 1)
        @test length(get_record(c)) === 0
        c(dmp, gds, 10)
        @test get_record(c) == [10]
        c(dmp, gds, 20)
        @test get_record(c) == [10, 20]
        c(dmp, gds, -1)
        @test get_record(c) == []
        c2 = RecordEvery(
            RecordGroup([RecordIteration(), RecordIteration()], Dict(:It1 => 1, :It2 => 2)),
            10,
        )
        @test repr(c2) == "RecordEvery($(repr(c2.record)), 10, true)"
        @test Manopt.status_summary(c2) == "[:Iteration, :Iteration, 10]"
        c2(dmp, gds, 5)
        c2(dmp, gds, 10)
        c2(dmp, gds, 20)
        @test c2[1] == [10, 20]
        @test c2[:It1] == [10, 20]
    end
    @testset "RecordChange" begin
        d = RecordChange()
        sd = "RecordChange(; inverse_retraction_method=LogarithmicInverseRetraction())"
        @test repr(d) == sd
        @test Manopt.status_summary(d) == ":Change"
        d(dmp, gds, 1)
        @test d.recorded_values == [0.0] # no p0 -> assume p is the first iterate
        set_iterate!(gds, M, p + [1.0, 0.0])
        d(dmp, gds, 2)
        @test d.recorded_values == [0.0, 1.0] # no p0 -> assume p is the first iterate
        e = RecordChange([4.0, 2.0])
        e(dmp, gds, 1)
        @test e.recorded_values == [1.0] # no p0 -> assume p is the first iterate

        dinvretr = RecordChange(; inverse_retraction_method = PolarInverseRetraction())
        dmani = RecordChange(SymplecticMatrices(2))
        @test dinvretr.inverse_retraction_method === PolarInverseRetraction()
        @test dmani.inverse_retraction_method === CayleyInverseRetraction()
        @test d.inverse_retraction_method === LogarithmicInverseRetraction()
    end
    @testset "RecordEntry" begin
        set_iterate!(gds, M, p)
        f = RecordEntry(p, :p)
        @test repr(f) == "RecordEntry(:p)"
        f(dmp, gds, 1)
        @test f.recorded_values == [p]
        f2 = RecordEntry(typeof(p), :p)
        f2(dmp, gds, 1)
        @test f2.recorded_values == [p]
    end
    @testset "RecordEntryChange" begin
        set_iterate!(gds, M, p)
        e = RecordEntryChange(:p, (p, o, x, y) -> distance(get_manifold(p), x, y))
        @test startswith(repr(e), "RecordEntryChange(:p")
        @test update_storage!(e.storage, dmp, gds) == [:p]
        e(dmp, gds, 1)
        @test e.recorded_values == [0.0]
        set_iterate!(gds, M, [3.0, 2.0])
        e(dmp, gds, 2)
        @test e.recorded_values == [0.0, 1.0]
    end
    @testset "RecordIterate" begin
        set_iterate!(gds, M, p)
        f = RecordIterate(p)
        @test Manopt.status_summary(f) == ":Iterate"
        @test repr(f) == "RecordIterate(Vector{Float64})"
        @test_throws ErrorException RecordIterate()
        f(dmp, gds, 1)
        @test f.recorded_values == [p]
    end
    @testset "RecordCost" begin
        g = RecordCost()
        @test repr(g) == "RecordCost()"
        @test Manopt.status_summary(g) == ":Cost"
        g(dmp, gds, 1)
        @test g.recorded_values == [0.0]
        gds.p = [3.0, 2.0]
        g(dmp, gds, 2)
        @test g.recorded_values == [0.0, 1.0]
    end
    @testset "RecordStoppingReason" begin
        g = RecordStoppingReason()
        @test repr(g) == "RecordStoppingReason()"
        @test Manopt.status_summary(g) == ":Stop"
        @test length(get_record(g)) == 0
        stop_solver!(dmp, gds, 21) # trigger stop
        g(dmp, gds, 21) # record
        @test length(get_record(g)) == 1
        gds.stop(dmp, gds, 0) # reset
    end
    @testset "RecordSubsolver" begin
        rss = RecordSubsolver()
        @test repr(rss) == "RecordSubsolver(; record=[:Iteration], record_type=Any)"
        @test Manopt.status_summary(rss) == ":Subsolver"
        epms = ExactPenaltyMethodState(M, dmp, rs)
        rss(dmp, epms, 1)
    end
    @testset "RecordWhenActive" begin
        i = RecordIteration()
        rwa = RecordWhenActive(i)
        @test repr(rwa) == "RecordWhenActive(RecordIteration(), true, true)"
        @test Manopt.status_summary(rwa) == repr(rwa)
        rwa(dmp, gds, 1)
        @test length(get_record(rwa)) == 1
        rwa(dmp, gds, -1) # Reset
        @test length(get_record(rwa)) == 0
        rwa(dmp, gds, 1)
        Manopt.set_parameter!(rwa, :Activity, false)
        # passthrough to inner
        Manopt.set_parameter!(rwa, :test, 1)
        @test !rwa.active
        # test inactive
        rwa(dmp, gds, 2)
        @test length(get_record(rwa)) == 1 # updated, but not cleared
        # test always update
        rwa(dmp, gds, -1)
        @test length(get_record(rwa)) == 0 # updated, but not cleared
    end
    @testset "Manopt.RecordFactory" begin
        gds.X = [0.0, 0.0]
        rf = RecordFactory(gds, [:Cost, :X])
        @test isa(rf[:Iteration], RecordGroup)
        @test isa(rf[:Iteration].group[1], RecordCost)
        @test isa(rf[:Iteration].group[2], RecordEntry)
        @test isa(RecordFactory(gds, [:Iteration, 2])[:Iteration], RecordEvery)
        @test rf[:Iteration].group[2].field == :X
        @test length(rf[:Iteration].group) == 2
        s = [:Cost, :Iteration, :Change, :Iterate, :Time, :IterativeTime]
        @test all(
            isa.(
                RecordFactory(gds, s)[:Iteration].group,
                [
                    RecordCost,
                    RecordIteration,
                    RecordChange,
                    RecordIterate,
                    RecordTime,
                    RecordTime,
                ],
            ),
        )
        rf2 = RecordFactory(gds, [:Iteration => [:Cost], :Iteration])
        # Check they are combined in iteration
        @test rf2[:Iteration] isa RecordGroup
        @test length(rf2[:Iteration].group) == 2
        rf3 = RecordFactory(gds, [:Stop => [:Cost], :Stop])
        # Check they are combined in iteration
        @test rf3[:Stop] isa RecordGroup
        @test length(rf3[:Stop].group) == 2
        @test length(RecordFactory(gds, [:Cost, :Stop])) == 2
    end
    @testset "Manopt.RecordActionFactory" begin
        g = RecordCost()
        @test RecordActionFactory(gds, g) == g
        rss = RecordActionFactory(gds, :Subsolver)
        @test rss isa RecordSubsolver
        @test rss.record == [:Iteration] # Default
        rss2 = RecordActionFactory(gds, (:Subsolver, :Stop))
        @test rss2 isa RecordSubsolver
        @test rss2.record == [:Stop]
        rss3 = RecordActionFactory(gds, (:X, :F)) # last gets ignored
        @test rss3 isa RecordEntry
    end
    @testset "Manopt.RecordGroupFactory" begin
        @test RecordGroupFactory(gds, [:Iteration, :Cost, :WhenActive]) isa RecordWhenActive
        @test RecordGroupFactory(gds, [:Iteration, :Cost, :WhenActive, 5]) isa
            RecordWhenActive
        @test RecordGroupFactory(gds, [:Iteration, :Cost, 5]) isa RecordEvery
        rg = RecordGroupFactory(gds, [:Cost, RecordCost() => :Cost2])
        @test (:Cost in keys(rg.indexSymbols)) && (:Cost2 in keys(rg.indexSymbols))
        @test (1 in values(rg.indexSymbols)) && (2 in values(rg.indexSymbols))
    end
    @testset "RecordGroup" begin
        @test length(RecordGroup([RecordCost(), RecordIteration() => :It]).group) == 2
    end
    @testset "RecordTime" begin
        h1 = RecordTime(; mode = :cumulative)
        @test repr(h1) == "RecordTime(; mode=:cumulative)"
        @test Manopt.status_summary(h1) == ":Time"
        t = h1.start
        @test t isa Nanosecond
        h1(dmp, gds, 1)
        @test h1.start == t
        h2 = RecordTime(; mode = :iterative)
        t = h2.start
        @test t isa Nanosecond
        sleep(0.002)
        h2(dmp, gds, 1)
        @test h2.start != t
        h3 = RecordTime(; mode = :total)
        h3(dmp, gds, 1)
        h3(dmp, gds, 10)
        h3(dmp, gds, 19)
        @test length(h3.recorded_values) == 0
        # stop after 20 so 21 hits
        h3(dmp, gds, 20)
        @test length(h3.recorded_values) == 1
        @test repr(RecordGradientNorm()) == "RecordGradientNorm()"
        # since only the type is stored can test
        @test repr(RecordGradient(zeros(3))) == "RecordGradient{Vector{Float64}}()"
    end
    @testset "Record and parameter passthrough" begin
        s = TestRecordParameterState(0)
        r = RecordSolverState(s, RecordIteration())
        Manopt.set_parameter!(r, :value, 1)
        @test Manopt.get_parameter(r, :value) == 1
    end
end
