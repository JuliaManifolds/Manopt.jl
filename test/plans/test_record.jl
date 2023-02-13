using Manifolds, Manopt, Test, ManifoldsBase, Dates

@testset "Record State" begin
    # helper to get debug as string
    io = IOBuffer()
    M = ManifoldsBase.DefaultManifold(2)
    p = [4.0, 2.0]
    gds = GradientDescentState(
        M, copy(p); stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(M)
    )
    f(M, q) = distance(M, q, p) .^ 2
    grad_f(M, q) = -2 * log(M, q, p)
    dmp = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
    a = RecordIteration()
    @test repr(a) == "RecordIteration()"
    @test Manopt.status_summary(a) == ":Iteration"
    # constructors
    rs = RecordSolverState(gds, a)
    @test Manopt.dispatch_state_decorator(rs) === Val{true}()
    @test get_state(gds) == gds
    @test get_state(rs) == gds
    @test_throws MethodError get_state(dmp)
    #
    @test get_initial_stepsize(dmp, rs) == 1.0
    @test get_stepsize(dmp, rs, 1) == 1.0
    @test get_last_stepsize(dmp, rs, 1) == 1.0
    #
    @test rs.recordDictionary[:Iteration] == a
    @test RecordSolverState(gds, [a]).recordDictionary[:Iteration].group[1] == a
    @test RecordSolverState(gds, a).recordDictionary[:Iteration] == a
    @test RecordSolverState(gds, Dict(:A => a)).recordDictionary[:A] == a
    @test isa(
        RecordSolverState(gds, :Iteration).recordDictionary[:Iteration], RecordIteration
    )
    @test isa(
        RecordSolverState(gds, [:Iteration]).recordDictionary[:Iteration], RecordGroup
    )
    @test isa(
        RecordSolverState(gds, [:Iteration]).recordDictionary[:Iteration].group[1],
        RecordIteration,
    )
    @test isa(
        RecordSolverState(gds, [:It => RecordIteration()]).recordDictionary[:Iteration].group[1],
        RecordIteration,
    )
    @test isa(RecordFactory(gds, :Iteration), RecordIteration)
    sa = :It3 => RecordIteration()
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
        RecordSolverState(gds, Dict{Symbol,RecordAction}())
    )
    @test_throws ErrorException get_record(gds)
    @test get_record(rs) == Array{Int64,1}()
    # RecordIteration
    @test a(dmp, gds, 0) == nothing # inactive
    @test a(dmp, gds, 1) == [1]
    @test a(dmp, gds, 2) == [1, 2]
    @test a(dmp, gds, 9) == [1, 2, 9]
    @test a(dmp, gds, -1) == []
    # RecordGroup
    @test length(RecordGroup().group) == 0
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
    #RecordEvery
    c = RecordEvery(a, 10, true)
    @test repr(c) == "RecordEvery(RecordIteration(), 10, true)"
    @test Manopt.status_summary(c) == "[RecordIteration(), 10]"
    @test c(dmp, gds, 0) === nothing
    @test c(dmp, gds, 1) === nothing
    @test c(dmp, gds, 10) == [10]
    @test c(dmp, gds, 20) == [10, 20]
    @test c(dmp, gds, -1) == []
    @test get_record(c) == []
    c2 = RecordEvery(
        RecordGroup([RecordIteration(), RecordIteration()], Dict(:It1 => 1, :It2 => 2)), 10
    )
    @test repr(c2) == "RecordEvery($(repr(c2.record)), 10, true)"
    @test Manopt.status_summary(c2) == "[:Iteration, :Iteration, 10]"
    c2(dmp, gds, 5)
    c2(dmp, gds, 10)
    c2(dmp, gds, 20)
    @test c2[1] == [10, 20]
    @test c2[:It1] == [10, 20]
    # RecordChange
    d = RecordChange()
    sd = "DebugChange(; inverse_retraction_method=LogarithmicInverseRetraction())"
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

    dinvretr = RecordChange(; inverse_retraction_method=PolarInverseRetraction())
    dmani = RecordChange(; manifold=Symplectic(2))
    @test dinvretr.inverse_retraction_method === PolarInverseRetraction()
    @test dmani.inverse_retraction_method === CayleyInverseRetraction()
    @test d.inverse_retraction_method === LogarithmicInverseRetraction()
    # RecordEntry
    set_iterate!(gds, M, p)
    f = RecordEntry(p, :p)
    @test repr(f) == "RecordEntry(:p)"
    f(dmp, gds, 1)
    @test f.recorded_values == [p]
    f2 = RecordEntry(typeof(p), :p)
    f2(dmp, gds, 1)
    @test f2.recorded_values == [p]
    # RecordEntryChange
    set_iterate!(gds, M, p)
    e = RecordEntryChange(:p, (p, o, x, y) -> distance(get_manifold(p), x, y))
    @test startswith(repr(e), "RecordEntryChange(:p")
    @test update_storage!(e.storage, gds) == (:p,)
    e2 = RecordEntryChange(dmp, :p, (p, o, x, y) -> distance(get_manifold(p), x, y))
    @test e.field == e2.field
    e(dmp, gds, 1)
    @test e.recorded_values == [0.0]
    set_iterate!(gds, M, [3.0, 2.0])
    e(dmp, gds, 2)
    @test e.recorded_values == [0.0, 1.0]
    # RecordIterate
    set_iterate!(gds, M, p)
    f = RecordIterate(p)
    @test Manopt.status_summary(f) == ":Iterate"
    @test repr(f) == "RecordIterate(Vector{Float64})"
    @test_throws ErrorException RecordIterate()
    f(dmp, gds, 1)
    @test f.recorded_values == [p]
    # RecordCost
    g = RecordCost()
    @test repr(g) == "RecordCost()"
    @test Manopt.status_summary(g) == ":Cost"
    g(dmp, gds, 1)
    @test g.recorded_values == [0.0]
    gds.p = [3.0, 2.0]
    g(dmp, gds, 2)
    @test g.recorded_values == [0.0, 1.0]
    #RecordFactory
    gds.X = [0.0, 0.0]
    rf = RecordFactory(gds, [:Cost, :X])
    @test isa(rf[:Iteration], RecordGroup)
    @test isa(rf[:Iteration].group[1], RecordCost)
    @test isa(rf[:Iteration].group[2], RecordEntry)
    @test isa(RecordFactory(gds, [2])[:Iteration], RecordEvery)
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
    @test RecordActionFactory(gds, g) == g

    h1 = RecordTime(; mode=:cumulative)
    @test repr(h1) == "RecordTime(; mode=:cumulative)"
    @test Manopt.status_summary(h1) == ":Time"
    t = h1.start
    @test t isa Nanosecond
    h1(dmp, gds, 1)
    @test h1.start == t
    h2 = RecordTime(; mode=:iterative)
    t = h2.start
    @test t isa Nanosecond
    sleep(0.002)
    h2(dmp, gds, 1)
    @test h2.start != t
    h3 = RecordTime(; mode=:total)
    h3(dmp, gds, 1)
    h3(dmp, gds, 10)
    h3(dmp, gds, 19)
    @test length(h3.recorded_values) == 0
    # stop after 20 so 21 hits
    h3(dmp, gds, 20)
    @test length(h3.recorded_values) == 1
end
