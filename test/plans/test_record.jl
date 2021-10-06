using Manifolds, Manopt, Test, ManifoldsBase

@testset "Record Options" begin
    # helper to get debug as string
    io = IOBuffer()
    M = ManifoldsBase.DefaultManifold(2)
    x = [4.0, 2.0]
    o = GradientDescentOptions(
        x; stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(1.0)
    )
    f(M, y) = distance(M, y, x) .^ 2
    gradf(M, y) = -2 * log(M, y, x)
    p = GradientProblem(M, f, gradf)
    a = RecordIteration()
    # constructors
    rO = RecordOptions(o, a)
    @test Manopt.dispatch_options_decorator(rO) === Val{true}()
    @test get_options(o) == o
    @test get_options(rO) == o
    @test_throws MethodError get_options(p)
    #
    @test get_initial_stepsize(p, rO) == 1.0
    @test get_stepsize(p, rO, 1) == 1.0
    @test get_last_stepsize(p, rO, 1) == 1.0
    #
    @test rO.recordDictionary[:Iteration] == a
    @test RecordOptions(o, [a]).recordDictionary[:Iteration].group[1] == a
    @test RecordOptions(o, a).recordDictionary[:Iteration] == a
    @test RecordOptions(o, Dict(:A => a)).recordDictionary[:A] == a
    @test isa(RecordOptions(o, :Iteration).recordDictionary[:Iteration], RecordIteration)
    @test isa(RecordOptions(o, [:Iteration]).recordDictionary[:Iteration], RecordGroup)
    @test isa(
        RecordOptions(o, [:Iteration]).recordDictionary[:Iteration].group[1],
        RecordIteration,
    )
    @test isa(
        RecordOptions(o, [:It => RecordIteration()]).recordDictionary[:Iteration].group[1],
        RecordIteration,
    )
    @test isa(RecordFactory(o, :Iteration), RecordIteration)
    sa = :It3 => RecordIteration()
    @test RecordActionFactory(o, sa) === sa
    @test !has_record(o)
    @test_throws ErrorException get_record(o)
    @test get_options(o) == o
    @test !has_record(DebugOptions(o, []))
    @test has_record(rO)
    @test_throws ErrorException get_record(o)
    @test length(get_record(rO, :Iteration)) == 0
    @test length(rO[:Iteration]) == 0
    @test length(get_record(rO)) == 0
    @test length(get_record(DebugOptions(rO, []))) == 0
    @test length(get_record(RecordOptions(o, [:Iteration]), :Iteration, 1)) == 0
    @test length(get_record(RecordIteration(), 1)) == 0
    @test_throws ErrorException get_record(RecordOptions(o, Dict{Symbol,RecordAction}()))
    @test_throws ErrorException get_record(o)
    @test get_record(rO) == Array{Int64,1}()
    # RecordIteration
    @test a(p, o, 0) == nothing # inactive
    @test a(p, o, 1) == [1]
    @test a(p, o, 2) == [1, 2]
    @test a(p, o, 9) == [1, 2, 9]
    @test a(p, o, -1) == []
    # RecordGroup
    @test length(RecordGroup().group) == 0
    @test_throws ErrorException RecordGroup(RecordAction[], Dict(:a => 1))
    @test_throws ErrorException RecordGroup(RecordAction[], Dict(:a => 0))
    b = RecordGroup([RecordIteration(), RecordIteration()], Dict(:It1 => 1, :It2 => 2))
    b(p, o, 1)
    b(p, o, 2)
    @test b.group[1].recorded_values == [1, 2]
    @test b.group[2].recorded_values == [1, 2]
    @test get_record(b) == [(1, 1), (2, 2)]
    @test get_record(b, 1) == [1, 2]
    @test b[1] == [1, 2]
    @test get_record(b, :It1) == [1, 2]
    @test b[:It1] == [1, 2]
    @test get_record(b, (:It1, :It2)) == [(1, 1), (2, 2)]
    @test b[(:It1, :It2)] == [(1, 1), (2, 2)]
    @test RecordOptions(o, b)[:Iteration, 1] == [1, 2]
    #RecordEvery
    c = RecordEvery(a, 10, true)
    @test c(p, o, 0) === nothing
    @test c(p, o, 1) === nothing
    @test c(p, o, 10) == [10]
    @test c(p, o, 20) == [10, 20]
    @test c(p, o, -1) == []
    @test get_record(c) == []
    c2 = RecordEvery(
        RecordGroup([RecordIteration(), RecordIteration()], Dict(:It1 => 1, :It2 => 2)), 10
    )
    c2(p, o, 5)
    c2(p, o, 10)
    c2(p, o, 20)
    @test c2[1] == [10, 20]
    @test c2[:It1] == [10, 20]
    # RecordChange
    d = RecordChange()
    d(p, o, 1)
    @test d.recorded_values == [0.0] # no x0 -> assume x0 is the first iterate
    o.x = [3.0, 2.0]
    d(p, o, 2)
    @test d.recorded_values == [0.0, 1.0] # no x0 -> assume x0 is the first iterate
    e = RecordChange([4.0, 2.0])
    e(p, o, 1)
    @test e.recorded_values == [1.0] # no x0 -> assume x0 is the first iterate
    # RecordEntry
    o.x = x
    f = RecordEntry(x, :x)
    f(p, o, 1)
    @test f.recorded_values == [x]
    f2 = RecordEntry(typeof(x), :x)
    f2(p, o, 1)
    @test f2.recorded_values == [x]
    # RecordEntryChange
    o.x = x
    e = RecordEntryChange(:x, (p, o, x, y) -> distance(p.M, x, y))
    @test update_storage!(e.storage, o) == (:x,)
    e2 = RecordEntryChange(x, :x, (p, o, x, y) -> distance(p.M, x, y))
    @test e.field == e2.field
    e(p, o, 1)
    @test e.recorded_values == [0.0]
    o.x = [3.0, 2.0]
    e(p, o, 2)
    @test e.recorded_values == [0.0, 1.0]
    # RecordIterate
    o.x = x
    f = RecordIterate(x)
    @test_throws ErrorException RecordIterate()
    f(p, o, 1)
    @test f.recorded_values == [x]
    # RecordCost
    g = RecordCost()
    g(p, o, 1)
    @test g.recorded_values == [0.0]
    o.x = [3.0, 2.0]
    g(p, o, 2)
    @test g.recorded_values == [0.0, 1.0]
    #RecordFactory
    o.gradient = [0.0, 0.0]
    rf = RecordFactory(o, [:Cost, :gradient])
    @test isa(rf[:Iteration], RecordGroup)
    @test isa(rf[:Iteration].group[1], RecordCost)
    @test isa(rf[:Iteration].group[2], RecordEntry)
    @test isa(RecordFactory(o, [2])[:Iteration], RecordEvery)
    @test rf[:Iteration].group[2].field == :gradient
    @test length(rf[:Iteration].group) == 2
    @test all(
        isa.(
            RecordFactory(o, [:Cost, :Iteration, :Change, :Iterate])[:Iteration].group,
            [RecordCost, RecordIteration, RecordChange, RecordIterate],
        ),
    )
    @test RecordActionFactory(o, g) == g
end
