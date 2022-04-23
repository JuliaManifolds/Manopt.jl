@testset "Debug Options" begin
    # helper to get debug as string
    io = IOBuffer()
    M = ManifoldsBase.DefaultManifold(2)
    x = [4.0, 2.0]
    o = GradientDescentOptions(
        x; stopping_criterion=StopAfterIteration(20), stepsize=ConstantStepsize(M)
    )
    f(M, y) = distance(M, y, x) .^ 2
    gradf(M, y) = -2 * log(M, y, x)
    p = GradientProblem(M, f, gradf)
    a1 = DebugDivider("|", io)
    @test Manopt.dispatch_options_decorator(DebugOptions(o, a1)) === Val{true}()
    # constructors
    @test DebugOptions(o, a1).debugDictionary[:All] == a1
    @test DebugOptions(o, [a1]).debugDictionary[:All].group[1] == a1
    @test DebugOptions(o, Dict(:A => a1)).debugDictionary[:A] == a1
    @test DebugOptions(o, ["|"]).debugDictionary[:All].group[1].divider == a1.divider
    # single AbstractOptionsActions
    # DebugDivider
    a1(p, o, 0)
    s = @test String(take!(io)) == "|"
    DebugGroup([a1, a1])(p, o, 0)
    @test String(take!(io)) == "||"
    DebugEvery(a1, 10, false)(p, o, 9)
    @test String(take!(io)) == ""
    DebugEvery(a1, 10, true)(p, o, 10)
    @test String(take!(io)) == "|"
    @test DebugEvery(a1, 10, true)(p, o, -1) == nothing
    # Debug Cost
    @test DebugCost(; format="A %f").format == "A %f"
    DebugCost(; long=false, io=io)(p, o, 0)
    @test String(take!(io)) == "F(x): 0.000000"
    DebugCost(; long=false, io=io)(p, o, -1)
    @test String(take!(io)) == ""
    # entry
    DebugEntry(:x; prefix="x:", io=io)(p, o, 0)
    @test String(take!(io)) == "x: $x"
    DebugEntry(:x; prefix="x:", io=io)(p, o, -1)
    @test String(take!(io)) == ""
    # Change
    a2 = DebugChange(; storage=StoreOptionsAction((:x,)), prefix="Last: ", io=io)
    a2(p, o, 0) # init
    o.x = [3.0, 2.0]
    a2(p, o, 1)
    @test String(take!(io)) == "Last: 1.000000"
    # Iterate
    DebugIterate(; io=io)(p, o, 0)
    @test String(take!(io)) == ""
    DebugIterate(; io=io)(p, o, 1)
    @test String(take!(io)) == "x: $(o.x)"
    # Iteration
    DebugIteration(; io=io)(p, o, 0)
    @test String(take!(io)) == "Initial"
    DebugIteration(; io=io)(p, o, 23)
    @test String(take!(io)) == "# 23    "
    # DEbugEntryChange - reset
    o.x = x
    a3 = DebugEntryChange(:x, (p, o, x, y) -> distance(p.M, x, y); prefix="Last: ", io)
    a4 = DebugEntryChange(
        :x, (p, o, x, y) -> distance(p.M, x, y); initial_value=x, format="Last: %1.1f", io
    )
    a3(p, o, 0) # init
    @test String(take!(io)) == ""
    a4(p, o, 0) # init
    @test String(take!(io)) == ""
    #change
    o.x = [3.0, 2.0]
    a3(p, o, 1)
    @test String(take!(io)) == "Last: 1.0"
    a4(p, o, 1)
    @test String(take!(io)) == "Last: 1.0"
    # StoppingCriterion
    DebugStoppingCriterion(io)(p, o, 1)
    @test String(take!(io)) == ""
    o.stop(p, o, 19)
    DebugStoppingCriterion(io)(p, o, 19)
    @test String(take!(io)) == ""
    o.stop(p, o, 20)
    DebugStoppingCriterion(io)(p, o, 20)
    @test String(take!(io)) ==
        "The algorithm reached its maximal number of iterations (20).\n"
    df = DebugFactory([:Stop, "|"])
    @test isa(df[:Stop], DebugStoppingCriterion)
    @test isa(df[:All], DebugGroup)
    @test isa(df[:All].group[1], DebugDivider)
    @test length(df[:All].group) == 1
    df = DebugFactory([:Stop, "|", 20])
    @test isa(df[:All], DebugEvery)
    @test all(
        isa.(
            DebugFactory([:Change, :Iteration, :Iterate, :Cost, :x])[:All].group,
            [DebugChange, DebugIteration, DebugIterate, DebugCost, DebugEntry],
        ),
    )
    @test all(
        isa.(
            DebugFactory([
                (:Change, "A"), (:Iteration, "A"), (:Iterate, "A"), (:Cost, "A"), (:x, "A")
            ])[:All].group,
            [DebugChange, DebugIteration, DebugIterate, DebugCost, DebugEntry],
        ),
    )
    @test DebugActionFactory(a3) == a3
    @test DebugFactory([(:x, "A")])[:All].group[1].format == "A"
    @test DebugActionFactory((:x, "A")).format == "A"
end
