@testset "Debug Options" begin
    # helper to get debug as string
    io = IOBuffer()
    M = Euclidean(2)
    x = RnPoint([4.,2.])
    o = GradientDescentOptions(x,stopAfterIteration(20), ConstantStepsize(1.))
    f = y -> distance(M,y,x).^2
    ∇f = y -> -2*log(M,y,x)
    p = GradientProblem(M,f,∇f)
    a1 = DebugDivider("|",x -> print(io,x))
    # constructors
    @test DebugOptions(o,a1).debugDictionary[:All] == a1
    @test DebugOptions(o,[a1]).debugDictionary[:All].group[1] == a1
    @test DebugOptions(o,Dict(:A => a1)).debugDictionary[:A] == a1
    @test DebugOptions(o, ["|"]).debugDictionary[:All].group[1].divider == a1.divider
    # single Actions
    # DebugDivider
    a1(p,o,0); s = 
    @test String(take!(io)) == "|"
    #DebugGroup
    DebugGroup([a1,a1])(p,o,0)
    @test String(take!(io)) == "||"
    # Debug Every
    DebugEvery(a1,10,false)(p,o,9)
    @test String(take!(io)) == ""
    DebugEvery(a1,10,true)(p,o,10)
    @test String(take!(io)) == "|"
    @test DebugEvery(a1,10,true)(p,o,-1) == nothing
    # Debug Cost
    @test DebugCost("A").prefix == "A"
    DebugCost(false,x -> print(io,x))(p,o,0)
    @test String(take!(io)) == "F(x): 0.0"
    DebugCost(false,x -> print(io,x))(p,o,-1)
    @test String(take!(io)) == ""
    # entry
    DebugEntry(:x,"x:",x->print(io,x))(p,o,0)
    @test String(take!(io)) == "x: $x"
    DebugEntry(:x,"x:",x->print(io,x))(p,o,-1)
    @test String(take!(io)) == ""
    # Change
    a2 = DebugChange(StoreOptionsAction( (:x,) ), "Last: ", x -> print(io,x))
    a2(p,o,0) # init
    o.x = RnPoint([3.,2.])
    a2(p,o,1)
    @test String(take!(io)) == "Last: 1.0"
    # Iterate
    DebugIterate(x -> print(io,x))(p,o,0)
    @test String(take!(io)) == "x:$(o.x)"
    DebugIterate(x -> print(io,x))(p,o,1)
    @test String(take!(io)) == "x:$(o.x)"
    # Iteration
    DebugIteration(x -> print(io,x))(p,o,0)
    @test String(take!(io)) == "Initial"
    DebugIteration(x -> print(io,x))(p,o,23)
    @test String(take!(io)) == "# 23"
    # DEbugEntryChange - reset
    o.x = x
    a3 = DebugEntryChange(:x,(p,o,x,y)-> distance(p.M,x,y), StoreOptionsAction( (:x,) ), "Last: ", x -> print(io,x))
    a4 = DebugEntryChange(x,:x,(p,o,x,y)-> distance(p.M,x,y), StoreOptionsAction( (:x,) ), "Last: ", x -> print(io,x))
    a3(p,o,0) # init
    @test String(take!(io)) == ""
    a4(p,o,0) # init
    @test String(take!(io)) == ""
    #change 
    o.x = RnPoint([3.,2.])
    a3(p,o,1)
    @test String(take!(io)) == "Last: 1.0"
    a4(p,o,1)
    @test String(take!(io)) == "Last: 1.0"
    # StoppingCriterion
    DebugStoppingCriterion(x -> print(io,x))(p, o, 1)
    @test String(take!(io)) == ""
    o.stop(p,o,20)
    DebugStoppingCriterion(x -> print(io,x))(p, o, 20)
    @test String(take!(io)) == ""
    o.stop(p,o,21)
    DebugStoppingCriterion(x -> print(io,x))(p, o, 21)
    @test String(take!(io)) == "The algorithm reached its maximal number of iterations (20).\n"
    #DebugFactory
    df = DebugFactory([:Stop,"|"])
    @test isa( df[:Stop], DebugStoppingCriterion)
    @test isa(df[:All], DebugGroup)
    @test isa( df[:All].group[1], DebugDivider )
    @test length(df[:All].group) == 1
    df = DebugFactory([:Stop,"|",20])
    @test isa(df[:All], DebugEvery)
    @test all(  isa.( DebugFactory([:Change, :Iteration, :Iterate, :Cost, :x])[:All].group,
        [DebugChange, DebugIteration, DebugIterate, DebugCost, DebugEntry]) )
    @test DebugActionFactory(a3) == a3
end