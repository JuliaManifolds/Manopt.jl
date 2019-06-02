@testset "Record Options" begin
    # helper to get debug as string
    io = IOBuffer()
    M = Euclidean(2)
    x = RnPoint([4.,2.])
    o = GradientDescentOptions(x,stopAfterIteration(20), ConstantStepsize(1.))
    f = y -> distance(M,y,x).^2
    ∇f = y -> -2*log(M,y,x)
    p = GradientProblem(M,f,∇f)
    a = RecordIteration()
    # constructors
    rO = RecordOptions(o,a)
    @test getOptions(o) == o
    @test getOptions(rO) == o
    @test_throws ErrorException getOptions(p)
    #
    @test getInitialStepsize(p,rO) == 1.
    @test getStepsize!(p,rO,1) == 1.
    @test getLastStepsize(p,rO,1) == 1.
    #
    @test rO.recordDictionary[:All] == a
    @test RecordOptions(o,[a]).recordDictionary[:All].group[1] == a
    @test RecordOptions(o,Dict(:A => a)).recordDictionary[:A] == a
    @test isa( RecordOptions(o, [:Iteration]).recordDictionary[:All].group[1], RecordIteration)
    @test !hasRecord(o)
    @test !hasRecord(DebugOptions(o,[]))
    @test hasRecord(rO)
    @test_throws ErrorException getRecord(o)
    @test length( getRecord(rO,:All) ) == 0
    @test length( getRecord(rO) ) == 0 
    @test length(  getRecord(DebugOptions( rO,[]) )  ) == 0 
    @test_throws ErrorException getRecord(RecordOptions(o,Dict{Symbol,RecordAction}()))
    @test_throws ErrorException getRecord(o)
    @test getRecord(rO) == Array{Int64,1}()
    # RecordIteration
    @test a(p,o,0) == nothing # inactive
    @test a(p,o,1) == [1]
    @test a(p,o,2) == [1,2]
    @test a(p,o,9) == [1,2,9]
    @test a(p,o,-1) == []
    # RecordGroup
    @test length(RecordGroup().group) == 0
    b = RecordGroup([ RecordIteration(), RecordIteration() ])
    b(p,o,1)
    b(p,o,2)
    @test b.group[1].recordedValues == [1,2]
    @test b.group[2].recordedValues == [1,2]
    @test getRecord(b) == [ (1,1), (2,2) ]
    #RecordEvery
    c = RecordEvery(a,10,true)
    @test c(p,o,0) == nothing
    @test c(p,o,1) == nothing
    @test c(p,o,10) == [10]
    @test c(p,o,20) == [10,20]
    @test c(p,o,-1) == []
    @test getRecord(c) == []
    # RecordChange
    c = RecordChange()
    c(p,o,1)
    @test c.recordedValues == [0.] # no x0 -> assume x0 is the first iterate
    o.x = RnPoint([3.,2.])
    c(p,o,2)
    @test c.recordedValues == [0.,1.] # no x0 -> assume x0 is the first iterate
    c = RecordChange(RnPoint([4.,2.]))
    c(p,o,1)
    @test c.recordedValues == [1.] # no x0 -> assume x0 is the first iterate
    # RecordEntry
    o.x = x
    d = RecordEntry(x,:x)
    d2 = RecordEntry(RnPoint{Float64},:x)
    @test d.field == d2.field
    d(p,o,1)
    @test d.recordedValues == [x]
    # RecordEntryChange
    o.x = x
    e = RecordEntryChange(:x, (p,o,x,y) -> distance(p.M,x,y))
    @test updateStorage!(e.storage,o) == (:x,)
    e2 = RecordEntryChange(x,:x,(p,o,x,y) -> distance(p.M,x,y))
    @test e.field == e2.field
    e(p,o,1)
    @test e.recordedValues == [0.]
    o.x = RnPoint([3.0,2.0])
    e(p,o,2)
    @test e.recordedValues == [0.,1.]
    # RecordIterate
    o.x = x
    f = RecordIterate(x)
    f2 = RecordIterate(RnPoint{Float64})
    @test_throws ErrorException RecordIterate()
    @test f.recordedValues == f2.recordedValues
    f(p,o,1)
    @test f.recordedValues == [x]
    # RecordCost
    g = RecordCost()
    g(p,o,1)
    @test g.recordedValues == [0.]
    o.x = RnPoint([3.,2.])
    g(p,o,2)
    @test g.recordedValues == [0.,1.]    
    #RecordFactory
    o.∇ = RnTVector([0.,0.])
    rf = RecordFactory(o,[:Cost,:∇])
    @test isa(rf[:All], RecordGroup)
    @test isa(rf[:All].group[1], RecordCost)
    @test isa(rf[:All].group[2], RecordEntry)
    @test isa( RecordFactory(o,[2])[:All], RecordEvery)
    @test rf[:All].group[2].field == :∇
    @test length(rf[:All].group) == 2
    @test all(  isa.( RecordFactory(o,[:Cost, :Iteration, :Change, :Iterate])[:All].group,
         [RecordCost, RecordIteration, RecordChange, RecordIterate]) )
    @test RecordActionFactory(o,g) == g
end