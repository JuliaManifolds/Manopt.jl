@testset "StoppingCriteria" begin
    struct myStoppingCriteriaSet <: StoppingCriterionSet end
    @test_throws ErrorException get_stopping_criteria(myStoppingCriteriaSet())

    s = StopWhenAll(StopAfterIteration(10),StopWhenChangeLess(0.1))
    s2 = StopWhenAll([StopAfterIteration(10),StopWhenChangeLess(0.1)])
    @test get_stopping_criteria(s)[1].maxIter == get_stopping_criteria(s2)[1].maxIter

    s3 = StopWhenCostLess(0.1)
    p = GradientProblem(Euclidean(1), x->x^2, x->2x)
    a = ArmijoLinesearch()
    o = GradientDescentOptions(1.0)
    @test !s3(p,o,1)
    @test length(s3.reason) == 0
    o.x = 0.3
    @test s3(p,o,2)
    @test length(s3.reason) > 0
    # repack
    sn = StopWhenAny(StopAfterIteration(10), s3)
    # then s3 is the only active one
    @test get_active_stopping_criteria(sn) == [s3]
    sm = StopWhenAll(StopAfterIteration(10),s3)
    @test !sm(p,o,9)
    @test sm(p,o,11)
    an = sm.reason
    m = match(r"^((.*)\n)+",an)
    @test length(m.captures)==2 # both have to be active
end