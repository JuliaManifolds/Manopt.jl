using Manifolds, Manopt, Test, ManifoldsBase, Dates

struct TestProblem <: Problem end
struct TestOptions <: Options end

@testset "StoppingCriteria" begin
    struct myStoppingCriteriaSet <: StoppingCriterionSet end
    @test_throws ErrorException get_stopping_criteria(myStoppingCriteriaSet())

    s = StopWhenAll(StopAfterIteration(10), StopWhenChangeLess(0.1))
    s2 = StopWhenAll([StopAfterIteration(10), StopWhenChangeLess(0.1)])
    @test get_stopping_criteria(s)[1].maxIter == get_stopping_criteria(s2)[1].maxIter

    s3 = StopWhenCostLess(0.1)
    p = GradientProblem(Euclidean(1), x -> x^2, x -> 2x)
    o = GradientDescentOptions(1.0)
    @test !s3(p, o, 1)
    @test length(s3.reason) == 0
    o.x = 0.3
    @test s3(p, o, 2)
    @test length(s3.reason) > 0
    # repack
    sn = StopWhenAny(StopAfterIteration(10), s3)
    sn2 = StopWhenAny([StopAfterIteration(10), s3])
    @test get_stopping_criteria(sn)[1].maxIter == get_stopping_criteria(sn2)[1].maxIter
    @test get_stopping_criteria(sn)[2].threshold == get_stopping_criteria(sn2)[2].threshold
    # then s3 is the only active one
    @test get_active_stopping_criteria(sn) == [s3]
    @test get_active_stopping_criteria(s3) == [s3]
    @test get_active_stopping_criteria(StopAfterIteration(1)) == []
    sm = StopWhenAll(StopAfterIteration(10), s3)
    @test !sm(p, o, 9)
    @test sm(p, o, 11)
    an = sm.reason
    m = match(r"^((.*)\n)+", an)
    @test length(m.captures) == 2 # both have to be active
end

@testset "Test StopAfter" begin
    p = TestProblem()
    o = TestOptions()
    s = StopAfter(Second(1))
    s(p, o, 0) # Start
    @test s(p, o, 1) == false
    sleep(1.02)
    @test s(p, o, 2) == true
    @test_throws ErrorException StopAfter(Second(-1))
end

@testset "Stopping Criterion &/| operators" begin
    a = StopAfterIteration(200)
    b = StopWhenChangeLess(1e-6)
    c = StopWhenGradientNormLess(1e-6)
    d = StopWhenAll(a, b, c)
    @test typeof(d) === typeof(a & b & c)
    @test typeof(d) === typeof(a & (b & c))
    @test typeof(d) === typeof((a & b) & c)
    e = StopWhenAny(a, b, c)
    @test typeof(e) === typeof(a | b | c)
    @test typeof(e) === typeof(a | (b | c))
    @test typeof(e) === typeof((a | b) | c)
end
