using Manifolds, Manopt, Test, ManifoldsBase, Dates

struct TestStopProblem <: AbstractManoptProblem{ManifoldsBase.DefaultManifold} end
struct TestStopState <: AbstractManoptSolverState end
struct myStoppingCriteriaSet <: StoppingCriterionSet end
struct DummyStoppingCriterion <: StoppingCriterion end

@testset "StoppingCriteria" begin
    @test_throws ErrorException get_stopping_criteria(myStoppingCriteriaSet())

    s = StopWhenAll(StopAfterIteration(10), StopWhenChangeLess(0.1))
    @test Manopt.indicates_convergence(s) #due to all and change this is true
    @test startswith(repr(s), "StopWhenAll with the")
    s2 = StopWhenAll([StopAfterIteration(10), StopWhenChangeLess(0.1)])
    @test get_stopping_criteria(s)[1].maxIter == get_stopping_criteria(s2)[1].maxIter

    s3 = StopWhenCostLess(0.1)
    p = DefaultManoptProblem(Euclidean(), ManifoldGradientObjective((M, x) -> x^2, x -> 2x))
    s = GradientDescentState(Euclidean(), 1.0)
    @test !s3(p, s, 1)
    @test length(s3.reason) == 0
    s.p = 0.3
    @test s3(p, s, 2)
    @test length(s3.reason) > 0
    # repack
    sn = StopWhenAny(StopAfterIteration(10), s3)
    @test !Manopt.indicates_convergence(sn) # since it might stop after 10 iters
    @test startswith(repr(sn), "StopWhenAny with the")
    sn2 = StopWhenAny([StopAfterIteration(10), s3])
    @test get_stopping_criteria(sn)[1].maxIter == get_stopping_criteria(sn2)[1].maxIter
    @test get_stopping_criteria(sn)[2].threshold == get_stopping_criteria(sn2)[2].threshold
    # then s3 is the only active one
    @test get_active_stopping_criteria(sn) == [s3]
    @test get_active_stopping_criteria(s3) == [s3]
    @test get_active_stopping_criteria(StopAfterIteration(1)) == []
    sm = StopWhenAll(StopAfterIteration(10), s3)
    s1 = "StopAfterIteration(10)\n    Max Iteration 10:\tnot reached"
    @test repr(StopAfterIteration(10)) == s1
    @test !sm(p, s, 9)
    @test sm(p, s, 11)
    an = sm.reason
    m = match(r"^((.*)\n)+", an)
    @test length(m.captures) == 2 # both have to be active
    update_stopping_criterion!(s3, :MinCost, 1e-2)
    @test s3.threshold == 1e-2
    # Dummy withoout iterations has a reasonable fallback
    @test Manopt.get_count(DummyStoppingCriterion(), Val(:Iterations)) == 0
end

@testset "Test StopAfter" begin
    p = TestStopProblem()
    o = TestStopState()
    s = StopAfter(Second(1))
    @test !Manopt.indicates_convergence(s)
    @test Manopt.status_summary(s) == "stopped after $(s.threshold):\tnot reached"
    @test repr(s) == "StopAfter(Second(1))\n    $(Manopt.status_summary(s))"
    s(p, o, 0) # Start
    @test s(p, o, 1) == false
    sleep(1.02)
    @test s(p, o, 2) == true
    @test_throws ErrorException StopAfter(Second(-1))
    @test_throws ErrorException update_stopping_criterion!(s, :MaxTime, Second(-1))
    update_stopping_criterion!(s, :MaxTime, Second(2))
    @test s.threshold == Second(2)
end

@testset "Stopping Criterion &/| operators" begin
    a = StopAfterIteration(200)
    b = StopWhenChangeLess(1e-6)
    sb = "StopWhenChangeLess(1.0e-6)\n    $(Manopt.status_summary(b))"
    @test repr(b) == sb
    b2 = StopWhenChangeLess(Euclidean(), 1e-6) # second constructor
    @test repr(b2) == sb
    c = StopWhenGradientNormLess(1e-6)
    sc = "StopWhenGradientNormLess(1.0e-6)\n    $(Manopt.status_summary(c))"
    @test repr(c) == sc
    c2 = StopWhenSubgradientNormLess(1e-6)
    d = StopWhenAll(a, b, c, c2)
    @test typeof(d) === typeof(a & b & c & c2)
    @test typeof(d) === typeof(a & (b & c & c2))
    @test typeof(d) === typeof((a & b) & c & c2)
    update_stopping_criterion!(d, :MinIterateChange, 1e-8)
    @test d.criteria[2].threshold == 1e-8
    e = StopWhenAny(a, b, c, c2)
    @test typeof(e) === typeof(a | b | c | c2)
    @test typeof(e) === typeof(a | (b | (c | c2)))
    @test typeof(e) === typeof(a | ((b | c) | c2))
    @test typeof(e) === typeof(((a | b) | c) | c2)
    @test typeof(e) === typeof((a | (b | c)) | c2)
    update_stopping_criterion!(e, :MinGradNorm, 1e-9)
    @test d.criteria[3].threshold == 1e-9
end

@testset "TCG stopping criteria" begin
    # create dummy criterion
    ho = ManifoldHessianObjective(x -> x, (M, x) -> x, (M, x) -> x, x -> x)
    hp = DefaultManoptProblem(Euclidean(), ho)
    tcgs = TruncatedConjugateGradientState(
        Euclidean(), 1.0, 0.0; trust_region_radius=2.0, randomize=false
    )
    tcgs.new_model_value = 2.0
    tcgs.model_value = 1.0
    s = StopWhenModelIncreased()
    @test !s(hp, tcgs, 0)
    @test s.reason == ""
    @test s(hp, tcgs, 1)
    @test length(s.reason) > 0
    s2 = StopWhenCurvatureIsNegative()
    tcgs.δHδ = -1.0
    @test !s2(hp, tcgs, 0)
    @test s2.reason == ""
    @test s2(hp, tcgs, 1)
    @test length(s2.reason) > 0
    s3 = StopWhenResidualIsReducedByFactorOrPower()
    update_stopping_criterion!(s3, :ResidualFactor, 0.5)
    @test s3.κ == 0.5
    update_stopping_criterion!(s3, :ResidualPower, 0.5)
    @test s3.θ == 0.5
end

@testset "Stop with step size" begin
    mgo = ManifoldGradientObjective((M, x) -> x^2, x -> 2x)
    dmp = DefaultManoptProblem(Euclidean(), mgo)
    gds = GradientDescentState(
        Euclidean(),
        1.0;
        stopping_criterion=StopAfterIteration(100),
        stepsize=ConstantStepsize(Euclidean()),
    )
    s1 = StopWhenStepsizeLess(0.5)
    @test !s1(dmp, gds, 1)
    @test s1.reason == ""
    gds.stepsize = ConstantStepsize(; stepsize=0.25)
    @test s1(dmp, gds, 2)
    @test length(s1.reason) > 0
    update_stopping_criterion!(gds, :MaxIteration, 200)
    @test gds.stop.maxIter == 200
    update_stopping_criterion!(s1, :MinStepsize, 1e-1)
    @test s1.threshold == 1e-1
end
