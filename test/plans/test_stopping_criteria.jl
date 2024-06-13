using Manifolds, Manopt, Test, ManifoldsBase, Dates

struct TestStopProblem <: AbstractManoptProblem{ManifoldsBase.DefaultManifold} end
struct TestStopState <: AbstractManoptSolverState end
struct myStoppingCriteriaSet <: StoppingCriterionSet end
struct DummyStoppingCriterion <: StoppingCriterion end

@testset "StoppingCriteria" begin
    @testset "Generic Tests" begin
        @test_throws ErrorException get_stopping_criteria(myStoppingCriteriaSet())

        s = StopWhenAll(StopAfterIteration(10), StopWhenChangeLess(Euclidean(), 0.1))
        @test Manopt.indicates_convergence(s) #due to all and change this is true
        @test startswith(repr(s), "StopWhenAll with the")
        s2 = StopWhenAll([StopAfterIteration(10), StopWhenChangeLess(Euclidean(), 0.1)])
        @test get_stopping_criteria(s)[1].maxIter == get_stopping_criteria(s2)[1].maxIter

        s3 = StopWhenCostLess(0.1)
        p = DefaultManoptProblem(
            Euclidean(), ManifoldGradientObjective((M, x) -> x^2, x -> 2x)
        )
        s = GradientDescentState(Euclidean(), 1.0)
        @test !s3(p, s, 1)
        @test length(s3.reason) == 0
        s.p = 0.3

        @test s3(p, s, 2)
        @test length(s3.reason) > 0
        # repack
        sn = StopWhenAny(StopAfterIteration(10), s3)
        @test !Manopt.indicates_convergence(sn) # since it might stop after 10 iterations
        @test startswith(repr(sn), "StopWhenAny with the")
        @test Manopt._fast_any(x -> false, ())

        sn2 = StopAfterIteration(10) | s3
        @test get_stopping_criteria(sn)[1].maxIter == get_stopping_criteria(sn2)[1].maxIter
        @test get_stopping_criteria(sn)[2].threshold ==
            get_stopping_criteria(sn2)[2].threshold
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
        # Dummy without iterations has a reasonable fallback
        @test Manopt.get_count(DummyStoppingCriterion(), Val(:Iterations)) == 0

        sn = StopWhenAny([StopAfterIteration(10)])
        @test sn isa StoppingCriterion
    end

    @testset "Test StopAfter" begin
        p = TestStopProblem()
        o = TestStopState()
        s = StopAfter(Millisecond(30))
        @test !Manopt.indicates_convergence(s)
        @test Manopt.status_summary(s) == "stopped after $(s.threshold):\tnot reached"
        @test repr(s) == "StopAfter(Millisecond(30))\n    $(Manopt.status_summary(s))"
        s(p, o, 0) # Start
        @test s(p, o, 1) == false
        sleep(0.05)
        @test s(p, o, 2) == true
        @test_throws ErrorException StopAfter(Second(-1))
        @test_throws ErrorException update_stopping_criterion!(s, :MaxTime, Second(-1))
        update_stopping_criterion!(s, :MaxTime, Second(2))
        @test s.threshold == Second(2)
    end

    @testset "Stopping Criterion &/| operators" begin
        a = StopAfterIteration(200)
        b = StopWhenChangeLess(Euclidean(), 1e-6)
        sb = "StopWhenChangeLess(M, 1.0e-6)\n    $(Manopt.status_summary(b))"
        @test repr(b) == sb
        b2 = StopWhenChangeLess(Euclidean(), 1e-6) # second constructor
        @test repr(b2) == sb
        c = StopWhenGradientNormLess(1e-6)
        sc = "StopWhenGradientNormLess(1.0e-6)\n    $(Manopt.status_summary(c))"
        @test repr(c) == sc
        c2 = StopWhenSubgradientNormLess(1e-6)
        sc2 = "StopWhenSubgradientNormLess(1.0e-6)\n    $(Manopt.status_summary(c2))"
        @test repr(c2) == sc2
        d = StopWhenAll(a, b, c)
        @test typeof(d) === typeof(a & b & c)
        @test typeof(d) === typeof(a & (b & c))
        @test typeof(d) === typeof((a & b) & c)
        update_stopping_criterion!(d, :MinIterateChange, 1e-8)
        @test d.criteria[2].threshold == 1e-8
        @test length((d & d).criteria) == 6
        e = a | b | c
        @test typeof(e) === typeof(a | b | c)
        @test typeof(e) === typeof(a | (b | c))
        @test typeof(e) === typeof((a | b) | c)
        update_stopping_criterion!(e, :MinGradNorm, 1e-9)
        @test e.criteria[3].threshold == 1e-9
        @test length((e | e).criteria) == 6
    end

    @testset "Stopping Criterion print&summary" begin
        f = StopWhenStepsizeLess(1e-6)
        sf1 = "Stepsize s < 1.0e-6:\tnot reached"
        @test Manopt.status_summary(f) == sf1
        sf2 = "StopWhenStepsizeLess(1.0e-6)\n    $(sf1)"
        @test repr(f) == sf2
        g = StopWhenCostLess(1e-4)
        @test Manopt.status_summary(g) == "f(x) < $(1e-4):\tnot reached"
        @test repr(g) == "StopWhenCostLess(0.0001)\n    $(Manopt.status_summary(g))"
        gf(M, p) = norm(p)
        grad_gf(M, p) = p
        gp = DefaultManoptProblem(Euclidean(2), ManifoldGradientObjective(gf, grad_gf))
        gs = GradientDescentState(Euclidean(2))
        Manopt.set_iterate!(gs, Euclidean(2), [0.0, 1e-2])
        g(gp, gs, 0) # reset
        @test length(g.reason) == 0
        @test !g(gp, gs, 1)
        Manopt.set_iterate!(gs, Euclidean(2), [0.0, 1e-8])
        @test g(gp, gs, 2)
        @test length(g.reason) > 0
        h = StopWhenSmallerOrEqual(:p, 1e-4)
        @test repr(h) ==
            "StopWhenSmallerOrEqual(:p, $(1e-4))\n    $(Manopt.status_summary(h))"
        swgcl1 = StopWhenGradientChangeLess(Euclidean(2), 1e-8)
        swgcl2 = StopWhenGradientChangeLess(1e-8)
        for swgcl in [swgcl1, swgcl2]
            repr(swgcl) ==
            "StopWhenGradientChangeLess($(1e-8); vector_transport_method=ParallelTransport())\n $(Manopt.status_summary(swgcl))"
            swgcl(gp, gs, 0) # reset
            @test swgcl(gp, gs, 1) # change 0 -> true
            @test endswith(Manopt.status_summary(swgcl), "reached")
        end
        update_stopping_criterion!(swgcl2, :MinGradientChange, 1e-9)
        @test swgcl2.threshold == 1e-9
    end

    @testset "TCG stopping criteria" begin
        # create dummy criterion
        ho = ManifoldHessianObjective(x -> x, (M, x) -> x, (M, x) -> x, x -> x)
        hp = DefaultManoptProblem(Euclidean(), ho)
        tcgs = TruncatedConjugateGradientState(
            TangentSpace(Euclidean(), 1.0), 0.0; trust_region_radius=2.0, randomize=false
        )
        tcgs.model_value = 1.0
        s = StopWhenModelIncreased()
        @test !s(hp, tcgs, 0)
        @test s.reason == ""
        s.model_value = 0.5 # tweak the model value to trigger a test
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

    @testset "Test further setters" begin
        mgo = ManifoldGradientObjective((M, x) -> x^2, x -> 2x)
        dmp = DefaultManoptProblem(Euclidean(), mgo)
        gds = GradientDescentState(
            Euclidean(),
            1.0;
            stopping_criterion=StopAfterIteration(100),
            stepsize=ConstantStepsize(Euclidean()),
        )
        swecl = StopWhenEntryChangeLess(:p, (p, s, v, w) -> norm(w - v), 1e-5)
        @test startswith(repr(swecl), "StopWhenEntryChangeLess\n")
        update_stopping_criterion!(swecl, :Threshold, 1e-4)
        @test swecl.threshold == 1e-4
        @test !swecl(dmp, gds, 1) #First call stores
        @test swecl(dmp, gds, 2) #Second triggers (no change)
        swecl(dmp, gds, 0) # reset
        @test length(swecl.reason) == 0
    end

    @testset "Subgradient Norm Stopping Criterion" begin
        M = Euclidean(2)
        p = [1.0, 2.0]
        f(M, q) = distance(M, q, p)
        function ∂f(M, q)
            if distance(M, p, q) == 0
                return zero_vector(M, q)
            end
            return -log(M, q, p) / max(10 * eps(Float64), distance(M, p, q))
        end
        mso = ManifoldSubgradientObjective(f, ∂f)
        mp = DefaultManoptProblem(M, mso)
        c2 = StopWhenSubgradientNormLess(1e-6)
        sc2 = "StopWhenSubgradientNormLess(1.0e-6)\n    $(Manopt.status_summary(c2))"
        @test repr(c2) == sc2
        st = SubGradientMethodState(M, p; stopping_criterion=c2)
        st.X = ∂f(M, 2p)
        @test !c2(mp, st, 1)
        st.X = ∂f(M, p)
        @test c2(mp, st, 2)
        @test length(get_reason(c2)) > 0
        c2(mp, st, 0) # verify that reset works
        @test length(get_reason(c2)) == 0
        @test Manopt.indicates_convergence(c2)
        update_stopping_criterion!(c2, :MinSubgradNorm, 1e-8)
        @test c2.threshold == 1e-8
    end

    @testset "StopWhenCostNaN & StopWhenIterateNaN" begin
        sc1 = StopWhenCostNaN()
        f(M, p) = NaN
        M = Euclidean(2)
        p = [1.0, 2.0]
        @test startswith(repr(sc1), "StopWhenCostNaN()\n")
        mco = ManifoldCostObjective(f)
        mp = DefaultManoptProblem(M, mco)
        s = NelderMeadState(M)
        @test sc1(mp, s, 1) #always returns true since `f` is always NaN
        @test !sc1(mp, s, 0) # test reset
        @test length(sc1.reason) == sc1.at_iteration # verify reset

        s.p .= NaN
        sc2 = StopWhenIterateNaN()
        @test startswith(repr(sc2), "StopWhenIterateNaN()\n")
        @test sc2(mp, s, 1) #always returns true since p was now set to NaN
        @test !sc2(mp, s, 0) # test reset
        @test length(sc2.reason) == sc2.at_iteration # verify reset
    end
end
