s = joinpath(@__DIR__, "..", "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using Manifolds, ManifoldsBase, Manopt, ManoptTestSuite, Test, ManifoldsBase, Dates

@testset "StoppingCriteria" begin
    @testset "Generic Tests" begin
        @test_throws ErrorException get_stopping_criteria(
            ManoptTestSuite.DummmyStoppingCriteriaSet()
        )

        s = StopWhenAll(StopAfterIteration(10), StopWhenChangeLess(Euclidean(), 0.1))
        @test Manopt.indicates_convergence(s) #due to all and change this is true
        @test startswith(repr(s), "StopWhenAll with the")
        @test get_reason(s) === ""
        # Trigger second one manually
        s.criteria[2].last_change = 0.05
        s.criteria[2].at_iteration = 3
        @test length(get_reason(s.criteria[2])) > 0
        s2 = StopWhenAll([StopAfterIteration(10), StopWhenChangeLess(Euclidean(), 0.1)])
        @test get_stopping_criteria(s)[1].max_iterations ==
            get_stopping_criteria(s2)[1].max_iterations

        s3 = StopWhenCostLess(0.1)
        p = DefaultManoptProblem(
            Euclidean(), ManifoldFirstOrderObjective((M, x) -> x^2, x -> 2x)
        )
        s = GradientDescentState(Euclidean(); p=1.0)
        @test !s3(p, s, 1)
        @test length(get_reason(s3)) == 0
        s.p = 0.3

        @test s3(p, s, 2)
        @test length(get_reason(s3)) > 0
        # repack
        sn = StopWhenAny(StopAfterIteration(10), s3)
        @test get_reason(sn) == ""
        @test !Manopt.indicates_convergence(sn) # since it might stop after 10 iterations
        @test startswith(repr(sn), "StopWhenAny with the")
        @test Manopt._fast_any(x -> false, ())

        sn2 = StopAfterIteration(10) | s3
        @test get_stopping_criteria(sn)[1].max_iterations ==
            get_stopping_criteria(sn2)[1].max_iterations
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
        an = get_reason(sm)
        m = match(r"^((.*)\n)+", an)
        @test length(m.captures) == 2 # both have to be active
        Manopt.set_parameter!(s3, :MinCost, 1e-2)
        @test s3.threshold == 1e-2
        # Dummy without iterations has a reasonable fallback
        @test Manopt.get_count(
            ManoptTestSuite.DummyStoppingCriterion(), Val(:Iterations)
        ) == 0

        sn = StopWhenAny([StopAfterIteration(10)])
        @test sn isa StoppingCriterion
    end

    @testset "Test StopAfter" begin
        p = ManoptTestSuite.DummyProblem{ManifoldsBase.DefaultManifold}()
        o = ManoptTestSuite.DummyState()
        s = StopAfter(Millisecond(30))
        @test !Manopt.indicates_convergence(s)
        @test Manopt.status_summary(s) == "stopped after $(s.threshold):\tnot reached"
        @test repr(s) == "StopAfter(Millisecond(30))\n    $(Manopt.status_summary(s))"
        s(p, o, 0) # Start
        @test s(p, o, 1) == false
        @test get_reason(s) == ""
        sleep(0.05)
        @test s(p, o, 2) == true
        @test length(get_reason(s)) > 0
        @test_throws ErrorException StopAfter(Second(-1))
        @test_throws ErrorException Manopt.set_parameter!(s, :MaxTime, Second(-1))
        Manopt.set_parameter!(s, :MaxTime, Second(2))
        @test s.threshold == Second(2)
    end

    @testset "Stopping Criterion &/| operators" begin
        a = StopAfterIteration(200)
        b = StopWhenChangeLess(Euclidean(), 1e-6)
        sb = "StopWhenChangeLess with threshold 1.0e-6.\n    $(Manopt.status_summary(b))"
        @test repr(b) == sb
        @test get_reason(b) == ""
        b2 = StopWhenChangeLess(Euclidean(), 1e-6) # second constructor
        @test repr(b2) == sb
        c = StopWhenGradientNormLess(1e-6)
        sc = "StopWhenGradientNormLess(1.0e-6)\n    $(Manopt.status_summary(c))"
        @test repr(c) == sc
        @test get_reason(c) == ""
        # Trigger manually
        c.last_change = 1e-11
        c.at_iteration = 3
        @test length(get_reason(c)) > 0
        c2 = StopWhenSubgradientNormLess(1e-6)
        sc2 = "StopWhenSubgradientNormLess(1.0e-6)\n    $(Manopt.status_summary(c2))"
        @test repr(c2) == sc2
        d = StopWhenAll(a, b, c)
        @test typeof(d) === typeof(a & b & c)
        @test typeof(d) === typeof(a & (b & c))
        @test typeof(d) === typeof((a & b) & c)
        Manopt.set_parameter!(d, :MinIterateChange, 1e-8)
        @test d.criteria[2].threshold == 1e-8
        @test length((d & d).criteria) == 6
        e = a | b | c
        @test typeof(e) === typeof(a | b | c)
        @test typeof(e) === typeof(a | (b | c))
        @test typeof(e) === typeof((a | b) | c)
        Manopt.set_parameter!(e, :MinGradNorm, 1e-9)
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
        gp = DefaultManoptProblem(Euclidean(2), ManifoldFirstOrderObjective(gf, grad_gf))
        gs = GradientDescentState(Euclidean(2))
        Manopt.set_iterate!(gs, Euclidean(2), [0.0, 1e-2])
        g(gp, gs, 0) # reset
        @test length(get_reason(g)) == 0
        @test !g(gp, gs, 1)
        Manopt.set_iterate!(gs, Euclidean(2), [0.0, 1e-8])
        @test g(gp, gs, 2)
        @test length(get_reason(g)) > 0
        h = StopWhenSmallerOrEqual(:p, 1e-4)
        @test repr(h) ==
            "StopWhenSmallerOrEqual(:p, $(1e-4))\n    $(Manopt.status_summary(h))"
        @test get_reason(h) == ""
        # Trigger manually
        h.at_iteration = 1
        @test length(get_reason(h)) > 0
        swgcl1 = StopWhenGradientChangeLess(Euclidean(2), 1e-8)
        swgcl2 = StopWhenGradientChangeLess(1e-8)
        for swgcl in [swgcl1, swgcl2]
            repr(swgcl) ==
            "StopWhenGradientChangeLess($(1e-8); vector_transport_method=ParallelTransport())\n $(Manopt.status_summary(swgcl))"
            swgcl(gp, gs, 0) # reset
            @test get_reason(swgcl) == ""
            @test swgcl(gp, gs, 1) # change 0 -> true
            @test endswith(Manopt.status_summary(swgcl), "reached")
            @test length(get_reason(swgcl)) > 0
        end
        Manopt.set_parameter!(swgcl2, :MinGradientChange, 1e-9)
        @test swgcl2.threshold == 1e-9
    end

    @testset "TCG stopping criteria" begin
        # create dummy criterion
        ho = ManifoldHessianObjective(x -> x, (M, x) -> x, (M, x) -> x, x -> x)
        hp = DefaultManoptProblem(Euclidean(), ho)
        tcgs = TruncatedConjugateGradientState(
            TangentSpace(Euclidean(), 1.0); X=0.0, trust_region_radius=2.0, randomize=false
        )
        tcgs.model_value = 1.0
        s = StopWhenModelIncreased()
        @test !s(hp, tcgs, 0)
        @test length(get_reason(s)) == 0
        s.model_value = 0.5 # tweak the model value to trigger a test
        @test s(hp, tcgs, 1)
        @test length(get_reason(s)) > 0
        s2 = StopWhenCurvatureIsNegative()
        tcgs.δHδ = -1.0
        @test !s2(hp, tcgs, 0)
        @test length(get_reason(s2)) == 0
        @test s2(hp, tcgs, 1)
        @test length(get_reason(s2)) > 0
        s3 = StopWhenResidualIsReducedByFactorOrPower()
        Manopt.set_parameter!(s3, :ResidualFactor, 0.5)
        @test s3.κ == 0.5
        Manopt.set_parameter!(s3, :ResidualPower, 0.5)
        @test s3.θ == 0.5
        @test get_reason(s3) == ""
        # Trigger manually
        s3.at_iteration = 1
        @test length(get_reason(s3)) > 0
    end

    @testset "Stop with step size" begin
        mgo = ManifoldFirstOrderObjective((M, x) -> x^2, x -> 2x)
        dmp = DefaultManoptProblem(Euclidean(), mgo)
        gds = GradientDescentState(
            Euclidean();
            p=1.0,
            stopping_criterion=StopAfterIteration(100),
            stepsize=Manopt.ConstantStepsize(Euclidean()),
        )
        s1 = StopWhenStepsizeLess(0.5)
        @test !s1(dmp, gds, 1)
        @test length(get_reason(s1)) == 0
        gds.stepsize = Manopt.ConstantStepsize(Euclidean(), 0.25)
        @test s1(dmp, gds, 2)
        @test length(get_reason(s1)) > 0
        Manopt.set_parameter!(gds, :StoppingCriterion, :MaxIteration, 200)
        @test gds.stop.max_iterations == 200
        Manopt.set_parameter!(s1, :MinStepsize, 1e-1)
        @test s1.threshold == 1e-1
    end

    @testset "Test further setters" begin
        mgo = ManifoldFirstOrderObjective((M, x) -> x^2, x -> 2x)
        dmp = DefaultManoptProblem(Euclidean(), mgo)
        gds = GradientDescentState(
            Euclidean();
            p=1.0,
            stopping_criterion=StopAfterIteration(100),
            stepsize=Manopt.ConstantStepsize(Euclidean()),
        )
        swecl = StopWhenEntryChangeLess(:p, (p, s, v, w) -> norm(w - v), 1e-5)
        @test startswith(repr(swecl), "StopWhenEntryChangeLess\n")
        Manopt.set_parameter!(swecl, :Threshold, 1e-4)
        @test swecl.threshold == 1e-4
        @test !swecl(dmp, gds, 1) #First call stores
        @test length(get_reason(swecl)) == 0
        @test swecl(dmp, gds, 2) #Second triggers (no change)
        @test length(get_reason(swecl)) > 0
        swecl(dmp, gds, 0) # reset
        @test length(get_reason(swecl)) == 0
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
        st = SubGradientMethodState(M; p=p, stopping_criterion=c2)
        st.X = ∂f(M, 2p)
        @test !c2(mp, st, 1)
        st.X = ∂f(M, p)
        @test c2(mp, st, 2)
        @test length(get_reason(c2)) > 0
        c2(mp, st, 0) # verify that reset works
        @test length(get_reason(c2)) == 0
        @test Manopt.indicates_convergence(c2)
        Manopt.set_parameter!(c2, :MinSubgradNorm, 1e-8)
        @test c2.threshold == 1e-8
    end

    @testset "StopWhenCostNaN, StopWhenCostChangeLess, StopWhenIterateNaN" begin
        sc1 = StopWhenCostNaN()
        f(M, p) = norm(p) > 2 ? NaN : norm(p)
        M = Euclidean(2)
        p = [1.0, 2.0]
        @test startswith(repr(sc1), "StopWhenCostNaN()\n")
        mco = ManifoldCostObjective(f)
        mp = DefaultManoptProblem(M, mco)
        s = NelderMeadState(M)
        s.p = p
        @test sc1(mp, s, 1) #always returns true since `f` is always NaN
        s.p = [0.0, 0.1]
        @test !sc1(mp, s, 0) # test reset. triggers again
        @test length(get_reason(sc1)) == 0
        @test sc1.at_iteration == -1
        # Trigger manually
        sc1.at_iteration = 1
        @test length(get_reason(sc1)) > 0

        sc2 = StopWhenCostChangeLess(1e-6)
        @test startswith(repr(sc2), "StopWhenCostChangeLess with threshold 1.0e-6.\n")
        @test get_reason(sc2) == ""
        s.p = [0.0, 0.1]
        @test !sc2(mp, s, 1) # Init check
        @test length(get_reason(sc2)) == 0
        @test sc2(mp, s, 2) # change zero -> triggers
        @test length(get_reason(sc2)) > 0
        # Reset
        @test !sc2(mp, s, 0) # reset
        @test length(get_reason(sc2)) == 0

        s.p .= NaN
        sc3 = StopWhenIterateNaN()
        @test startswith(repr(sc3), "StopWhenIterateNaN()\n")
        @test sc3(mp, s, 1) #always returns true since p was now set to NaN
        @test length(get_reason(sc3)) > 0
        s.p = p
        @test !sc3(mp, s, 0) # test reset, though this already again triggers
        @test length(get_reason(sc3)) == 0 # verify reset
        @test sc3.at_iteration == -1
        # Trigger manually
        sc3.at_iteration = 1
        @test length(get_reason(sc3)) > 0
    end

    @testset "StopWhenRepeated" begin
        p = ManoptTestSuite.DummyProblem{ManifoldsBase.DefaultManifold}()
        o = ManoptTestSuite.DummyState()
        s = StopAfterIteration(2)
        sc = StopWhenRepeated(s, 3)
        sc2 = s × 3
        @test Manopt.indicates_convergence(sc) == Manopt.indicates_convergence(s)
        @test get_reason(sc) == ""
        @test startswith(repr(sc), "StopWhenRepeated with the Stopping Criterion:\n")
        @test startswith(Manopt.status_summary(sc), "0 ≥ 3 (consecutive): not reached")
        @test !sc(p, o, 1) # still count 0
        @test !sc(p, o, 2) # 1
        @test !sc(p, o, 2) # 2
        @test sc(p, o, 3) # 3 -> hits
        @test length(get_reason(sc)) > 0
        # reset
        sc(p, o, 0) # reset
        @test length(get_reason(sc)) == 0
    end

    @testset "StopWhenCriterionWithIterationCondition" begin
        f(M, p) = 0.0 # Always triggrers
        M = Euclidean(2)
        p = [1.0, 2.0]
        mco = ManifoldCostObjective(f)
        mp = DefaultManoptProblem(M, mco)
        st = NelderMeadState(M)
        st.p = p

        s = StopWhenCostLess(1e-4)

        sc = StopWhenCriterionWithIterationCondition(s, 5)
        @test Manopt.indicates_convergence(sc) == Manopt.indicates_convergence(s)
        @test get_reason(sc) == ""
        @test startswith(
            repr(sc),
            "StopWhenCriterionWithIterationCondition with the Stopping Criterion:\n",
        )
        @test startswith(Manopt.status_summary(sc), "Base.Fix2{typeof(>), Int64}(>, 5) &&")
        sc2 = s > 5
        @test typeof(sc) === typeof(sc2)
        # Test other constructors
        sc3 = s >= 5
        @test sc3.comp === (>=(5))
        sc4 = s == 5
        @test sc4.comp === (==(5))
        sc5 = s <= 5
        @test sc5.comp === (<=(5))
        sc6 = s < 5
        @test sc6.comp === (<(5))

        # test that it does not hit at 5
        @test !sc(mp, st, 5) # still count 0
        @test sc(mp, st, 6) # triggers
        @test length(get_reason(sc)) > 0
        sc(mp, st, 0) # reset
        @test length(get_reason(sc)) == 0
    end
end
