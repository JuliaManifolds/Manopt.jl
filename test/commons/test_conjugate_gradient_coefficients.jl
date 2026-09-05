using Manopt, Manifolds, Test

struct DummyCGCoeff <: DirectionUpdateRule end
(::DummyCGCoeff)(pr, st, k; kwargs...) = 0.2
(::Manopt.DirectionUpdateRuleStorage{DummyCGCoeff})(pr, st, k) = 0.2
Manopt.update_rule_storage_points(::DummyCGCoeff) = Tuple{}
Manopt.update_rule_storage_vectors(::DummyCGCoeff) = Tuple{}

@testset "Conjugate Gradient Coefficients" begin
    @testset "Test Restart CG" begin
        M = Euclidean(2)
        du = DummyCGCoeff()
        dur2 = ConjugateGradientBealeRestart(du; threshold = 0.3)
        dur3 = ConjugateGradientBealeRestart(du; threshold = 0.1)
        f(M, p) = norm(M, p)^2
        grad_f(M, p) = p
        p0 = [1.0, 0.0]
        pr = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        cgs2 = ConjugateGradientDescentState(
            M; p = p0,
            stopping_criterion = StopAfterIteration(2),
            stepsize = Manopt.ConstantStepsize(M, 1.0),
            coefficient = dur2,
        )
        cgs2.X = [0.0, 0.2]
        # Fake update history to get a certain old X and old p
        cgs2.coefficient(pr, cgs2, 0)
        # the inner check is 0.2 which is still less than 0.3
        @test cgs2.coefficient(pr, cgs2, 1) != 0
        cgs3 = ConjugateGradientDescentState(
            M; p = p0,
            stopping_criterion = StopAfterIteration(2),
            stepsize = Manopt.ConstantStepsize(M, 1.0),
            coefficient = dur3,
        )
        cgs3.X = [0.0, 0.2]
        # Fake update history to get a certain old X and old p
        cgs3.coefficient(pr, cgs3, 0)
        # then we are above the threshold 0.1 (namely at 0.2) and we get a descent step
        @test cgs3.coefficient(pr, cgs3, 1) == 0
    end
    @testset "representation and summary of Coefficients" begin
        p = ParallelTransport()
        pt = repr(p)
        M = Euclidean(2)
        @test repr(Manopt.ConjugateDescentCoefficient()(M)) ==
            "Manopt.ConjugateDescentCoefficientRule()"
        @test repr(FletcherReevesCoefficient()()) ==
            "Manopt.FletcherReevesCoefficientRule()"
        # either in the factory constructor or in the factory call we need M
        # so lets alternate
        @test repr(Manopt.DaiYuanCoefficient(M; vector_transport_method = p)()) ==
            "Manopt.DaiYuanCoefficientRule(; vector_transport_method=$pt)"
        @test repr(HagerZhangCoefficient(; vector_transport_method = p)(M)) ==
            "Manopt.HagerZhangCoefficientRule(; vector_transport_method=$pt)"
        @test repr(HestenesStiefelCoefficient()(M)) ==
            "Manopt.HestenesStiefelCoefficientRule(; vector_transport_method=$pt)"
        # Requires a manifold
        @test_throws MethodError HestenesStiefelCoefficient()()
        @test repr(PolakRibiereCoefficient()(M)) ==
            "Manopt.PolakRibiereCoefficientRule(; vector_transport_method=$(pt))"
        cgbr = Manopt.ConjugateGradientBealeRestartRule(
            Euclidean(), ConjugateDescentCoefficient()
        )
        s1 = "Manopt.ConjugateGradientBealeRestartRule(Manopt.ConjugateDescentCoefficientRule(); threshold=0.2, vector_transport_method=$(pt))"
        @test repr(cgbr) == s1
        cgbr2 = Manopt.ConjugateGradientBealeRestartRule(ConjugateDescentCoefficient())
        @test cgbr2.threshold == cgbr.threshold
        @test repr(LiuStoreyCoefficient(M)()) ==
            "Manopt.LiuStoreyCoefficientRule(; vector_transport_method=$pt)"
        hcs = repr(HybridCoefficient(PolakRibiereCoefficient(), FletcherReevesCoefficient())(M))
        @test contains(hcs, "Manopt.HybridCoefficientRule")
        @test contains(hcs, "Manopt.PolakRibiereCoefficientRule")
        @test contains(hcs, "Manopt.FletcherReevesCoefficientRule")
        @test contains(hcs, "Manopt.SteepestDescentCoefficientRule")
        @test contains(hcs, "lower_bound_scale = 1.0")
    end
    @testset "Dai-Yuan on a point-dependent metric" begin
        # the denominator inner product is evaluated at the new point
        Ms = SymmetricPositiveDefinite(2)
        f(M, q) = distance(M, q, [1.0 0.0; 0.0 1.0])^2
        grad_f(M, q) = -2 * log(M, q, [1.0 0.0; 0.0 1.0])
        dmp = DefaultManoptProblem(Ms, ManifoldGradientObjective(f, grad_f))
        p_old = [2.0 0.0; 0.0 1.0]
        cgs = ConjugateGradientDescentState(Ms; p = [1.5 0.1; 0.1 1.0])
        cgs.X = grad_f(Ms, cgs.p)
        X_old = grad_f(Ms, p_old)
        δ_old = -X_old
        dy = Manopt.DaiYuanCoefficientRule(Ms)
        β = dy(dmp, cgs, 1; p = p_old, X = X_old, δ = δ_old)
        vtm = dy.vector_transport_method
        ν = cgs.X - vector_transport_to(Ms, p_old, X_old, cgs.p, vtm)
        δtr = vector_transport_to(Ms, p_old, δ_old, cgs.p, vtm)
        @test β ≈ inner(Ms, cgs.p, cgs.X, cgs.X) / inner(Ms, cgs.p, δtr, ν)
    end
end
