using Manopt, Manifolds, Test

struct DummyCGCoeff <: DirectionUpdateRule end
(u::DummyCGCoeff)(p, s, k) = 0.2
Manopt.update_rule_storage_points(::DummyCGCoeff) = Tuple{}
Manopt.update_rule_storage_vectors(::DummyCGCoeff) = Tuple{}

@testset "Conjugate Gradient Descent Plan" begin
    @testset "Test Restart CG" begin
        M = Euclidean(2)
        du = DummyCGCoeff()
        dur2 = ConjugateGradientBealeRestart(du; threshold=0.3)
        dur3 = ConjugateGradientBealeRestart(du; threshold=0.1)
        f(M, p) = norm(M, p)^2
        grad_f(M, p) = p
        p0 = [1.0, 0.0]
        pr = DefaultManoptProblem(M, ManifoldFirstOrderObjective(f, grad_f))
        cgs2 = ConjugateGradientDescentState(
            M;
            p=p0,
            stopping_criterion=StopAfterIteration(2),
            stepsize=Manopt.ConstantStepsize(M, 1.0),
            coefficient=dur2,
        )
        cgs2.X = [0.0, 0.2]
        @test cgs2.coefficient(pr, cgs2, 1) != 0
        cgs3 = ConjugateGradientDescentState(
            M;
            p=p0,
            stopping_criterion=StopAfterIteration(2),
            stepsize=Manopt.ConstantStepsize(M, 1.0),
            coefficient=dur3,
        )
        cgs3.X = [0.0, 0.2]
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
        @test repr(Manopt.DaiYuanCoefficient(M; vector_transport_method=p)()) ==
            "Manopt.DaiYuanCoefficientRule(; vector_transport_method=$pt)"
        @test repr(HagerZhangCoefficient(; vector_transport_method=p)(M)) ==
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
        @test cgbr.threshold == cgbr.threshold
        @test repr(LiuStoreyCoefficient(M)()) ==
            "Manopt.LiuStoreyCoefficientRule(; vector_transport_method=$pt)"
    end
end
