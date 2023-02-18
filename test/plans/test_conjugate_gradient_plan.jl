using Manopt, Manifolds, Test

struct DummyCGCoeff <: DirectionUpdateRule end
(u::DummyCGCoeff)(p, s, i) = 0.2

@testset "Conjugate Gradient Descent Plan" begin
    @testset "Test Restart CG" begin
        M = Euclidean(2)
        du = DummyCGCoeff()
        dur2 = ConjugateGradientBealeRestart(du, 0.3)
        dur3 = ConjugateGradientBealeRestart(du, 0.1)
        f(M, p) = norm(M, p)^2
        grad_f(M, p) = p
        p0 = [1.0, 0.0]
        pr = DefaultManoptProblem(M, ManifoldGradientObjective(f, grad_f))
        cgs2 = ConjugateGradientDescentState(
            M, p0, StopAfterIteration(2), ConstantStepsize(1.0), dur2
        )
        cgs2.X = [0.0, 0.2]
        @test cgs2.coefficient(pr, cgs2, 1) != 0
        cgs3 = ConjugateGradientDescentState(
            M, p0, StopAfterIteration(2), ConstantStepsize(1.0), dur3
        )
        cgs3.X = [0.0, 0.2]
        @test cgs3.coefficient(pr, cgs3, 1) == 0
    end
    @testset "representation and summary of Coefficients" begin
        pt = repr(ParallelTransport())
        @test repr(ConjugateDescentCoefficient()) == "ConjugateDescentCoefficient()"
        @test repr(DaiYuanCoefficient()) == "DaiYuanCoefficient($pt)"
        @test repr(HagerZhangCoefficient()) == "HagerZhangCoefficient($pt)"
        @test repr(HestenesStiefelCoefficient()) == "HestenesStiefelCoefficient($pt)"
        @test repr(PolakRibiereCoefficient()) == "PolakRibiereCoefficient($pt)"
        cgbr = ConjugateGradientBealeRestart(ConjugateDescentCoefficient())
        s1 = "ConjugateGradientBealeRestart(ConjugateDescentCoefficient(), 0.2; vector_transport_method=$pt)"
        @test repr(cgbr) == s1
        @test repr(LiuStoreyCoefficient()) == "LiuStoreyCoefficient($pt)"
    end
end
