using Manopt, Manifolds, ManifoldsBase, Test

@testset "test deprecated definitions still work" begin
    @testset "outdated kwargs in constructors" begin
        @test_logs (:warn,) DebugChange(; invretr=LogarithmicInverseRetraction())
        @test_logs (:warn,) DebugChange(; manifold=ManifoldsBase.DefaultManifold())
        @test_logs (:warn,) RecordChange(; manifold=ManifoldsBase.DefaultManifold())
        @test_logs (:warn,) StopWhenChangeLess(1e-9; manifold=Euclidean())
    end

    @testset "Outdated constrained accessors" begin
        M = ManifoldsBase.DefaultManifold(3)
        f(::ManifoldsBase.DefaultManifold, p) = norm(p)^2
        grad_f(M, p) = 2 * p
        g(M, p) = [p[1] - 1, -p[2] - 1]
        grad_g(M, p) = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        h(M, p) = [2 * p[3] - 1]
        grad_h(M, p) = [[0.0, 0.0, 2.0]]
        co = ConstrainedManifoldObjective(
            ManifoldGradientObjective(f, grad_f);
            equality_constraints=VectorGradientFunction(g, grad_g, 2),
            inequality_constraints=VectorGradientFunction(h, grad_h, 1),
        )
        dmp = DefaultManoptProblem(M, co)
        p = [1.0, 2.0, 3.0]
        @test_logs (:warn,) get_constraints(dmp, p)
        @test_logs (:warn,) get_constraints(M, co, p)
    end
end
