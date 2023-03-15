using Manopt, Manifolds, ManifoldsBase, Test

@testset "test deprecated definitions still work" begin
    @test_logs (:warn,) DebugChange(; invretr=LogarithmicInverseRetraction())
    @test_logs (:warn,) DebugChange(; manifold=ManifoldsBase.DefaultManifold())
    @test_logs (:warn,) RecordChange(; manifold=ManifoldsBase.DefaultManifold())
    @test_logs (:warn,) StopWhenChangeLess(1e-9; manifold=Euclidean())
end
