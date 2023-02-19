using Manopt, ManifoldsBase, Test

@testset "test deprecated definitions still work" begin
    @test_logs (:warn,) DebugChange(; invretr=LogarithmicInverseRetraction())
    @test_logs (:warn,) DebugChange(; manifold=ManifoldsBase.DefaultManifold())
end
