using Manopt, Manifolds, ManifoldsBase, Test

@testset "test deprecated definitions still work" begin
    # after breaking releases this is usually empty.
    @test Manopt.AbstractManifoldGradientObjective === AbstractManifoldFirstOrderObjective
end
