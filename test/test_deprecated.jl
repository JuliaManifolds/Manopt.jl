using Manopt, Manifolds, ManifoldsBase, Test

@testset "test deprecated definitions still work" begin
    # after breaking releases this is usually empty.
    @test Manopt.AbstractManifoldGradientObjective === AbstractManifoldFirstOrderObjective
    # the positional start point of three states
    M = Euclidean(2)
    p = [1.0, 2.0]
    @test ProjectedGradientMethodState(M, p).p == p
    @test MeshAdaptiveDirectSearchState(M, p).p == p
    @test CoordinatesNormalSystemState(M, p) isa CoordinatesNormalSystemState
end
