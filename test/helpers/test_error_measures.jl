@testset "Manopt.jl Error Measures" begin
    M = Sphere(2)
    N = PowerManifold(M, NestedPowerRepresentation(), 2)
    using Random: seed!
    seed!(42)
    d = Manifolds.uniform_distribution(M, [1.0, 0.0, 0.0])
    w = rand(d)
    x = rand(d)
    y = rand(d)
    z = rand(d)
    a = [w, x]
    b = [y, z]
    @test meanSquaredError(M, x, y) == distance(M, x, y)^2
    @test meanSquaredError(N, a, b) == 1 / 2 * (distance(M, w, y)^2 + distance(M, x, z)^2)
    @test meanAverageError(M, x, y) == distance(M, x, y)
    @test meanAverageError(N, a, b) == 1 / 2 * sum(distance.(Ref(M), a, b))
end
