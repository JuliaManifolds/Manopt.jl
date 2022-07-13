using Manifolds, Manopt, Test, ManifoldsBase

@testset "Test Costs" begin
    M = Sphere(2)
    N = PowerManifold(M, NestedPowerRepresentation(), 3, 3)
    f = repeat([[1.0, 0.0, 0.0]], 3, 3)
    x = repeat([[1.0, 0.0, 0.0]], 3, 3)
    x[2, 1] = [0.0, 1.0, 0.0]

    @test costIntrICTV12(N, f, f, f, 0.0, 0.0) == 0.0
    @test costIntrICTV12(N, f, x, x, 0.0, 0.0) == 1 / 2 * distance(N, x, f)^2
    @test costIntrICTV12(N, x, x, x, 2.0, 0.5) ≈ costTV2(N, x) + costTV(N, x)
    @test costIntrICTV12(N, x, x, x, 1.0, 0.0) ≈ costTV2(N, x)
    @test costIntrICTV12(N, x, x, x, 1.0, 1.0) ≈ costTV(N, x)
    @test costL2TV2(N, f, 1.0, x) == 1 / 2 * distance(N, f, x)^2 + 1.0 * costTV2(N, x)
    #
    @test costL2TV(N, f, 1.0, f) ≈ 0.0
    @test costL2TV(N, x, 1.0, x) ≈ 3 * π / 2
    @test costL2TVTV2(N, f, 0.0, 1.0, x) ≈ 1 / 2 * distance(N, x, f)^2 + costTV2(N, x)
    @test costL2TVTV2(N, f, 1.0, 1.0, x) ≈
        1 / 2 * distance(N, x, f)^2 + costTV(N, x) + costTV2(N, x)

    @test costTV2(M, Tuple(x[1:3, 1])) ≈ π / 2
    @test costTV(N, x, 1, 2) ≈ sqrt(5 / 4) * π
    @test sum(costTV2(N, x, 1, false)) == costTV2(N, x)
    @test costTV2(N, f, 2) == 0
end
