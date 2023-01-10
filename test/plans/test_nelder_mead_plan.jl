using Manifolds, Manopt, Test

@testset "Nelder Mead State" begin
    M = Euclidean(2)
    o = NelderMeadState(M)
    o2 = NelderMeadState(M, o.population)
    @test o.p == o2.p
    @test o.population == o2.population
    @test get_state(o) == o
    p = [1.0, 1.0]
    set_iterate!(o, M, p)
    @test get_iterate(o) == p
end
