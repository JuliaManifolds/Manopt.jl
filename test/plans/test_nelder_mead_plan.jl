using Manifolds, Manopt, Test

@testset "Nelder Mead State" begin
    o = NelderMeadState(Euclidean(2))
    o2 = NelderMeadState(Euclidean(2), o.population)
    @test o.x == o2.x
    @test o.population == o2.population
    @test get_state(o) == o
    p = [1.0, 1.0]
    set_iterate!(o, p)
    @test get_iterate(o) == p
end
