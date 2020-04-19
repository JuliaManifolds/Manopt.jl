using Manifolds, Manopt, Test

@testset "Nelder Mead Options" begin
    o = NelderMeadOptions(Euclidean(2))
    o2 = NelderMeadOptions(o.population)
    @test o.x == o2.x
    @test o.population == o2.population
    @test get_options(o) == o
end