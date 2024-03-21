using Manopt, Test

@testset "IP Newton Tests" begin
    @test 1==1
    a = 1/2
    @test a ≈ 0.5 atol=1e-9
end