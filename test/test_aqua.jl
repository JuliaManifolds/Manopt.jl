using Aqua, Manopt, Test

@testset "Aqua.jl" begin
    Aqua.test_all(Manopt; ambiguities = (broken = false,))
end
