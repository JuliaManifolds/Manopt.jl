using Aqua, Manopt, Test

@testset "Aqua.jl" begin
    Aqua.test_all(Manopt; ambiguities=(exclude=[
    #Manopt.truncated_conjugate_gradient_descent!, # will be fixed by removing deprecated methods
], broken=false))
end
