using Aqua, Manopt, Test

@testset "Aqua.jl" begin
    Aqua.test_all(Manopt; ambiguities=(exclude=[#For now exclude some high-level functions, since in their
    # different call schemes some ambiguities appear
    # We should carefully check these
    #Manopt.truncated_conjugate_gradient_descent, # ambiguity corresponds a problem with p and the Hessian and both being positional
    #Manopt.difference_of_convex_proximal_point, # should be fixed
    #Manopt.particle_swarm, # should be fixed
    #Manopt.stochastic_gradient_descent, # should be fixed
    #Manopt.truncated_conjugate_gradient_descent!, # will be fixed by removing deprecated methods
], broken=false))
end
