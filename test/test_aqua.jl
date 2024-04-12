using Aqua, Manopt, Test

@testset "Aqua.jl" begin
    Aqua.test_all(
        Manopt;
        ambiguities=(
            exclude=[#For now exclude some high-level functions, since in their
                # different call schemes some ambiguities appear
                # We should carefully check these
                Manopt.truncated_conjugate_gradient_descent,
                Manopt.difference_of_convex_proximal_point,
                Manopt.difference_of_convex_proximal_point,
                Manopt.particle_swarm,
                Manopt.stochastic_gradient_descent,
                Manopt.truncated_conjugate_gradient_descent!,
                Manopt.get_last_stepsize, #Maybe redesign?
            ],
            broken=false,
        ),
    )
end
