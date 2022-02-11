using Manifolds, Manopt, ManifoldsBase, Test

@testset "Plan Tests         " begin
    include("plans/test_options.jl")
    include("plans/test_debug.jl")
    include("plans/test_nelder_mead_plan.jl")
    include("plans/test_nonmutating.jl")
    include("plans/test_gradient_plan.jl")
    include("plans/test_hessian_plan.jl")
    include("plans/test_primal_dual_plan.jl")
    include("plans/test_record.jl")
    include("plans/test_stopping_criteria.jl")
end
@testset "Function Tests     " begin
    include("functions/test_adjoint_differentials.jl")
    include("functions/test_bezier.jl")
    include("functions/test_differentials.jl")
    include("functions/test_costs.jl")
    include("functions/test_gradients.jl")
    include("functions/test_proximal_maps.jl")
    include("functions/test_manifold.jl")
end
@testset "Helper & Data Tests" begin
    include("helpers/test_error_measures.jl")
    include("helpers/test_data.jl")
end
@testset "Solver Tests       " begin
    include("solvers/test_alternating_gradient.jl")
    include("solvers/test_ChambollePock.jl")
    include("solvers/test_conjugate_gradient.jl")
    include("solvers/test_Douglas_Rachford.jl")
    include("solvers/test_cyclic_proximal_point.jl")
    include("solvers/test_gradient_descent.jl")
    include("solvers/test_Nelder_Mead.jl")
    include("solvers/test_quasi_Newton.jl")
    include("solvers/test_trust_regions.jl")
    include("solvers/test_particle_swarm.jl")
    include("solvers/test_stochastic_gradient_descent.jl")
    include("solvers/test_subgradient_method.jl")
end
