s = joinpath(@__DIR__, "ManoptTestSuite.jl")
!(s in LOAD_PATH) && (push!(LOAD_PATH, s))

using Manifolds, ManifoldsBase, Manopt, ManoptTestSuite, Test

@testset "Manopt.jl" begin
    @testset "Plan Tests         " begin
        include("plans/test_objective.jl")
        include("plans/test_problem.jl")
        include("plans/test_state.jl")
        include("plans/test_conjugate_gradient_plan.jl")
        include("plans/test_counts.jl")
        include("plans/test_debug.jl")
        include("plans/test_difference_of_convex_plan.jl")
        include("plans/test_embedded.jl")
        include("plans/test_cache.jl")
        include("plans/test_conjugate_residual_plan.jl")
        include("plans/test_interior_point_newton_plan.jl")
        include("plans/test_mesh_adaptive_plan.jl")
        include("plans/test_nelder_mead_plan.jl")
        include("plans/test_nonlinear_least_squares_plan.jl")
        include("plans/test_gradient_plan.jl")
        include("plans/test_constrained_plan.jl")
        include("plans/test_constrained_set_plan.jl")
        include("plans/test_hessian_plan.jl")
        include("plans/test_parameters.jl")
        include("plans/test_primal_dual_plan.jl")
        include("plans/test_proximal_plan.jl")
        include("plans/test_higher_order_primal_dual_plan.jl")
        include("plans/test_defaults_factory.jl")
        include("plans/test_record.jl")
        include("plans/test_scaled_objective.jl")
        include("plans/test_stepsize.jl")
        include("plans/test_stochastic_gradient_plan.jl")
        include("plans/test_stopping_criteria.jl")
        include("plans/test_storage.jl")
        include("plans/test_subgradient_plan.jl")
        include("plans/test_vectorial_plan.jl")
    end
    @testset "Helper Tests       " begin
        include("helpers/test_checks.jl")
        include("helpers/test_linesearches.jl")
        include("helpers/test_manifold_extra_functions.jl")
    end
    @testset "Solver Tests       " begin
        include("solvers/test_adaptive_regularization_with_cubics.jl")
        include("solvers/test_alternating_gradient.jl")
        include("solvers/test_augmented_lagrangian.jl")
        include("solvers/test_cma_es.jl")
        include("solvers/test_convex_bundle_method.jl")
        include("solvers/test_ChambollePock.jl")
        include("solvers/test_conjugate_gradient.jl")
        include("solvers/test_difference_of_convex.jl")
        include("solvers/test_Douglas_Rachford.jl")
        include("solvers/test_conjugate_residual.jl")
        include("solvers/test_cyclic_proximal_point.jl")
        include("solvers/test_exact_penalty.jl")
        include("solvers/test_Frank_Wolfe.jl")
        include("solvers/test_gradient_descent.jl")
        include("solvers/test_interior_point_Newton.jl")
        include("solvers/test_Levenberg_Marquardt.jl")
        include("solvers/test_mesh_adaptive_direct_search.jl")
        include("solvers/test_Nelder_Mead.jl")
        include("solvers/test_projected_gradient.jl")
        include("solvers/test_proximal_bundle_method.jl")
        include("solvers/test_proximal_gradient_method.jl")
        include("solvers/test_proximal_point.jl")
        include("solvers/test_quasi_Newton.jl")
        include("solvers/test_particle_swarm.jl")
        include("solvers/test_primal_dual_semismooth_Newton.jl")
        include("solvers/test_stochastic_gradient_descent.jl")
        include("solvers/test_subgradient_method.jl")
        include("solvers/test_truncated_cg.jl")
        include("solvers/test_trust_regions.jl")
    end
    include("MOI_wrapper.jl")
    include("test_aqua.jl")
    include("test_deprecated.jl")
end
