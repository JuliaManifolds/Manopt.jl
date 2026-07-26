include("stopping_criterion.jl")

include("stepsize/initial_guess.jl")
include("stepsize/stepsize_message.jl")
include("stepsize/linesearch.jl")
include("stepsize/stepsize.jl")

# Generic plans I: based on objective structure
include("hessian_plan.jl")

# Linear systems
include("conjugate_residual_plan.jl")
# Robutsifiers
include("robustifiers.jl")

# Generic plans II: based on subsolvers
include("trust_regions_plan.jl")

# Specific solver plans
include("adaptive_regularization_with_cubics_plan.jl")
include("alternating_gradient_plan.jl")
include("augmented_lagrangian_plan.jl")
include("conjugate_gradient_plan.jl")
include("exact_penalty_method_plan.jl")
include("interior_point_Newton_plan.jl")
include("quasi_newton_plan.jl")
include("nonlinear_least_squares/linear_surrogate_plan.jl")
include("nonlinear_least_squares/nls_objective.jl")
include("nonlinear_least_squares/nls_general_plan.jl")
include("nonlinear_least_squares/nls_in_coordinates_plan.jl")
include("nonlinear_least_squares/box_nls_plan.jl")
include("difference_of_convex_plan.jl")

include("primal_dual_plan.jl")
include("higher_order_primal_dual_plan.jl")

include("stochastic_gradient_plan.jl")

include("box_plan.jl")

include("embedded_objective.jl")
include("scaled_objective.jl")

include("cache.jl")
include("count.jl")
