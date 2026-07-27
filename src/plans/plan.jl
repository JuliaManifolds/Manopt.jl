include("stopping_criterion.jl")

include("stepsize/initial_guess.jl")
include("stepsize/stepsize_message.jl")
include("stepsize/linesearch.jl")
include("stepsize/stepsize.jl")

# Generic plans I: based on objective structure
include("hessian_plan.jl")

# Linear systems
include("conjugate_residual_plan.jl")

# Specific solver plans
include("alternating_gradient_plan.jl")
include("augmented_lagrangian_plan.jl")
include("conjugate_gradient_plan.jl")
include("exact_penalty_method_plan.jl")
include("interior_point_Newton_plan.jl")
include("quasi_newton_plan.jl")
include("difference_of_convex_plan.jl")

include("stochastic_gradient_plan.jl")

include("box_plan.jl")

include("embedded_objective.jl")
include("scaled_objective.jl")

include("cache.jl")
include("count.jl")
