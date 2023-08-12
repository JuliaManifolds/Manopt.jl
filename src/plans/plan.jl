"""
    status_summary(e)

Return a string reporting about the current status of `e`,
where `e` is a type from Manopt, e.g. an [`AbstractManoptSolverState`](@ref)s.

This method is similar to `show` but just returns a string.
It might also be more verbose in explaining, or hide internal information.
"""
status_summary(e) = "$(e)"

"""
    set_manopt_parameter!(f, element::Symbol , args...)

For any `f` and a `Symbol` `e` we dispatch on its value so by default, to
set some `args...` in `f` or one of uts sub elements.
"""
function set_manopt_parameter!(f, e::Symbol, args...)
    return set_manopt_parameter!(f, Val(e), args...)
end
function set_manopt_parameter!(f, ::Val, args...)
    return f
end

include("objective.jl")
include("problem.jl")
include("solver_state.jl")

include("debug.jl")
include("record.jl")

include("stopping_criterion.jl")
include("stepsize.jl")

include("cost_plan.jl")
include("gradient_plan.jl")
include("alternating_gradient_plan.jl")
include("constrained_plan.jl")
include("augmented_lagrangian_plan.jl")
include("conjugate_gradient_plan.jl")
include("exact_penalty_method_plan.jl")
include("frank_wolfe_plan.jl")
include("quasi_newton_plan.jl")
include("hessian_plan.jl")
include("proximal_plan.jl")
include("subgradient_plan.jl")
include("nonlinear_least_squares_plan.jl")
include("difference_of_convex_plan.jl")

include("primal_dual_plan.jl")
include("higher_order_primal_dual_plan.jl")

include("stochastic_gradient_plan.jl")

include("subsolver_plan.jl")

include("cache.jl")
include("count.jl")
