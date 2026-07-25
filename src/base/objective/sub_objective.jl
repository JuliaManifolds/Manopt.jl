"""
    AbstractManifoldSubObjective{O<:AbstractManifoldObjective} <: AbstractManifoldObjective

An abstract type for objectives of sub problems within a solver but still store the
original objective internally to generate generic objectives for sub solvers.
"""
abstract type AbstractManifoldSubObjective{O <: AbstractManifoldObjective} <: AbstractManifoldObjective end

function get_gradient_function(amso::AbstractManifoldSubObjective)
    return (M, p) -> get_gradient(M, get_objective(amso), p)
end

@doc """
    get_objective(amso::AbstractManifoldSubObjective)

Return the (original) objective stored the sub objective is build on.
"""
get_objective(amso::AbstractManifoldSubObjective)

@doc """
    get_objective_cost(M, amso::AbstractManifoldSubObjective, p)

Evaluate the cost of the (original) objective stored within the sub objective.
"""
function get_objective_cost(M::AbstractManifold, amso::AbstractManifoldSubObjective, p)
    return get_cost(M, get_objective(amso), p)
end

@doc """
    X = get_objective_gradient(M, amso::AbstractManifoldSubObjective, p)
    get_objective_gradient!(M, X, amso::AbstractManifoldSubObjective, p)

Evaluate the gradient of the (original) objective stored within the sub objective `amso`.
"""
function get_objective_gradient(M::AbstractManifold, amso::AbstractManifoldSubObjective, p)
    return get_gradient(M, get_objective(amso), p)
end
function get_objective_gradient!(M::AbstractManifold, X, amso::AbstractManifoldSubObjective, p)
    return get_gradient!(M, X, get_objective(amso), p)
end

@doc """
    Y = get_objective_Hessian(M, amso::AbstractManifoldSubObjective, p, X)
    get_objective_Hessian!(M, Y, amso::AbstractManifoldSubObjective, p, X)

Evaluate the Hessian of the (original) objective stored within the sub objective `amso`.
"""
function get_objective_hessian(M::AbstractManifold, amso::AbstractManifoldSubObjective, p, X)
    return get_hessian(M, get_objective(amso), p, X)
end
function get_objective_hessian!(M::AbstractManifold, Y, amso::AbstractManifoldSubObjective, p, X)
    return get_hessian!(M, Y, get_objective(amso), p, X)
end

@doc """
    Y = get_objective_preconditioner(M, amso::AbstractManifoldSubObjective, p, X)
    get_objective_preconditioner(M, Y, amso::AbstractManifoldSubObjective, p, X)

Evaluate the Hessian of the (original) objective stored within the sub objective `amso`.
"""
function get_objective_preconditioner(M::AbstractManifold, amso::AbstractManifoldSubObjective, p, X)
    return get_preconditioner(M, get_objective(amso), p, X)
end
function get_objective_preconditioner!(M::AbstractManifold, Y, amso::AbstractManifoldSubObjective, p, X)
    return get_preconditioner!(M, Y, get_objective(amso), p, X)
end
