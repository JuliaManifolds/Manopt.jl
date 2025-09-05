"""
    AbstractSubProblemSolverState <: AbstractManoptSolverState

An abstract type for solvers that involve a subsolver.
"""
abstract type AbstractSubProblemSolverState <: AbstractManoptSolverState end

"""
    AbstractManifoldSubObjective{O<:AbstractManifoldObjective} <: AbstractManifoldObjective

An abstract type for objectives of sub problems within a solver but still store the
original objective internally to generate generic objectives for sub solvers.
"""
abstract type AbstractManifoldSubObjective{
    E <: AbstractEvaluationType, O <: AbstractManifoldObjective,
} <: AbstractManifoldObjective{E} end

function get_gradient_function(cgo::AbstractManifoldSubObjective)
    return (M, p) -> get_gradient(M, cgo, p)
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
function get_objective_cost(
        M::AbstractManifold, amso::AbstractManifoldSubObjective{E, O}, p
    ) where {E, O <: AbstractManifoldCostObjective}
    return get_cost(M, get_objective(amso), p)
end

@doc """
    X = get_objective_gradient(M, amso::AbstractManifoldSubObjective, p)
    get_objective_gradient!(M, X, amso::AbstractManifoldSubObjective, p)

Evaluate the gradient of the (original) objective stored within the sub objective `amso`.
"""
function get_objective_gradient(
        M::AbstractManifold, amso::AbstractManifoldSubObjective{E, O}, p
    ) where {E, O <: AbstractManifoldObjective{E}}
    return get_gradient(M, get_objective(amso), p)
end
function get_objective_gradient!(
        M::AbstractManifold, X, amso::AbstractManifoldSubObjective{E, O}, p
    ) where {E, O <: AbstractManifoldObjective{E}}
    return get_gradient!(M, X, get_objective(amso), p)
end

@doc """
    Y = get_objective_Hessian(M, amso::AbstractManifoldSubObjective, p, X)
    get_objective_Hessian!(M, Y, amso::AbstractManifoldSubObjective, p, X)

Evaluate the Hessian of the (original) objective stored within the sub objective `amso`.
"""
function get_objective_hessian(
        M::AbstractManifold, amso::AbstractManifoldSubObjective{E, O}, p, X
    ) where {E, O <: AbstractManifoldObjective{E}}
    return get_hessian(M, get_objective(amso), p, X)
end
function get_objective_hessian!(
        M::AbstractManifold, Y, amso::AbstractManifoldSubObjective{E, O}, p, X
    ) where {E, O <: AbstractManifoldObjective{E}}
    get_hessian!(M, Y, get_objective(amso), p, X)
    return Y
end

@doc """
    Y = get_objective_preconditioner(M, amso::AbstractManifoldSubObjective, p, X)
    get_objective_preconditioner(M, Y, amso::AbstractManifoldSubObjective, p, X)

Evaluate the Hessian of the (original) objective stored within the sub objective `amso`.
"""
function get_objective_preconditioner(
        M::AbstractManifold, amso::AbstractManifoldSubObjective{E, O}, p, X
    ) where {E, O <: AbstractManifoldHessianObjective{E}}
    return get_preconditioner(M, get_objective(amso), p, X)
end
function get_objective_preconditioner!(
        M::AbstractManifold, Y, amso::AbstractManifoldSubObjective{E, O}, p, X
    ) where {E, O <: AbstractManifoldHessianObjective{E}}
    return get_preconditioner!(M, Y, get_objective(amso), p, X)
end

@doc """
    get_sub_problem(ams::AbstractSubProblemSolverState)

Access the sub problem of a solver state that involves a sub optimisation task.
By default this returns `ams.sub_problem`.
"""
get_sub_problem(ams::AbstractSubProblemSolverState) = ams.sub_problem

@doc """
    get_sub_state(ams::AbstractSubProblemSolverState)

Access the sub state of a solver state that involves a sub optimisation task.
By default this returns `ams.sub_state`.
"""
get_sub_state(ams::AbstractSubProblemSolverState) = ams.sub_state

"""
    set_parameter!(ams::AbstractManoptSolverState, element::Symbol, args...)

Set a certain field or semantic element from the [`AbstractManoptSolverState`](@ref) `ams` to `value`.
This function passes to `Val(element)` and specific setters should dispatch on `Val{element}`.

By default, this function just does nothing.
"""
function set_parameter!(ams::AbstractManoptSolverState, e::Symbol, args...)
    return set_parameter!(ams, Val(e), args...)
end
# Default: do nothing
function set_parameter!(ams::AbstractManoptSolverState, ::Val, args...)
    return ams
end

"""
    set_parameter!(ams::DebugSolverState, ::Val{:SubProblem}, args...)

Set certain values specified by `args...` to the sub problem.
"""
function set_parameter!(ams::AbstractSubProblemSolverState, ::Val{:SubProblem}, args...)
    set_parameter!(get_sub_problem(ams), args...)
    return ams
end
"""
    set_parameter!(ams::DebugSolverState, ::Val{:SubState}, args...)

Set certain values specified by `args...` to the sub state.
"""
function set_parameter!(ams::AbstractSubProblemSolverState, ::Val{:SubState}, args...)
    set_parameter!(get_sub_state(ams), args...)
    return ams
end
