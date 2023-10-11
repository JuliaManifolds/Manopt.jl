"""
    AbstractSubProblemSolverState <: AbstractManoptSolverState

An abstract type for problems that involve a subsolver
"""
abstract type AbstractSubProblemSolverState <: AbstractManoptSolverState end

"""
   AbstractManifoldSubObjective{O<:AbstractManifoldObjective} <: AbstractManifoldObjective

An abstract type for objectives of sub problems within a solver but still store the
original objective internally to generate generic objectives for sub solvers.
"""
abstract type AbstractManifoldSubObjective{O<:AbstractManifoldObjective} <:
              AbstractManifoldObjective end

@doc raw"""
    get_objective(amso::AbstractManifoldSubObjective)

Return the (original) objective stored within the sub obective.
"""
get_objective(amso::AbstractManifoldSubObjective)

@doc raw"""
    get_objective_cost(M, amso::AbstractManifoldSubObjective, p)

Evaluate the cost of the (original) objective stored within the subobjective.
"""
function get_objective_cost(
    M::AbstractManifold, amso::AbstractManifoldSubObjective{O}, p
) where {O<:AbstractManifoldCostObjective}
    return get_cost(M, get_objective(amso), p)
end

@doc raw"""
    X = get_objective_gradient(M, amso::AbstractManifoldSubObjective, p)
    get_objective_gradient!(M, X, amso::AbstractManifoldSubObjective, p)

Evaluate the gradient of the (original) objective stored within the subobjective `amso`.
"""
function get_objective_gadient(
    M::AbstractManifold, amso::AbstractManifoldSubObjective{O}, p
) where {O<:AbstractManifoldGradientObjective}
    return get_gradient(M, get_objective(amso), p)
end
function get_objective_gadient!(
    M::AbstractManifold, X, amso::AbstractManifoldSubObjective{O}, p
) where {O<:AbstractManifoldGradientObjective}
    return get_gradient!(M, X, get_objective(amso), p)
end

@doc raw"""
    Y = get_objective_Hessian(M, amso::AbstractManifoldSubObjective, p, X)
    get_objective_Hessian!(M, Y, amso::AbstractManifoldSubObjective, p, X)

Evaluate the Hessian of the (original) objective stored within the subobjective `amso`.
"""
function get_objective_Hessian(
    M::AbstractManifold, amso::AbstractManifoldSubObjective{O}, p, X
) where {O<:AbstractManifoldHessianObjective}
    return get_Hessian(M, get_objective(amso), p, X)
end
function get_objective_gadient!(
    M::AbstractManifold, Y, amso::AbstractManifoldSubObjective{O}, p, X
) where {O<:AbstractManifoldHessianObjective}
    return get_Hessian!(M, Y, get_objective(amso), p, X)
end

@doc raw"""
    get_sub_problem(ams::AbstractSubProblemSolverState)

Access the sub problem of a solver state that involves a sub optimisation task.
By default this returns `ams.sub_problem`.
"""
get_sub_problem(ams::AbstractSubProblemSolverState) = ams.sub_problem

@doc raw"""
    get_sub_state(ams::AbstractSubProblemSolverState)

Access the sub state of a solver state that involves a sub optimisation task.
By default this returns `ams.sub_state`.
"""
get_sub_state(ams::AbstractSubProblemSolverState) = ams.sub_state

"""
    set_manopt_parameter!(ams::AbstractManoptSolverState, element::Symbol, args...)

Set a certain field or semantic element from the [`AbstractManoptSolverState`](@ref) `ams` to `value`.
This function passes to `Val(element)` and specific setters should dispatch on `Val{element}`.

By default, this function just does nothing.
"""
function set_manopt_parameter!(ams::AbstractManoptSolverState, e::Symbol, args...)
    return set_manopt_parameter!(ams, Val(e), args...)
end
# Default: Do nothing
function set_manopt_parameter!(ams::AbstractManoptSolverState, ::Val, args...)
    return ams
end
"""
    set_manopt_parameter!(ams::DebugSolverState, ::Val{:Debug}, args...)

Set certain values specified by `args...` into the elements of the `debugDictionary`
"""
function set_manopt_parameter!(dss::DebugSolverState, ::Val{:Debug}, args...)
    for d in values(dss.debugDictionary)
        set_manopt_parameter!(d, args...)
    end
    return dss
end

"""
    set_manopt_parameter!(ams::DebugSolverState, ::Val{:SubProblem}, args...)

Set certain values specified by `args...` to the sub problem.
"""
function set_manopt_parameter!(
    ams::AbstractSubProblemSolverState, ::Val{:SubProblem}, args...
)
    set_manopt_parameter!(get_sub_problem(ams), args...)
    return ams
end
"""
    set_manopt_parameter!(ams::DebugSolverState, ::Val{:SubState}, args...)

Set certain values specified by `args...` to the sub state.
"""
function set_manopt_parameter!(
    ams::AbstractSubProblemSolverState, ::Val{:SubState}, args...
)
    set_manopt_parameter!(get_sub_state(ams), args...)
    return ams
end

"""
    get_manopt_parameter(ams::AbstractManoptSolverState, element::Symbol, args...)

Obtain a certain field or semantic element from the [`AbstractManoptSolverState`](@ref) `ams`.
This function passes to `Val(element)` and specific setters should dispatch on `Val{element}`.
"""
function get_manopt_parameter(ams::AbstractManoptSolverState, e::Symbol, args...)
    return get_manopt_parameter(ams, Val(e), args...)
end
