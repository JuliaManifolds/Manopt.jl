"""
    AbstractSubProblemSolverState <: AbstractManoptSolverState

An abstract type for problems that involve a subsolver
"""
abstract type AbstractSubProblemSolverState <: AbstractManoptSolverState end

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

Set a certain field/element from the [`AbstractManoptSolverState`](@ref) `ams` to `value.
This function dispatches on `Val(element)`.
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
function set_manopt_parameter!(amp::AbstractManoptProblem, ev::Val, fv::Val, v)
    set_manopt_parameter!(get_objective(amp), ev, fv, v)
    return amp
end
function set_manopt_parameter!(amp::AbstractManoptProblem, ::Val{:Manifold}, fv::Val, v)
    set_manopt_parameter!(get_manifold(amp), fv, v)
    return amp
end
function set_manopt_parameter!(amp::AbstractManoptProblem, ::Val{:Objective}, fv::Val, v)
    set_manopt_parameter!(get_objective(amp), fv, v)
    return amp
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
