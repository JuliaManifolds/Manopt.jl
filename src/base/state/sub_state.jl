"""
    AbstractSubProblemSolverState <: AbstractManoptSolverState

An abstract type for solvers that involve a subsolver.
"""
abstract type AbstractSubProblemSolverState <: AbstractManoptSolverState end

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
    set_parameter!(ams::DebugSolverState, :SubProblem, args...)

Set certain values specified by `args...` to the sub problem.
"""
function set_parameter!(ams::AbstractSubProblemSolverState, ::Val{:SubProblem}, args...)
    return set_parameter!(get_sub_problem(ams), args...)
end
"""
    set_parameter!(ams::DebugSolverState, :SubState, args...)

Set certain values specified by `args...` to the sub state.
"""
function set_parameter!(ams::AbstractSubProblemSolverState, ::Val{:SubState}, args...)
    return set_parameter!(get_sub_state(ams), args...)
end
