"""
    has_sub_problem(::Type{<:AbstractManoptSolverState})

Return whether a solver state type stores a sub task as the pair `sub_problem` and `sub_state`,
see [`get_sub_problem`](@ref) and [`get_sub_state`](@ref).
This is `false` by default; a state that stores such a pair declares it as
`has_sub_problem(::Type{<:MyState}) = true`.
"""
has_sub_problem(::Type{<:AbstractManoptSolverState}) = false

@doc """
    get_sub_problem(ams::AbstractManoptSolverState)

Access the sub problem of a solver state that involves a sub optimization task,
see [`has_sub_problem`](@ref). By default this returns `ams.sub_problem`.
"""
get_sub_problem(ams::AbstractManoptSolverState) = _get_sub_problem(ams, Val(has_sub_problem(typeof(ams))))
_get_sub_problem(ams::AbstractManoptSolverState, ::Val{true}) = ams.sub_problem
function _get_sub_problem(ams::AbstractManoptSolverState, ::Val{false})
    return error("The state $(typeof(ams)) does not store a sub problem.")
end

@doc """
    get_sub_state(ams::AbstractManoptSolverState)

Access the sub state of a solver state that involves a sub optimization task,
see [`has_sub_problem`](@ref). By default this returns `ams.sub_state`.
"""
get_sub_state(ams::AbstractManoptSolverState) = _get_sub_state(ams, Val(has_sub_problem(typeof(ams))))
_get_sub_state(ams::AbstractManoptSolverState, ::Val{true}) = ams.sub_state
function _get_sub_state(ams::AbstractManoptSolverState, ::Val{false})
    return error("The state $(typeof(ams)) does not store a sub state.")
end

# The part of the fallback `set_parameter!` for states that store a sub task (see `has_sub_problem`):
# pass `:SubProblem` on to the sub problem and `:SubState` to the sub state, do nothing otherwise.
function _set_sub_parameter!(ams::AbstractManoptSolverState, ::Val{true}, ::Val{:SubProblem}, args...)
    set_parameter!(get_sub_problem(ams), args...)
    return ams
end
function _set_sub_parameter!(ams::AbstractManoptSolverState, ::Val{true}, ::Val{:SubState}, args...)
    set_parameter!(get_sub_state(ams), args...)
    return ams
end
_set_sub_parameter!(ams::AbstractManoptSolverState, ::Val, ::Val, args...) = ams
