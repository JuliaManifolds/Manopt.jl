"""
    available_callbacks(state::AbstractManoptSolverState)

For a given state, indicate, which callbacks are stored.
"""
function available_callbacks(state::AbstractManoptSolverState)
    ac = intersect(keys(get_callbacks(state)), possible_callbacks(typeof(state)))
    return Symbol[ac...]
end

"""
    callback(name::Symbol, problem::AbstractManoptProblem, state::AbstractManoptState, iteration::Int)

Access the callback of `name` and call it with `problem, state, iteration`
"""
function callback(
        name::Symbol, problem::AbstractManoptProblem, state::AbstractManoptSolverState, iteration::Int
    )
    cb = get(get_callbacks(state), name, (problem, state, iteration) -> nothing)
    return cb(problem, state, iteration)
end

"""
    get_callbacks(state::AbstractManoptSolverState)

Access the callbacks dictionary of the [`AbstractManoptSolverState`](@ref) `state`.
"""
get_callbacks(state::AbstractManoptSolverState) = _get_callbacks(state, dispatch_state_decorator(s))
function _get_callbacks(state::AbstractManoptSolverState, ::Val{false})
    # Fallback: No callbacks, so return an empty Dictionary
    return Dict{Symbol, Function}()
end
_get_callbacks(state::AbstractManoptSolverState, ::Val{true}) = get_iterate(state.state)

"""
    possible_callbacks(state_type::Type{S}) where {S<:AbstractManoptSolverState})

For a solver of type `S` return the callbacks actually available by returning a vector
`Symbol`s that can be used.
"""
function possible_callbacks(::Type{S}) where {S <: AbstractManoptSolverState}
    return Symbol[]
end
