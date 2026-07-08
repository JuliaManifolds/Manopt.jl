const _MANOPT_DEFAULT_CALLBACKS = [:BeforeInit, :BeforeStep, :BeforeStop, :Init, :Step, :Stop]
const _MANOPT_EMPTY_CALLBACK = (problem, state, iteration) -> nothing
const _MANOPT_EMPTY_ANY_CALLBACK = (symbol::Symbol, problem, state, iteration) -> nothing

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
    cb = get_callbacks(state)
    cbs = get(cb, name, _MANOPT_EMPTY_CALLBACK)
    cbs(problem, state, iteration)
    cba = get(cb, :Any, _MANOPT_EMPTY_ANY_CALLBACK)
    cba(name, problem, state, iteration)
    return nothing
end

"""
    get_callbacks(state::AbstractManoptSolverState)

Access the callbacks dictionary of the [`AbstractManoptSolverState`](@ref) `state`.
"""
get_callbacks(state::AbstractManoptSolverState) = _get_callbacks(state, dispatch_state_decorator(s))
function _get_callbacks(state::AbstractManoptSolverState, ::Val{false})
    @warn """
        This is a safety fallback! Upon initialization/setup, reaching this means your callback(s)
        are not stored in the state.
        Reaching this during a solver run, means your callbacks will not be called
    """
    # Fallback: No callbacks, so return an empty Dictionary
    return Dict{Symbol, Any}()
end
_get_callbacks(state::AbstractManoptSolverState, ::Val{true}) = get_iterate(state.state)

"""
    possible_callbacks(state_type::Type{S}) where {S<:AbstractManoptSolverState})

For a solver of type `S` return the callbacks actually available by returning a vector
`Symbol`s that can be used.
"""
function possible_callbacks(::Type{S}) where {S <: AbstractManoptSolverState}
    return _MANOPT_DEFAULT_CALLBACKS
end
