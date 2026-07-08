const _MANOPT_DEFAULT_CALLBACKS = [:BeforeInit, :BeforeStep, :BeforeStop, :Init, :Step, :Stop]
const _MANOPT_EMPTY_CALLBACK = (problem, state, iteration) -> nothing
const _MANOPT_EMPTY_ANY_CALLBACK = (symbol::Symbol, problem, state, iteration) -> nothing

"""
    available_callbacks(state::AbstractManoptSolverState)

For a given state, indicate, which callbacks are in use, i.e. stored and also called from this solver.

See also: [`possible_callbacks`](@ref).
"""
function available_callbacks(state::AbstractManoptSolverState)
    ac = intersect(keys(get_callbacks(state)), possible_callbacks(typeof(state)))
    return Symbol[ac...]
end

"""
    callback(name::Symbol, problem::AbstractManoptProblem, state::AbstractManoptState, iteration::Int)

Perform a callback.

This function performs a call to both possible approaches
* if a callback exists in the dictionary under the symbol `name` it is called with the parameters `problem, state, iteration`
* if a callback exists in the dictionary that shall be called `:Any` time, this one is called with `name, problem, state, iteration`
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

For a solver of type `S` return the callbacks actually uses in practive, i.e. the ones
which from within the solver are actually called. This function returns a vector
`Symbol`s that can be used.
"""
function possible_callbacks(::Type{S}) where {S <: AbstractManoptSolverState}
    return _MANOPT_DEFAULT_CALLBACKS
end

"""
    process_callbacks_arg(callbacks)

Given an array `callbacks` a user has passed to a solver, this helper function processes the
array in the following way

* a pair `:Hook => fct` is kept as is, where the `fct`` can also be some callable structure
* a single element `function` is turned into `:Any => fct` allowing the case a single callback
  to be specified just with the function
* a pair `[Hook1, Hook2] => fct` is a shortcut for an array of pairs and split into these here.

The result is then wrapped into a dictionary. Be aware that from an array of pairs this function
reduces, the dictionary “takes” the last `:Hook` pair as the entry in the dictionary. This
function does not check for duplicates.
"""
function process_callbacks_arg(callbacks::Array)
    c = Pair{Symbol, Any}[]
    for cb in callbacks
        if cb isa Pair
            if cb[1] isa Symbol
                push!(c, cb)
            elseif cb[1] isa AbstractVector{Symbol}
                for s in cb[1]
                    push!(c, s => cb[2])
                end
            else
                error("Unkown key $(cb[1])")
            end
        else
            push!(c, :Any => cb)
        end
    end
    return Dict(c...)
end
