const _MANOPT_DEFAULT_CALLBACKS = [:Any, :BeforeInit, :BeforeStep, :BeforeStop, :Init, :Step, :Stop]
const _MANOPT_EMPTY_CALLBACK = (problem, state, iteration) -> nothing
const _MANOPT_EMPTY_ANY_CALLBACK = (symbol::Symbol, problem, state, iteration) -> nothing

"""
    active_callbacks(state::AbstractManoptSolverState)

For a given state, indicate, which callbacks are in use, i.e. stored and also called from this solver.

See also: [`provided_fallbacks`](@ref).
"""
function active_callbacks(state::AbstractManoptSolverState)
    dk = keys(get_callbacks(state))
    return Symbol[ intersect(dk, provided_fallbacks(typeof(state))) ... ]
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
get_callbacks(state::AbstractManoptSolverState) = _get_callbacks(state, dispatch_state_decorator(state))
function _get_callbacks(state::AbstractManoptSolverState, ::Val{false})
    @warn """
        This is a safety fallback! Upon initialization/setup, reaching this means your callback(s)
        are not stored in the state.
        Reaching this during a solver run, means your callbacks will not be called
    """
    # Fallback: No callbacks, so return an empty Dictionary
    return Dict{Symbol, Any}()
end
# For all decorators: Pass down
_get_callbacks(state::AbstractManoptSolverState, ::Val{true}) = get_callbacks(state.state)

"""
    provided_fallbacks(state_type::Type{S}) where {S<:AbstractManoptSolverState})

For a solver of type `S` return the callbacks the solver provides, i.e. all ones
that are called during the solver run.
This function returns a vector `Symbol`s representing the hooks. These can be kays in
a dictionary of callbacks.
"""
function provided_fallbacks(::Type{S}) where {S <: AbstractManoptSolverState}
    return _MANOPT_DEFAULT_CALLBACKS
end

"""
    process_callbacks_arg(callbacks::Array, statetype=missing)
    process_callbacks_arg(callbacks::Dict, statetype=missing) = callbacks

Given an array `callbacks` a user has passed to a solver, this helper function processes the
array in the following way

* a pair `:Hook => fct` is kept as is, where the `fct`` can also be some callable structure
* a single element `function` is turned into `:Any => fct` allowing the case a single callback
  to be specified just with the function
* a pair `[Hook1, Hook2] => fct` is a shortcut for an array of pairs and split into these here.

The result is then wrapped into a dictionary. Be aware that from an array of pairs this function
reduces, the dictionary “takes” the last `:Hook` pair as the entry in the dictionary. This
function does not check for duplicates.

This function keeps a dictionary unchanged. Hence they are only processed once
even if this function is applied multiple times.
"""
function process_callbacks_arg(callbacks::Array, statetype=missing)
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
    res = Dict{Symbol, Any}(c...)
    _warn_if_unused_callbacks(res, statetype)
    return res
end
"""
    process_callbacks_arg(callback)

Passing a single callback nadles this as if it is an array of length one with
this element in the [`process_callbacks_arg`](@ref process_callbacks_arg(::Array)) array case.
"""
process_callbacks_arg(callback, statetype=missing) = process_callbacks_arg([callback,], statetype)
#
function process_callbacks_arg(callbacks::Dict, statetype)
    _warn_if_unused_callbacks(callbacks, statetype)
    return callbacks
end
function _warn_if_unused_callbacks(callbacks::Dict, statetype=missing)
    if !ismissing(statetype)
        ck = keys(callbacks)
        pk = provided_fallbacks(statetype)
        uk = [setdiff(ck, pk)...]
        (length(uk) > 0) && (@warn "The following brovided callback hooks are not used by $(statetype):\n\t$(uk)\nThe corresponding function will be ignored.\nAvailable hooks: $(pk)")
    end
    return nothing
end