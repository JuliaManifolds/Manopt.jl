const _MANOPT_DEFAULT_CALLBACKS = [:Any, :BeforeInit, :BeforeStep, :BeforeStop, :Init, :Step, :Stop]
const _MANOPT_EMPTY_CALLBACK = (problem, state, k) -> nothing
const _MANOPT_EMPTY_ANY_CALLBACK = (symbol::Symbol, problem, state, k) -> nothing

"""
    active_callbacks(state::AbstractManoptSolverState)

For a given state, indicate which callbacks are in use, that is, stored and also called from this solver.

See also: [`provided_callbacks`](@ref).
"""
function active_callbacks(state::AbstractManoptSolverState)
    dk = keys(get_callbacks(state))
    return collect(intersect(dk, provided_callbacks(typeof(state))))
end

"""
    callback(name::Symbol, problem::AbstractManoptProblem, state::AbstractManoptSolverState, k::Int)

Perform a callback.

This function performs a call to both possible approaches

* if a callback exists in the dictionary under the symbol `name` it is called with the parameters `problem, state, k`
* if a callback exists in the dictionary that shall be called `:Any` time, this one is called with `name, problem, state, k`
"""
function callback(
        name::Symbol, problem::AbstractManoptProblem, state::AbstractManoptSolverState, k::Int
    )
    cb = get_callbacks(state)
    if isempty(cb)
        # avoid checks when no callbacks are provided, this is a performance optimization
        return nothing
    end
    cbs = get(cb, name, _MANOPT_EMPTY_CALLBACK)
    cbs(problem, state, k)
    cba = get(cb, :Any, _MANOPT_EMPTY_ANY_CALLBACK)
    cba(name, problem, state, k)
    return nothing
end

"""
    get_callbacks(state::AbstractManoptSolverState)

Access the callbacks dictionary of the [`AbstractManoptSolverState`](@ref) `state`.
"""
get_callbacks(state::AbstractManoptSolverState) = _get_callbacks(state, dispatch_state_decorator(state))
function _get_callbacks(state::AbstractManoptSolverState, ::Val{false})
    # nonbreaking / safeguard Fallback: No callbacks, so return an empty Dictionary
    return Dict{Symbol, Any}()
end
# For all decorators: Pass down
_get_callbacks(state::AbstractManoptSolverState, ::Val{true}) = get_callbacks(state.state)

"""
    provided_callbacks(state_type::Type{S}) where {S<:AbstractManoptSolverState}

For a solver of type `S` return the callbacks the solver provides, that is, all those
that are called during the solver run.
This function returns a vector of `Symbol`s representing the hooks. These can be keys in
a dictionary of callbacks.
"""
function provided_callbacks(::Type{S}) where {S <: AbstractManoptSolverState}
    return _MANOPT_DEFAULT_CALLBACKS
end

"""
    _callbacks_summary(state::AbstractManoptSolverState)

Generate a callback summary for a given solver `state`. This function returns a string
that contains the active callbacks in the solver state. If no callbacks are active,
an empty string is returned.
"""
function _callbacks_summary(state::AbstractManoptSolverState)
    ac = active_callbacks(state)
    as = length(ac) > 0 ? "\n* active callbacks: :$(join(ac, ", :"))" : ""
    return as
end

"""
    process_callbacks_arg(callbacks::Vector, statetype=missing)
    process_callbacks_arg(callbacks::Dict, statetype=missing) = callbacks

Given a vector `callbacks` a user has passed to a solver, this helper function processes the
vector in the following way

* a pair `:Hook => fct` is kept as is, where the `fct` can also be some callable structure
* a single element `function` is turned into `:Any => fct`, allowing a single callback
  to be specified just by the function
* a pair `[Hook1, Hook2] => fct` is a shortcut for an array of pairs and split into these here.

The result is then wrapped into a dictionary. Be aware that when an array of pairs is reduced
to a dictionary, the last `:Hook` pair “wins” and becomes the entry in the dictionary. This
function does not check for duplicates.

This function keeps a dictionary unchanged. Hence callbacks are only processed once
even if this function is applied multiple times.

If a `statetype` is provided, the final dictionary of callbacks is validated against
the callback hooks provided by that state and a warning is issued if there are hooks that
do not have an effect.
"""
function process_callbacks_arg(callbacks::Vector, statetype = missing)
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
                throw(ArgumentError("Unknown key $(cb[1]) of type $(typeof(cb[1])). Expected a Symbol or an array of symbols"))
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
    process_callbacks_arg(callback, statetype = missing)

Passing a single callback handles this as if it is an array of length one with
this element in the [`process_callbacks_arg`](@ref process_callbacks_arg(::Vector)) array case.
"""
process_callbacks_arg(callback, statetype = missing) = process_callbacks_arg([callback], statetype)
#
function process_callbacks_arg(callbacks::Dict, statetype = missing)
    _warn_if_unused_callbacks(callbacks, statetype)
    return callbacks
end
function _warn_if_unused_callbacks(callbacks::Dict, statetype = missing)
    if !ismissing(statetype)
        ck = keys(callbacks)
        pk = provided_callbacks(statetype)
        uk = collect(setdiff(ck, pk))
        (length(uk) > 0) && (@warn "The following provided callback hooks are not used by $(statetype):\n\t$(uk)\nThe corresponding functions will be ignored.\nAvailable hooks: $(pk)")
    end
    return nothing
end
