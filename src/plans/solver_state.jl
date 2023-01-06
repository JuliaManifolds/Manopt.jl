
@inline _extract_val(::Val{T}) where {T} = T

@doc raw"""
    AbstractManoptSolverState

A general super type for all solver states.

# Fields

The following fields are assumed to be default. If you use different ones,
provide the access functions accordingly

* `p` a point on a manifold with the current iterate
* `stop` a [`StoppingCriterion`](@ref).
"""
abstract type AbstractManoptSolverState end

@doc raw"""
    AbstractGradientSolverState <: AbstractManoptSolverState

A generic [`AbstractManoptSolverState`](@ref) type for gradient based options data.

It assumes that

* the iterate is stored in the field `p`
* the gradient at `p` is stored in `X`.

# see also
[`GradientDescentState`](@ref), [`StochasticGradientDescentState`](@ref), [`SubGradientMethodState`](@ref), [`QuasiNewtonState`](@ref).
"""
abstract type AbstractGradientSolverState <: AbstractManoptSolverState end

"""
    dispatch_state_decorator(s::AbstractManoptSolverState)

Indicate internally, whether an [`AbstractManoptSolverState`](@ref) `s` to be of decorating type, i.e.
it stores (encapsulates) state in itself, by default in the field `s.state`.

Decorators indicate this by returning `Val{true}` for further dispatch.

The default is `Val{false}`, i.e. by default an state is not decorated.
"""
dispatch_state_decorator(::AbstractManoptSolverState) = Val(false)

"""
    is_state_decorator(s::AbstractManoptSolverState)

Indicate, whether [`AbstractManoptSolverState`](@ref) `s` are of decorator type.
"""
function is_state_decorator(s::AbstractManoptSolverState)
    return _extract_val(dispatch_state_decorator(s))
end

@doc raw"""
    ReturnSolverState{O<:AbstractManoptSolverState} <: AbstractManoptSolverState

This internal type is used to indicate that the contained [`AbstractManoptSolverState`](@ref) `state`
should be returned at the end of a solver instead of the usual minimizer.

# See also

[`get_solver_result`](@ref)
"""
struct ReturnSolverState{S<:AbstractManoptSolverState} <: AbstractManoptSolverState
    state::S
end
dispatch_state_decorator(::ReturnSolverState) = Val(true)

"""
    get_solver_return(O::AbstractManoptSolverState)

determine the result value of a call to a solver. By default this returns the same as [`get_solver_result`](@ref),
i.e. the last iterate or (approximate) minimizer.

    get_solver_return(O::ReturnSolverState)

return the internally stored state of the [`ReturnSolverState`](@ref) instead of the minimizer.
This means that when the state are decorated like this, the user still has to call [`get_solver_result`](@ref)
on the internal state separately.
"""
function get_solver_return(s::AbstractManoptSolverState)
    return _get_solver_return(s, dispatch_state_decorator(s))
end
_get_solver_return(s::AbstractManoptSolverState, ::Val{false}) = get_solver_result(s)
_get_solver_return(s::AbstractManoptSolverState, ::Val{true}) = get_solver_return(s.state)
get_solver_return(s::ReturnSolverState) = s.state

@doc raw"""
    get_state(s::AbstractManoptSolverState)

return the undecorated [`AbstractManoptSolverState`](@ref) of the (possibly) decorated `s`.
As long as your decorated state store the state within `s.state` and
the [`dispatch_state_decorator`](@ref) is set to `Val{true}`,
the internal state are extracted.
"""
get_state(s::AbstractManoptSolverState) = _get_state(s, dispatch_state_decorator(s))
_get_state(s::AbstractManoptSolverState, ::Val{false}) = s
_get_state(s::AbstractManoptSolverState, ::Val{true}) = get_state(s.state)

"""
    get_gradient(s::AbstractManoptSolverState)

return the (last stored) gradient within [`AbstractManoptSolverState`](@ref)` `s`.
By default also undecorates the state beforehand
"""
get_gradient(s::AbstractManoptSolverState) = _get_gradient(s, dispatch_state_decorator(s))
function _get_gradient(s::AbstractManoptSolverState, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s do not provide access to a gradient"
    )
end
_get_gradient(s::AbstractManoptSolverState, ::Val{true}) = get_gradient(s.state)

"""
    set_gradient!(s::AbstractManoptSolverState, M::AbstractManifold, p, X)

set the gradient within an (possibly decorated) [`AbstractManoptSolverState`](@ref)
to some (start) value `X` in the tangent space at `p`.
"""
function set_gradient!(s::AbstractManoptSolverState, M, p, X)
    return _set_gradient!(s, M, p, X, dispatch_state_decorator(s))
end
function _set_gradient!(s::AbstractManoptSolverState, ::Any, ::Any, ::Any, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s do not provide (write) access to a gradient",
    )
end
function _set_gradient!(s::AbstractManoptSolverState, M, p, X, ::Val{true})
    return set_gradient!(s.state, M, p, X)
end

"""
    get_iterate(O::AbstractManoptSolverState)

return the (last stored) iterate within [`AbstractManoptSolverState`](@ref)` `s`.
By default also undecorates the state beforehand.
"""
get_iterate(s::AbstractManoptSolverState) = _get_iterate(s, dispatch_state_decorator(s))
function _get_iterate(s::AbstractManoptSolverState, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s do not provide access to an iterate"
    )
end
_get_iterate(s::AbstractManoptSolverState, ::Val{true}) = get_iterate(s.state)

"""
    set_iterate!(s::AbstractManoptSolverState, M::AbstractManifold, p)

set the iterate within an [`AbstractManoptSolverState`](@ref) to some (start) value `p`.
"""
function set_iterate!(s::AbstractManoptSolverState, M, p)
    return _set_iterate!(s, M, p, dispatch_state_decorator(s))
end
function _set_iterate!(s::AbstractManoptSolverState, ::Any, ::Any, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s do not provide (write) access to an iterate",
    )
end
_set_iterate!(s::AbstractManoptSolverState, M, p, ::Val{true}) = set_iterate!(s.state, M, p)

"""
    get_solver_result(ams::AbstractManoptSolverState)

Return the final result after all iterations that is stored within
the [`AbstractManoptSolverState`](@ref) `ams`, which was modified during the iterations.
"""
function get_solver_result(s::AbstractManoptSolverState)
    return get_solver_result(s, dispatch_state_decorator(s))
end
get_solver_result(s::AbstractManoptSolverState, ::Val{false}) = get_iterate(s)
get_solver_result(s::AbstractManoptSolverState, ::Val{true}) = get_solver_result(s.state)

#
# Common Actions for decorated AbstractManoptSolverState
#
@doc raw"""
    AbstractStateAction

a common `Type` for `AbstractStateActions` that might be triggered in decoraters,
for example within the [`DebugSolverState`](@ref) or within the [`RecordSolverState`](@ref).
"""
abstract type AbstractStateAction end

@doc raw"""
    StoreStateAction <: AbstractStateAction

internal storage for [`AbstractStateAction`](@ref)s to store a tuple of fields from an
[`AbstractManoptSolverState`](@ref)s

This functor posesses the usual interface of functions called during an
iteration, i.e. acts on `(p,o,i)`, where `p` is a [`AbstractManoptProblem`](@ref),
`o` is an [`AbstractManoptSolverState`](@ref) and `i` is the current iteration.

# Fields
* `values` – a dictionary to store interims values based on certain `Symbols`
* `keys` – an `NTuple` of `Symbols` to refer to fields of `AbstractManoptSolverState`
* `once` – whether to update the internal values only once per iteration
* `lastStored` – last iterate, where this `AbstractStateAction` was called (to determine `once`

# Constructiors

    AbstractStateAction([keys=(), once=true])

Initialize the Functor to an (empty) set of keys, where `once` determines
whether more that one update per iteration are effective

    AbstractStateAction(keys, once=true])

Initialize the Functor to a set of keys, where the dictionary is initialized to
be empty. Further, `once` determines whether more that one update per iteration
are effective, otherwise only the first update is stored, all others are ignored.
"""
mutable struct StoreStateAction <: AbstractStateAction
    values::Dict{Symbol,<:Any}
    keys::NTuple{N,Symbol} where {N}
    once::Bool
    last_stored::Int
    function StoreStateAction(
        keys::NTuple{N,Symbol} where {N}=NTuple{0,Symbol}(), once=true
    )
        return new(Dict{Symbol,Any}(), keys, once, -1)
    end
end
function (a::StoreStateAction)(
    ::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    #update values (maybe only once)
    if !a.once || a.last_stored != i
        for key in a.keys
            if hasproperty(s, key)
                merge!(a.values, Dict{Symbol,Any}(key => deepcopy(getproperty(s, key))))
            elseif key == :Iterate
                merge!(a.values, Dict{Symbol,Any}(key => deepcopy(get_iterate(s))))
            elseif key == :Gradient
                merge!(a.values, Dict{Symbol,Any}(key => deepcopy(get_gradient(s))))
            end
        end
    end
    return a.last_stored = i
end

"""
    get_storage(a,key)

return the internal value of the [`AbstractStateAction`](@ref) `a` at the
`Symbol` `key`.
"""
get_storage(a::AbstractStateAction, key) = a.values[key]

"""
    get_storage(a,key)

return whether the [`AbstractStateAction`](@ref) `a` has a value stored at the
`Symbol` `key`.
"""
has_storage(a::AbstractStateAction, key) = haskey(a.values, key)

"""
    update_storage!(a,o)

update the [`AbstractStateAction`](@ref) `a` internal values to the ones given on
the [`AbstractManoptSolverState`](@ref) `o`.
"""
function update_storage!(a::AbstractStateAction, s::AbstractManoptSolverState)
    return update_storage!(
        a,
        Dict(
            key => if key === :Iterate
                get_iterate(s)
            else
                (key === :gradient ? get_gradient(s) : getproperty(s, key))
            end for key in a.keys
        ),
    )
end

"""
    update_storage!(a,o)

update the [`AbstractStateAction`](@ref) `a` internal values to the ones given in
the dictionary `d`. The values are merged, where the values from `d` are preferred.
"""
function update_storage!(a::AbstractStateAction, d::Dict{Symbol,<:Any})
    merge!(a.values, d)
    # update keys
    return a.keys = Tuple(keys(a.values))
end
