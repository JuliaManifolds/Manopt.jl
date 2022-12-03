
@inline _extract_val(::Val{T}) where {T} = T

"""
    AbstractManoptSolverState

A general super type for all solver states.

# Fields

The following fields are assumed to be default. If you use different ones,
provide the access functions accordingly

* `p` a point on a manifold with the current iterate
* `stop` a [`StoppingCriterion`](@ref).
"""
abstract type AbstractManoptSolverState end

"""
    dispatch_options_decorator(o::AbstractManoptSolverState)

Indicate internally, whether an [`AbstractManoptSolverState`](@ref) `o` to be of decorating type, i.e.
it stores (encapsulates) options in itself, by default in the field `o. options`.

Decorators indicate this by returning `Val{true}` for further dispatch.

The default is `Val{false}`, i.e. by default an options is not decorated.
"""
dispatch_options_decorator(::AbstractManoptSolverState) = Val(false)

"""
    is_options_decorator(o::AbstractManoptSolverState)

Indicate, whether [`AbstractManoptSolverState`](@ref) `o` are of decorator type.
"""
function is_options_decorator(o::AbstractManoptSolverState)
    return _extract_val(dispatch_options_decorator(o))
end

@doc raw"""
    ReturnSolverState{O<:AbstractManoptSolverState} <: AbstractManoptSolverState

This internal type is used to indicate that the contained [`AbstractManoptSolverState`](@ref) `options`
should be returned at the end of a solver instead of the usual minimizer.

# See also
[`get_solver_result`](@ref)
"""
struct ReturnSolverState{O<:AbstractManoptSolverState} <: AbstractManoptSolverState
    options::O
end
dispatch_options_decorator(::ReturnSolverState) = Val(true)

"""
    get_solver_return(O::AbstractManoptSolverState)

determine the result value of a call to a solver. By default this returns the same as [`get_solver_result`](@ref),
i.e. the last iterate or (approximate) minimizer.

    get_solver_return(O::ReturnSolverState)

return the internally stored options of the [`ReturnSolverState`](@ref) instead of the minimizer.
This means that when the options are decorated like this, the user still has to call [`get_solver_result`](@ref)
on the internal options separately.
"""
function get_solver_return(O::AbstractManoptSolverState)
    return get_solver_return(O, dispatch_options_decorator(O))
end
get_solver_return(O::AbstractManoptSolverState, ::Val{false}) = get_solver_result(O)
get_solver_return(O::AbstractManoptSolverState, ::Val{true}) = get_solver_return(O.options)
get_solver_return(O::ReturnSolverState) = O.options

#
# StoppingCriterion meta
#
@doc raw"""
    StoppingCriterion

An abstract type for the functors representing stopping criteria, i.e. they are
callable structures. The naming Scheme follows functions, see for
example [`StopAfterIteration`](@ref).

Every StoppingCriterion has to provide a constructor and its function has to have
the interface `(p,o,i)` where a [`Problem`](@ref) as well as [`AbstractManoptSolverState`](@ref)
and the current number of iterations are the arguments and returns a Bool whether
to stop or not.

By default each `StoppingCriterion` should provide a fields `reason` to provide
details when a criterion is met (and that is empty otherwise).
"""
abstract type StoppingCriterion end

@doc raw"""
    StoppingCriterionGroup <: StoppingCriterion

An abstract type for a Stopping Criterion that itself consists of a set of
Stopping criteria. In total it acts as a stopping criterion itself. Examples
are [`StopWhenAny`](@ref) and [`StopWhenAll`](@ref) that can be used to
combine stopping criteria.
"""
abstract type StoppingCriterionSet <: StoppingCriterion end
#
# StepsizeAbstractManoptSolverState
#
"""
    Stepsize

An abstract type for the functors representing step sizes, i.e. they are callable
structures. The naming scheme is `TypeOfStepSize`, e.g. `ConstantStepsize`.

Every Stepsize has to provide a constructor and its function has to have
the interface `(p,o,i)` where a [`Problem`](@ref) as well as [`AbstractManoptSolverState`](@ref)
and the current number of iterations are the arguments
and returns a number, namely the stepsize to use.

# See also
[`Linesearch`](@ref)
"""
abstract type Stepsize end

@doc raw"""
    get_state(o::AbstractManoptSolverState)

return the undecorated [`AbstractManoptSolverState`](@ref) of the (possibly) decorated `o`.
As long as your decorated options store the options within `o.options` and
the [`dispatch_options_decorator`](@ref) is set to `Val{true}`,
the internal options are extracted.
"""
get_state(o::AbstractManoptSolverState) = get_state(o, dispatch_options_decorator(o))
get_state(o::AbstractManoptSolverState, ::Val{false}) = o
get_state(o::AbstractManoptSolverState, ::Val{true}) = get_state(o.options)

"""
    get_gradient(O::AbstractManoptSolverState)

return the (last stored) gradient within [`AbstractManoptSolverState`](@ref)``O`. By default also undecorates the options beforehand
"""
get_gradient(o::AbstractManoptSolverState) = get_gradient(o, dispatch_options_decorator(o))
function get_gradient(o::AbstractManoptSolverState, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $o do not provide access to a gradient"
    )
end
get_gradient(o::AbstractManoptSolverState, ::Val{true}) = get_gradient(o.options)

"""
    get_iterate(O::AbstractManoptSolverState)

return the (last stored) iterate within [`AbstractManoptSolverState`](@ref)``O`. By default also undecorates the options beforehand
"""
get_iterate(O::AbstractManoptSolverState) = get_iterate(O, dispatch_options_decorator(O))
function get_iterate(O::AbstractManoptSolverState, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $O do not provide access to an iterate"
    )
end
get_iterate(O::AbstractManoptSolverState, ::Val{true}) = get_iterate(O.options)

"""
    set_iterate!(O::AbstractManoptSolverState, p)

set the iterate to some (start) value `p`.
"""
function set_iterate!(O::AbstractManoptSolverState, p)
    return set_iterate!(O, p, dispatch_options_decorator(O))
end
function set_iterate!(O::AbstractManoptSolverState, p, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $O do not provide (write) access to an iterate",
    )
end
set_iterate!(O::AbstractManoptSolverState, p, ::Val{true}) = set_iterate!(O.options, p)

"""
    get_solver_result(O::AbstractManoptSolverState)

return the (last stored) iterate within [`AbstractManoptSolverState`](@ref)``O`. By default also undecorates the options beforehand
"""
function get_solver_result(o::AbstractManoptSolverState)
    return get_solver_result(o, dispatch_options_decorator(o))
end
get_solver_result(o::AbstractManoptSolverState, ::Val{false}) = get_iterate(o)
get_solver_result(o::AbstractManoptSolverState, ::Val{true}) = get_solver_result(o.options)

#
# Common Actions for decorated AbstractManoptSolverState
#
@doc raw"""
    AbstractStateAction

a common `Type` for `AbstractStateActions` that might be triggered in decoraters,
for example [`DebugAbstractManoptSolverState`](@ref) or [`RecordAbstractManoptSolverState`](@ref).
"""
abstract type AbstractStateAction end

@doc raw"""
    StoreStateAction <: AbstractStateAction

internal storage for [`AbstractStateAction`](@ref)s to store a tuple of fields from an
[`AbstractManoptSolverState`](@ref)s

This functor posesses the usual interface of functions called during an
iteration, i.e. acts on `(p,o,i)`, where `p` is a [`Problem`](@ref),
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
    ::P, o::O, i::Int
) where {P<:AbstractManoptProblem,O<:AbstractManoptSolverState}
    #update values (maybe only once)
    if !a.once || a.last_stored != i
        for key in a.keys
            if hasproperty(o, key)
                merge!(a.values, Dict{Symbol,Any}(key => deepcopy(getproperty(o, key))))
            elseif key == :Iterate
                merge!(a.values, Dict{Symbol,Any}(key => deepcopy(get_iterate(o))))
            elseif key == :Gradient
                merge!(a.values, Dict{Symbol,Any}(key => deepcopy(get_gradient(o))))
            else
                @warn "$key is not a field of $o, no storage updated."
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
function update_storage!(a::AbstractStateAction, o::O) where {O<:AbstractManoptSolverState}
    return update_storage!(
        a,
        Dict(
            key => key == :Iterate ? get_iterate(o) : getproperty(o, key) for key in a.keys
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
