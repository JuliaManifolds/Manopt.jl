#
# While a problem consists of all things one has to know about the Optimization
# problem itself (independent of the solver), the options collect parameters,
# that steer the solver (indipendent of the problem at hand)
#
import Base: copy

export StoppingCriterion, Stepsize
export EvalOrder, LinearEvalOrder, RandomEvalOrder, FixedRandomEvalOrder
export Options, getOptions, getReason
export IsOptionsDecorator

export Action, StoreOptionsAction
export hasStorage, getStorage, updateStorage!

"""
    IsOptionsDecorator{O}

A trait to specify that a certain `Option` decorates, i.e. internally
stores the original [`Options`](@ref) under consideration.
"""
@traitdef IsOptionsDecorator{O}

"""
    Options

A general super type for all options.

# Fields
The following fields are assumed to be default. If you use different ones,
provide the access functions accordingly
* `x` an [`MPoint`](@ref) with the current iterate
* `stop` a [`StoppingCriterion`](@ref).

"""
abstract type Options end
#
# StoppingCriterion meta
#
@doc doc""" 
    StoppingCriterion

An abstract type for the functors representing stoping criteria, i.e. they are
callable structures. The naming Scheme follows functions, see for
example [`stopAfterIteration`](@ref).

Every StoppingCriterion has to provide a constructor and its function has to have
the interface `(p,o,i)` where a [`Problem`](@ref) as well as [`Options`](@ref)
and the current number of iterations are the arguments and returns a Bool whether
to stop or not.

By default each `StoppingCriterion` should provide a fiels `reason` to provide
details when a criteion is met (and that is empty otherwise).
"""
abstract type StoppingCriterion end
#
#
# StepsizeOptions
#
#
"""
    Stepsize

An abstract type for the functors representing step sizes, i.e. they are callable
structurs. The naming scheme is `TypeOfStepSize`, e.g. `ConstantStepsize`.

Every Stepsize has to provide a constructor and its function has to have
the interface `(p,o,i)` where a [`Problem`](@ref) as well as [`Options`](@ref)
and the current number of iterations are the arguments
and returns a number, namely the stepsize to use.

# See also
[`Linesearch`](@ref)
"""
abstract type Stepsize end
#
#
# Evalualtion Orders
#
#
"""
    EvalOrder
type for specifying an evaluation order for any cyclicly evaluated algorithms
"""
abstract type EvalOrder end
"""
    LinearEvalOrder <: EvalOrder
evaluate in a linear order, i.e. for each cycle of length l evaluate in the
order 1,2,...,l.
"""
mutable struct LinearEvalOrder <: EvalOrder end
"""
    RandomEvalOrder <: EvalOrder
choose a random order for each evaluation of the l functionals.
"""
mutable struct RandomEvalOrder <: EvalOrder end
"""
    FixedRandomEvalOrder <: EvalOrder
Choose a random order once and evaluate always in this order, i.e. for
l elements there is one chosen permutation used for each iteration cycle.
"""
mutable struct FixedRandomEvalOrder <: EvalOrder end

@doc doc"""
    getOptions(O)

return the undecorated [`Options`](@ref) of the (possibly) decorated `O`.
As long as your decorated options stores the options within `o.options` and
implements the `SimpleTrait` `IsOptionsDecorator`, this is behaviour is optained
automatically.
"""
getOptions(O) = error("Not implemented for types that are not `Options`")
# this might seem like a trick/fallback just for documentation reasons
@traitfn getOptions(o::O) where {O <: Options; !IsOptionsDecorator{O}} = o
@traitfn getOptions(o::O) where {O <: Options; IsOptionsDecorator{O}} = getOptions(o.options)

@doc doc"""
    getReason(o)

return the current reason stored within the [`StoppingCriterion`](@ref) from
within the [`Options`](@ref) This reason is empty if the criterion has never
been met.
"""
getReason(o::O) where O <: Options = getReason( getOptions(o).stop )

#
# Common Actions for decorated Options
#
@doc doc"""
    Action

a common `Type` for `Actions` that might be triggered in decoraters,
for example [`DebugOptions`](@ref) or [`RecordOptions`](@ref).
"""
abstract type Action end


@doc doc"""
    StoreTupleAction <: Action

internal storage for [`Action`](@ref)s to store a tuple of fields from an
[`Options`](@ref)s 

This functor posesses the usual interface of functions called during an
iteration, i.e. acts on `(p,o,i)`, where `p` is a [`Problem`](@ref),
`o` is an [`Options`](@ref) and `i` is the current iteration.

# Fields
* `values` – a dictionary to store interims values based on certain `Symbols`
* `keys` – an `NTuple` of `Symbols` to refer to fields of `Options`
* `once` – whether to update the internal values only once per iteration
* `lastStored` – last iterate, where this `Action` was called (to determine `once`

# Constructiors

    StoreOptionsAction([keys=(), once=true])

Initialize the Functor to an (empty) set of keys, where `once` determines
whether more that one update per iteration are effective

    StoreOptionsAction(keys, once=true])

Initialize the Functor to a set of keys, where the dictionary is initialized to
be empty. Further, `once` determines whether more that one update per iteration
are effective, otherwise only the first update is stored, all others are ignored.
"""
mutable struct StoreOptionsAction <: Action
    values::Dict{Symbol,<:Any}
    keys::NTuple{N,Symbol} where N
    once::Bool
    lastStored::Int
    StoreOptionsAction(keys::NTuple{N,Symbol} where N = NTuple{0,Symbol}(),once=true) = new(Dict{Symbol,Any}(), keys, once,-1 )
end
function (a::StoreOptionsAction)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    #update values (maybe only once)
    if !a.once || a.lastStored != i
        merge!(a.values, Dict( key => getproperty(o,key) for key in a.keys) )
    end
    a.lastStored = i
end
"""
    getStorage(a,key)

return the internal value of the [`StoreOptionsAction`](@ref) `a` at the
`Symbol` `key`.
"""
getStorage(a::StoreOptionsAction,key) = a.values[key]
"""
    getStorage(a,key)

return whether the [`StoreOptionsAction`](@ref) `a` has a value stored at the
`Symbol` `key`.
"""
hasStorage(a::StoreOptionsAction,key) = haskey(a.values,key)
"""
    updateStorage!(a,o)

update the [`StoreOptionsAction`](@ref) `a` internal values to the ones given on
the [`Options`](@ref) `o`.
"""
updateStorage!(a::StoreOptionsAction,o::O) where {O <: Options} = updateStorage!(a, Dict( key => getproperty(o, key) for key in a.keys) )
"""
    updateStorage!(a,o)

update the [`StoreOptionsAction`](@ref) `a` internal values to the ones given in
the dictionary `d`. The values are merged, where the values from `d` are preferred.
"""
function updateStorage!(a::StoreOptionsAction,d::Dict{Symbol,<:Any}) where {O <: Options}
    merge!(a.values, d)
    # update keys
    a.keys = Tuple( keys(a.values) )
end