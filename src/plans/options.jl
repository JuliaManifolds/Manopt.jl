#
# While a problem consists of all things one has to know about the Optimization
# problem itself (independent of the solver), the options collect parameters,
# that steer the solver (indipendent of the problem at hand)
#
import Base: copy

export StoppingCriterion, StepSize
export EvalOrder, LinearEvalOrder, RandomEvalOrder, FixedRandomEvalOrder
export Options, getOptions, getReason
export IsOptionsDecorator

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
* `xOld` an [`MPoint`](@ref) with the previous iterate
* `stop` a [`StoppingCriterion`](@ref).

"""
abstract type Options end
copy(x::T) where {T <: Options} = T([getfield(x, k) for k ∈ fieldnames(T)]...)
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
    getReason(c)

return the current reason stored within the [`StoppingCriterion`](@ref) from
within the [`Options`](@ref) This reason is empty if the criterion has never
been met.
"""
getReason(o::O) where O <: Options = getReason( getOptions(o).stop )
