#
# While a problem consists of all things one has to know about the Optimization
# problem itself (independent of the solver), the options collect parameters,
# that steer the solver (indipendent of the problem at hand)
#
import Base: copy

export StepsizeOptions, SimpleStepsizeOptions
export ConstantStepsize, DecreasingStepsize
export LinesearchOptions

export EvalOrder, LinearEvalOrder, RandomEvalOrder, FixedRandomEvalOrder
export Options, getOptions
export getOptions
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
"""
abstract type Options end
copy(x::T) where {T <: Options} = T([getfield(x, k) for k ∈ fieldnames(T)]...)
#
#
# StepsizeOptions
#
#
"""
    StepsizeOptions <: Options
A general super type for all options that refer to some line search
"""
abstract type StepsizeOptions <: Options end
"""
    LinesearchOptons <: StepsizeOptions
A general super type for all StepsizeOptions that perform a certain line search
"""
abstract type LinesearchOptions <: StepsizeOptions end

"""
    SimpleStepsizeOptions <: StepsizeOptions
A line search without additional no information required, e.g.
[`ConstantStepsize`](@ref) or [`DecreasingStepsize`](@ref).
"""
mutable struct SimpleStepsizeOptions <: StepsizeOptions end
"""
    ConstantStepsize(s)

returns a `Tuple{Function,<:StepsizeOptions}` with

* a function depenting on a [`Problem`](@ref)` p`, some
[`Options`](@ref)` o`, (empty) [`SimpleStepsizeOptions`](@ref)` lO` and
optionally the current iterate `i` to return a decreasing step size given by
`c/(i^k)`
* The initial (still empty) [`SimpleStepsizeOptions`](@ref)
"""
ConstantStepsize(s::Number) = (
  (p::P where P <: Problem, o::O where O <: Options, lO::SimpleStepsizeOptions, i::Int=1) -> s,
  SimpleStepsizeOptions()
)

@doc doc"""
    DecreasingStepsize(c[,k=1])
returns a `Tuple{Function,<:StepsizeOptions}` with

* a function depenting on a [`Problem`](@ref)` p`, some
[`Options`](@ref)` o`, (empty) [`SimpleStepsizeOptions`](@ref)` lO` and
optionally the current iterate `i` to return a decreasing step size given by
`c/(i^k)`
* The initial (still empty) [`SimpleStepsizeOptions`](@ref)
"""
DecreasingStepsize(s::Number,k::Number=1) = (
  (p::P where P <: Problem, o::O where O <: Options, lO::SimpleStepsizeOptions,i::Int=1) -> s/(i^k),
  SimpleStepsizeOptions()
)
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
