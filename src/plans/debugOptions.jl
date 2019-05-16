import Base: stdout
export DebugOptions, getOptions
export DebugAction, DebugGroup, DebugEvery
export DebugChange, DebugIterate, DebugIteration, DebugDivider
export DebugCost, DebugStoppingCriterion

export DebugICC
#
#
# Debug Decorator
#
#
@doc doc"""
    DebugAction

A `DebugAction` is a small functor to store information (i.e. from the last iteration)
and to print/issue debug output. The usual call is given by
`(p,o,i) -> s` that performs the debug and (optionally) returns a string `s`
based on a [`Problem`](@ref) `p`, [`Options`](@ref) `o` and the current iterate
`i`.
By convention `i=0` is interpreted as "For Initialization only", i.e. only debug
info that prints initialization reacts, `i<0` triggers updates of variables
internally but does not trigger any output.

# Fields (assumed by subtypes to exist)
* `print` method to perform the actual print. Can for example be set to a file export,
or to @info. The default is the `print` function on the default `Base.stdout`.
""" 
abstract type DebugAction end

@doc doc"""
    DebugOptions <: Options

The debug options append to any options a debug functionality, i.e. they act as
a decorator pattern. Internally a `Dict`ionary is kept that stores a
[`DebugAction`](@ref) for several occasions using a `Symbol` as reference.
The default occasion is `:All` and for example solvers join this field with
`:Init`, `:Iteration` and `:Final` at the beginning, every iteration or the
end of the algorithm, respectively

The original options can still be accessed using the [`getOptions`](@ref) function.

# Fields (defaults in brackets)
* `options` – the options that are extended by debug information
* `debugDictionary` – a `Dict{Symbol,DebugAction}` to keep track of Debug for different actions

# Constructors
    DebugOptions(o,dA)

construct debug decorated options, where `dD` can be
* a [`DebugAction`](@ref), then it is stored within the dictionary at `:All`
* an `Array` of [`DebugAction`](@ref)s, then it is stored as a
  [`DebugGroup`](@ref) using the first elements `print` for output
* a `Dict{Symbol,DebugAction}`.
"""
mutable struct DebugOptions{O<:Options} <: Options
    options::O
    debugDictionary::Dict{Symbol,<:DebugAction}
    DebugOptions{O}(o::O, dA::Dict{Symbol,<:DebugAction}) where {O <: Options} = new(o,dA)
end
DebugOptions(o::O, dD::D) where {O <: Options, D <: DebugAction} = DebugOptions{O}(o,Dict(:All => dD))
DebugOptions(o::O, dD::Array{<:DebugAction,1}) where {O <: Options} = DebugOptions{O}(o,Dict(:All => DebugGroup(dD,first(dD).print)))
DebugOptions(o::O, dD::Dict{Symbol,<:DebugAction}) where {O <: Options} = DebugOptions{O}(o,dD)

@traitimpl IsOptionsDecorator{DebugOptions}
#
# Summaries / easy Access
#
@doc doc"""
    DebugICC([e=0,print=print])

generate a debug output for [I]teration, current [C]ost and last [C]hange that
only prints every `e`th ieration (deactivated by nonpositive integers)
"""
DebugICC(print::Function=print,e::Int=0) = e>0 ? DebugEvery(DebugGroup([DebugIteration(),DebugCost(),DebugChange()],print),e) : DebugGroup([DebugIteration(),DebugCost(),DebugChange()],print)
#
# Meta Debugs
#
"""
    DebugGroup <: DebugAction

group a set of [`DebugAction`](@ref)s into one action, where the internal prints
are removed by default and the resulting strings are concatenated

# Constructor
    DebugGroup(g)

construct a group consisting of an Array of [`DebugAction`](@ref)s `g`,
that are evaluated `en bloque`; the method does not perform any print itself,
but relies on the internal prints. It still concatenates the result and returns
the complete string
"""
mutable struct DebugGroup <: DebugAction
  group::Array{DebugAction,1}
  DebugGroup(g::Array{DebugAction,1}) = new(g)
end
function (d::DebugGroup)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    for di in d.group
        di(p,o,i)
    end
end

@doc doc"""
    DebugEvery <: DebugAction

evaluate and print debug only every $i$th iteration. Otherwise no print is performed.
Whether internal variables are updates is determined by `alwaysUpdate`.

This method does not perform any print itself but relies on it's childrens print.
"""
mutable struct DebugEvery <: DebugAction
    debug::DebugAction
    every::Int
    alwaysUpdate::Bool
    DebugEvery(d::DebugAction,every::Int=1,alwaysUpdate::Bool=true) = new(d,every,alwaysUpdate)
end
function (d::DebugEvery)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    if (rem(i,d.every)==0)
        d.debug(p,o,i)
    elseif d.alwaysUpdate
        d.debug(p,o,-1)
    end
end

#
# Special single ones
#
@doc doc"""
    DebugChange <: DebugAction

debug for the amount of change of the iterate (stored in `o.x` of the [`Options`](@ref))
during the last iteration.

# Additional Fields
* `xOld` stores the last iterate.
"""
mutable struct DebugChange <: DebugAction
    print::Function
    xOld::MPoint
    DebugChange( print::Function=print) = new(print)
    DebugChange(x0::MPoint, print::Function=print) = new(print, x0)
end
function (d::DebugChange)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    s= (i>0) ? ( isdefined(d,:xOld) ? "Last Change: " * string(distance(p.M,o.x, d.xOld)) : "") : ""
    d.xOld = o.x
    d.print(s)
end
@doc doc"""
    DebugChange <: DebugAction

debug for the current iterate (stored in `o.x`).

# Parameters
* `long::Bool` whether to print `x:` or `current iterate`
"""
mutable struct DebugIterate <: DebugAction
    print::Function
    prefix::String
    DebugIterate(print::Function=print,long::Bool=false) = new(print, long ? "current Iterate:" : "x:")
end
(d::DebugIterate)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = d.print( (i>=0) ? prefix*"$(o.x)" : "")

@doc doc"""
    DebugIteration <: DebugAction

debug for the current iteration (prefixed with `#`)
"""
mutable struct DebugIteration <: DebugAction
    print::Function
    DebugIteration(print::Function=print) = new(print)
end
(d::DebugIteration)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = d.print( (i>0) ? "# $(i)" : "")
    
@doc doc"""
    DebugCost <: DebugAction

print the current cost function value, see [`getCost`](@ref).

# Constructors
    DebugCost(long,print)

where `long` indicated whether to print `F(x):` (default) or `costFunction: `

    DebugCost(prefix,print)

set a prefix manually.
"""
mutable struct DebugCost <: DebugAction
    print::Function
    prefix::String
    DebugCost(long::Bool=false,print::Function=print) = new(print, long ? "Cost Function: " : "F(x): ")
    DebugCost(prefix::String,print::Function=print) = new(print,prefix)
end
(d::DebugCost)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = d.print( (i>=0) ? d.prefix*string(getCost(p,o.x)) : "")

@doc doc"""
    DebugDivider <: DebugAction

print a small `div`ider (default `" | "`).

# Constructor
    DebugDivider(div,print)

"""
mutable struct DebugDivider <: DebugAction
    print::Function
    divider::String
    DebugDivider(divider=" | ",print::Function=print) = new(print,divider)
end
(d::DebugDivider)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = d.print((i>=0) ? d.divider : "")

@doc doc"""
    DebugStoppingCriterion <: DebugAction

print the Reason provided by the stopping criterion. Usually this should be
empty, unless the algorithm stops.
"""
mutable struct DebugStoppingCriterion <: DebugAction
    print::Function
    DebugStoppingCriterion(print::Function=print) = new(print)
end
(d::DebugStoppingCriterion)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = d.print( (i>=0) ? getReason(o) : "")
