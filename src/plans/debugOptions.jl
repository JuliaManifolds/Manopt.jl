import Base: stdout
export DebugOptions, getOptions
export DebugAction, DebugGroup, DebugEntry, DebugEntryChange, DebugEvery
export DebugChange, DebugIterate, DebugIteration, DebugDivider
export DebugCost, DebugStoppingCriterion, DebugFactory, DebugActionFactory
#
#
# Debug Options Decorator
#
#
@doc doc"""
    DebugAction

A `DebugAction` is a small functor to print/issue debug output.
The usual call is given by `(p,o,i) -> s` that performs the debug based on
a [`Problem`](@ref) `p`, [`Options`](@ref) `o` and the current iterate `i`.

By convention `i=0` is interpreted as "For Initialization only", i.e. only debug
info that prints initialization reacts, `i<0` triggers updates of variables
internally but does not trigger any output. Finally `typemin(Int)` is used
to indicate a call from [`stopSolver!`](@ref) that returns true afterwards.

# Fields (assumed by subtypes to exist)
* `print` method to perform the actual print. Can for example be set to a file export,
or to @info. The default is the `print` function on the default `Base.stdout`.
""" 
abstract type DebugAction <: Action end

@doc doc"""
    DebugOptions <: Options

The debug options append to any options a debug functionality, i.e. they act as
a decorator pattern. Internally a `Dict`ionary is kept that stores a
[`DebugAction`](@ref) for several occasions using a `Symbol` as reference.
The default occasion is `:All` and for example solvers join this field with
`:Start`, `:Step` and `:Stop` at the beginning, every iteration or the
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
  `debugDictionary` within `:All`.
* a `Dict{Symbol,DebugAction}`.
* an Array of Symbols, String and an Int for the [`DebugFactory`](@ref)
"""
mutable struct DebugOptions{O<:Options} <: Options
    options::O
    debugDictionary::Dict{Symbol, <: DebugAction}
    DebugOptions{O}(o::O, dA::Dict{Symbol,<:DebugAction}) where {O <: Options} = new(o,dA)
end
DebugOptions(o::O, dD::D) where {O <: Options, D <: DebugAction} = DebugOptions{O}(o,Dict(:All => dD))
DebugOptions(o::O, dD::Array{<:DebugAction,1}) where {O <: Options} = DebugOptions{O}(o,Dict(:All => DebugGroup(dD)))
DebugOptions(o::O, dD::Dict{Symbol,<:DebugAction}) where {O <: Options} = DebugOptions{O}(o,dD)
DebugOptions(o::O, format::Array{<:Any,1}) where {O <: Options} = DebugOptions{O}(o, DebugFactory(format))

@traitimpl IsOptionsDecorator{DebugOptions}

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
  DebugGroup(g::Array{<:DebugAction,1}) = new(g)
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
    DebugChange(a,prefix,print)

debug for the amount of change of the iterate (stored in `o.x` of the [`Options`](@ref))
during the last iteration. See [`DebugEntryChange`](@ref)

# Parameters
* `x0` – an initial value to already get a Change after the first iterate. Can be left out
* `a` – (`StoreOptionsAction( (:x,) )`) – the storage of the previous action
* `prefix` – (`"Last Change:"`) prefix of the debug output
* `print` – (`print`) default method to peform the print.
"""
mutable struct DebugChange <: DebugAction
    print::Function
    prefix::String
    storage::StoreOptionsAction
    DebugChange(a::StoreOptionsAction=StoreOptionsAction( (:x,) ),
            prefix = "Last Change: ",
            print::Function=print
        ) = new(print, prefix, a)
end
function (d::DebugChange)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    s = (i>0) ? ( hasStorage(d.storage, :x) ? d.prefix * string(
            distance(p.M, o.x, getStorage(d.storage, :x))
            ) : "") : ""
    d.storage(p,o,i)
    d.print(s)
end
@doc doc"""
    DebugIterate <: DebugAction

debug for the current iterate (stored in `o.x`).

# Parameters
* `long::Bool` whether to print `x:` or `current iterate`
"""
mutable struct DebugIterate <: DebugAction
    print::Function
    prefix::String
    DebugIterate(print::Function=print,long::Bool=false) = new(print, long ? "current Iterate:" : "x:")
end
(d::DebugIterate)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = d.print( (i>=0) ? d.prefix*"$(o.x)" : "")

@doc doc"""
    DebugIteration <: DebugAction

debug for the current iteration (prefixed with `#`)
"""
mutable struct DebugIteration <: DebugAction
    print::Function
    DebugIteration(print::Function=print) = new(print)
end
(d::DebugIteration)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = d.print( (i>0) ? "# $(i)" : ((i==0) ? "Initial" : "") )
    
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
    DebugEntry <: RecordAction

print a certain fields entry of type {T} during the iterates

# Addidtional Fields
* `field` – Symbol the entry can be accessed with within [`Options`](@ref)

# Constructor

    DebugEntry(f[, prefix="$f:", print=print])

"""
mutable struct DebugEntry <: DebugAction
    print::Function
    prefix::String
    field::Symbol
    DebugEntry(f::Symbol,prefix="$f:",print::Function=print) = new(print,prefix,f)
end
(d::DebugEntry)(p::Pr,o::O,i::Int) where {Pr <: Problem, O <: Options} = d.print(
    (i>=0) ? d.prefix*" "*string(getfield(o, d.field)) : "")

@doc doc"""
    DebugEntryChange{T} <: DebugAction

print a certain entries change during iterates

# Additional Fields
* `print` – (`print`) function to print the result
* `prefix` – (`"Change of :x"`) prefix to the print out
* `field` – Symbol the field can be accessed with within [`Options`](@ref)
* `distance` – function (p,o,x1,x2) to compute the change/distance between two values of the entry 
* `storage` – a [`StoreOptionsAction`](@ref) to store the previous value of `:f`

# Constructors

    DebugEntryChange(f,d[, a, prefix, print])

initialize the Debug to a field `f` and a `distance` `d`.


    DebugEntryChange(v,f,d[, a, prefix="Change of $f:", print])

initialize the Debug to a field `f` and a `distance` `d` with initial value `v`
for the history of `o.field`.
"""
mutable struct DebugEntryChange <: DebugAction
    print::Function
    prefix::String
    field::Symbol
    distance::Function
    storage::StoreOptionsAction
    DebugEntryChange(f::Symbol,d::Function,
            a::StoreOptionsAction=StoreOptionsAction( (f,) ),
            prefix = "Change of $f:",
            print::Function=print
        ) = new(print, prefix, f, d, a)
    function DebugEntryChange(v::T where T, f::Symbol, d::Function,
            a::StoreOptionsAction=StoreOptionsAction( (f,) ),
            prefix = "Change of $f:",
            print::Function=print
        )
        updateStorage!(a,Dict(f=>v))
        return new(print, prefix, f, d, a)
    end
end
function (d::DebugEntryChange)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    s= (i>0) ? ( hasStorage(d.storage,d.field) ? d.prefix * string(
            d.distance( p, o, getproperty(o, d.field), getStorage(d.storage,d.field))
            ) : "") : ""
    d.storage(p,o,i)
    d.print(s)
end

@doc doc"""
    DebugStoppingCriterion <: DebugAction

print the Reason provided by the stopping criterion. Usually this should be
empty, unless the algorithm stops.
"""
mutable struct DebugStoppingCriterion <: DebugAction
    print::Function
    DebugStoppingCriterion(print::Function=print) = new(print)
end
(d::DebugStoppingCriterion)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = d.print( (i>=0 || i==typemin(Int)) ? getReason(o) : "")

@doc doc"""
    DebugFactory(a)

given an array of `Symbol`s, `String`s [`DebugAction`](@ref)s and `Ints`

* The symbol `:Stop` creates an entry of to display the stoping criterion at the end
  (`:Stop => DebugStoppingCriterion()`)
* The symbol `:Cost` creates a [`DebugCost`](@ref)
* The symbol `:iteration` creates a [`DebugIteration`](@ref)
* The symbol `:Change` creates a [`DebugChange`](@ref)
* any other symbol creates debug output of the corresponding field in [`Options`](@ref)
* any string creates a [`DebugDivider`](@ref)
* any [`DebugAction`](@ref) is directly included
* an Integer `k`introduces that debug is only printed every `k`th iteration
"""
function DebugFactory(a::Array{<:Any,1} )
    # filter out every
    group = Array{DebugAction,1}()
    for s in filter(x -> !isa(x,Int) && x!=:Stop, a) # filter ints and stop
        push!(group,DebugActionFactory(s) )
    end
    dictionary = Dict{Symbol,DebugAction}()
    if length(group) > 0
        debug = DebugGroup(group)
        # filter ints
        e = filter(x -> isa(x,Int),a)
        if length(e) > 0
            debug = DebugEvery(debug,last(e))
        end
        dictionary[:All] = debug
    end
    if :Stop in a
        dictionary[:Stop] = DebugStoppingCriterion()
    end
    return dictionary
end
@doc doc"""
    DebugActionFactory(s)

create a [`DebugAction`](@ref) where

* a `String`yields the correspoinding divider
* a [`DebugAction`](@ref) is passed through
* a [`Symbol`] creates [`DebugEntry`](@ref) of that symbol, with the exceptions
  of `:Change`, `:Iterate`, `:Iteration`, and `:Cost`.
"""
DebugActionFactory(s::String) = DebugDivider(s)
DebugActionFactory(a::A) where {A <: DebugAction} = a
function DebugActionFactory(s::Symbol)
    if (s==:Change)
        return DebugChange()
    elseif (s==:Iteration)
        return DebugIteration()
    elseif (s==:Iterate)
        return DebugIterate()
    elseif (s==:Cost)
        return DebugCost()
    end
        return DebugEntry(s)
end
