@doc raw"""
    DebugAction

A `DebugAction` is a small functor to print/issue debug output.
The usual call is given by `(p,o,i) -> s` that performs the debug based on
a [`Problem`](@ref) `p`, [`Options`](@ref) `o` and the current iterate `i`.

By convention `i=0` is interpreted as "For Initialization only", i.e. only debug
info that prints initialization reacts, `i<0` triggers updates of variables
internally but does not trigger any output. Finally `typemin(Int)` is used
to indicate a call from [`stop_solver!`](@ref) that returns true afterwards.

# Fields (assumed by subtypes to exist)
* `print` method to perform the actual print. Can for example be set to a file export,
or to @info. The default is the `print` function on the default `Base.stdout`.
"""
abstract type DebugAction <: AbstractOptionsAction end

@doc raw"""
    DebugOptions <: Options

The debug options append to any options a debug functionality, i.e. they act as
a decorator pattern. Internally a `Dict`ionary is kept that stores a
[`DebugAction`](@ref) for several occasions using a `Symbol` as reference.
The default occasion is `:All` and for example solvers join this field with
`:Start`, `:Step` and `:Stop` at the beginning, every iteration or the
end of the algorithm, respectively

The original options can still be accessed using the [`get_options`](@ref) function.

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
    debugDictionary::Dict{Symbol,<:DebugAction}
    DebugOptions{O}(o::O, dA::Dict{Symbol,<:DebugAction}) where {O<:Options} = new(o, dA)
end
function DebugOptions(o::O, dD::D) where {O<:Options,D<:DebugAction}
    return DebugOptions{O}(o, Dict(:All => dD))
end
function DebugOptions(o::O, dD::Array{<:DebugAction,1}) where {O<:Options}
    return DebugOptions{O}(o, Dict(:All => DebugGroup(dD)))
end
function DebugOptions(o::O, dD::Dict{Symbol,<:DebugAction}) where {O<:Options}
    return DebugOptions{O}(o, dD)
end
function DebugOptions(o::O, format::Array{<:Any,1}) where {O<:Options}
    return DebugOptions{O}(o, DebugFactory(format))
end

dispatch_options_decorator(o::DebugOptions) = Val(true)

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
function (d::DebugGroup)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    for di in d.group
        di(p, o, i)
    end
end

@doc raw"""
    DebugEvery <: DebugAction

evaluate and print debug only every $i$th iteration. Otherwise no print is performed.
Whether internal variables are updates is determined by `alwaysUpdate`.

This method does not perform any print itself but relies on it's childrens print.
"""
mutable struct DebugEvery <: DebugAction
    debug::DebugAction
    every::Int
    alwaysUpdate::Bool
    function DebugEvery(d::DebugAction, every::Int=1, alwaysUpdate::Bool=true)
        return new(d, every, alwaysUpdate)
    end
end
function (d::DebugEvery)(p::Problem, o::Options, i::Int)
    if (rem(i, d.every) == 0)
        d.debug(p, o, i)
    elseif d.alwaysUpdate
        d.debug(p, o, -1)
    end
end

#
# Special single ones
#
@doc raw"""
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
    io::IO
    prefix::String
    storage::StoreOptionsAction
    function DebugChange(
        a::StoreOptionsAction=StoreOptionsAction((:x,)),
        prefix="Last Change: ",
        io::IO=stdout,
    )
        return new(io, prefix, a)
    end
end
function (d::DebugChange)(p::Problem, o::Options, i::Int)
    s = if (i > 0)
        (
            if has_storage(d.storage, :x)
                d.prefix * string(distance(p.M, o.x, get_storage(d.storage, :x)))
            else
                ""
            end
        )
    else
        ""
    end
    d.storage(p, o, i)
    print(d.io, s)
    return nothing
end
@doc raw"""
    DebugIterate <: DebugAction

debug for the current iterate (stored in `o.x`).

# Constructor
    DebugIterate(io=stdout, long::Bool=false)

# Parameters
* `long::Bool` whether to print `x:` or `current iterate`
"""
mutable struct DebugIterate <: DebugAction
    io::IO
    prefix::String
    function DebugIterate(io::IO=stdout, long::Bool=false)
        return new(io, long ? "current Iterate:" : "x:")
    end
end
function (d::DebugIterate)(::Problem, o::Options, i::Int)
    print(d.io, (i >= 0) ? d.prefix * "$(o.x)" : "")
    return nothing
end

@doc raw"""
    DebugIteration <: DebugAction

debug for the current iteration (prefixed with `#`)
"""
mutable struct DebugIteration <: DebugAction
    io::IO
    DebugIteration(io::IO=stdout) = new(io)
end
function (d::DebugIteration)(::Problem, ::Options, i::Int)
    print(d.io, (i > 0) ? "# $(i)" : ((i == 0) ? "Initial" : ""))
    return nothing
end

@doc raw"""
    DebugCost <: DebugAction

print the current cost function value, see [`get_cost`](@ref).

# Constructors
    DebugCost(long,print)

where `long` indicated whether to print `F(x):` (default) or `cost: `

    DebugCost(prefix,print)

set a prefix manually.
"""
mutable struct DebugCost <: DebugAction
    io::IO
    prefix::String
    function DebugCost(long::Bool=false, io::IO=stdout)
        return new(io, long ? "Cost Function: " : "F(x): ")
    end
    DebugCost(prefix::String, io::IO=stdout) = new(io, prefix)
end
function (d::DebugCost)(p::Problem, o::Options, i::Int)
    print(d.io, (i >= 0) ? d.prefix * string(get_cost(p, o.x)) : "")
    return nothing
end

@doc raw"""
    DebugDivider <: DebugAction

print a small `div`ider (default `" | "`).

# Constructor
    DebugDivider(div,print)

"""
mutable struct DebugDivider <: DebugAction
    io::IO
    divider::String
    DebugDivider(divider=" | ", io::IO=stdout) = new(io, divider)
end
function (d::DebugDivider)(::Problem, ::Options, i::Int) where {P<:Problem,O<:Options}
    print(d.io, (i >= 0) ? d.divider : "")
    return nothing
end

@doc raw"""
    DebugEntry <: RecordAction

print a certain fields entry of type {T} during the iterates

# Addidtional Fields
* `field` – Symbol the entry can be accessed with within [`Options`](@ref)

# Constructor

    DebugEntry(f[, prefix="$f:", io=stdout])

"""
mutable struct DebugEntry <: DebugAction
    io::IO
    prefix::String
    field::Symbol
    DebugEntry(f::Symbol, prefix="$f:", io::IO=stdout) = new(io, prefix, f)
end
function (d::DebugEntry)(::Problem, o::Options, i::Int)
    print(d.io, (i >= 0) ? d.prefix * " " * string(getfield(o, d.field)) : "")
    return nothing
end

@doc raw"""
    DebugEntryChange{T} <: DebugAction

print a certain entries change during iterates

# Additional Fields
* `print` – (`print`) function to print the result
* `prefix` – (`"Change of :x"`) prefix to the print out
* `field` – Symbol the field can be accessed with within [`Options`](@ref)
* `distance` – function (p,o,x1,x2) to compute the change/distance between two values of the entry
* `storage` – a [`StoreOptionsAction`](@ref) to store the previous value of `:f`

# Constructors

    DebugEntryChange(f,d[, a, prefix, io])

initialize the Debug to a field `f` and a `distance` `d`.


    DebugEntryChange(v,f,d[, a, prefix="Change of $f:", io])

initialize the Debug to a field `f` and a `distance` `d` with initial value `v`
for the history of `o.field`.
"""
mutable struct DebugEntryChange <: DebugAction
    io::IO
    prefix::String
    field::Symbol
    distance::Any
    storage::StoreOptionsAction
    function DebugEntryChange(
        f::Symbol,
        d,
        a::StoreOptionsAction=StoreOptionsAction((f,)),
        prefix="Change of $f:",
        io::IO=stdout,
    )
        return new(io, prefix, f, d, a)
    end
    function DebugEntryChange(
        v::T where {T},
        f::Symbol,
        d,
        a::StoreOptionsAction=StoreOptionsAction((f,)),
        prefix="Change of $f:",
        io::IO=stdout,
    )
        update_storage!(a, Dict(f => v))
        return new(io, prefix, f, d, a)
    end
end
function (d::DebugEntryChange)(p::Problem, o::Options, i::Int)
    s = if (i > 0)
        (
            if has_storage(d.storage, d.field)
                d.prefix * string(
                    d.distance(
                        p, o, getproperty(o, d.field), get_storage(d.storage, d.field)
                    ),
                )
            else
                ""
            end
        )
    else
        ""
    end
    d.storage(p, o, i)
    print(d.io, s)
    return nothing
end

@doc raw"""
    DebugStoppingCriterion <: DebugAction

print the Reason provided by the stopping criterion. Usually this should be
empty, unless the algorithm stops.
"""
mutable struct DebugStoppingCriterion <: DebugAction
    io::IO
    DebugStoppingCriterion(io::IO=stdout) = new(io)
end
function (d::DebugStoppingCriterion)(::Problem, o::Options, i::Int)
    print(d.io, (i >= 0 || i == typemin(Int)) ? get_reason(o) : "")
    return nothing
end

@doc raw"""
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
function DebugFactory(a::Array{<:Any,1})
    # filter out every
    group = Array{DebugAction,1}()
    for s in filter(x -> !isa(x, Int) && x != :Stop, a) # filter ints and stop
        push!(group, DebugActionFactory(s))
    end
    dictionary = Dict{Symbol,DebugAction}()
    if length(group) > 0
        debug = DebugGroup(group)
        # filter ints
        e = filter(x -> isa(x, Int), a)
        if length(e) > 0
            debug = DebugEvery(debug, last(e))
        end
        dictionary[:All] = debug
    end
    if :Stop in a
        dictionary[:Stop] = DebugStoppingCriterion()
    end
    return dictionary
end
@doc raw"""
    DebugActionFactory(s)

create a [`DebugAction`](@ref) where

* a `String`yields the correspoinding divider
* a [`DebugAction`](@ref) is passed through
* a [`Symbol`] creates [`DebugEntry`](@ref) of that symbol, with the exceptions
  of `:Change`, `:Iterate`, `:Iteration`, and `:Cost`.
"""
DebugActionFactory(s::String) = DebugDivider(s)
DebugActionFactory(a::A) where {A<:DebugAction} = a
function DebugActionFactory(s::Symbol)
    if (s == :Change)
        return DebugChange()
    elseif (s == :Iteration)
        return DebugIteration()
    elseif (s == :Iterate)
        return DebugIterate()
    elseif (s == :Cost)
        return DebugCost()
    end
    return DebugEntry(s)
end
