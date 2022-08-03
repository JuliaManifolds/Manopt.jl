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

dispatch_options_decorator(::DebugOptions) = Val(true)

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
function (d::DebugGroup)(p::Problem, o::Options, i)
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
function (d::DebugEvery)(p::Problem, o::Options, i)
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
    DebugChange()

debug for the amount of change of the iterate (stored in `get_iterate(o)` of the [`Options`](@ref))
during the last iteration. See [`DebugEntryChange`](@ref) for the general case

# Keyword Parameters
* `storage` – (`StoreOptionsAction( (:Iterate,) )`) – (eventually shared) the storage of the previous action
* `prefix` – (`"Last Change:"`) prefix of the debug output (ignored if you set `format`)
* `io` – (`stdout`) default steream to print the debug to.
* `format` - ( `"$prefix %f"`) format to print the output using an sprintf format.
"""
mutable struct DebugChange <: DebugAction
    io::IO
    format::String
    storage::StoreOptionsAction
    function DebugChange(;
        storage::StoreOptionsAction=StoreOptionsAction((:Iterate,)),
        io::IO=stdout,
        prefix::String="Last Change: ",
        format::String="$(prefix)%f",
    )
        return new(io, format, storage)
    end
end
@deprecate DebugChange(a::StoreOptionsAction, pre::String="Last Change: ", io::IO=stdout) DebugChange(;
    storage=a, prefix=pre, io=io
)
function (d::DebugChange)(p::Problem, o::Options, i)
    (i > 0) && Printf.format(
        d.io,
        Printf.Format(d.format),
        distance(p.M, get_iterate(o), get_storage(d.storage, :Iterate)),
    )
    d.storage(p, o, i)
    return nothing
end
@doc raw"""
    DebugGradientChange()

debug for the amount of change of the gradient (stored in `get_gradient(o)` of the [`Options`](@ref) `o`)
during the last iteration. See [`DebugEntryChange`](@ref) for the general case

# Keyword Parameters
* `storage` – (`StoreOptionsAction( (:Gradient,) )`) – (eventually shared) the storage of the previous action
* `prefix` – (`"Last Change:"`) prefix of the debug output (ignored if you set `format`)
* `io` – (`stdout`) default steream to print the debug to.
* `format` - ( `"$prefix %f"`) format to print the output using an sprintf format.
"""
mutable struct DebugGradientChange <: DebugAction
    io::IO
    format::String
    storage::StoreOptionsAction
    function DebugGradientChange(;
        storage::StoreOptionsAction=StoreOptionsAction((:Gradient,)),
        io::IO=stdout,
        prefix::String="Last Change: ",
        format::String="$(prefix)%f",
    )
        return new(io, format, storage)
    end
end
function (d::DebugGradientChange)(p::Problem, o::Options, i)
    (i > 0) && Printf.format(
        d.io,
        Printf.Format(d.format),
        distance(p.M, get_gradient(o), get_storage(d.storage, :Gradient)),
    )
    d.storage(p, o, i)
    return nothing
end
@doc raw"""
    DebugIterate <: DebugAction

debug for the current iterate (stored in `get_iterate(o)`).

# Constructor
    DebugIterate()

# Parameters

* `io` – (`stdout`) default steream to print the debug to.
* `long::Bool` whether to print `x:` or `current iterate`
"""
mutable struct DebugIterate <: DebugAction
    io::IO
    format::String
    function DebugIterate(;
        io::IO=stdout,
        long::Bool=false,
        prefix=long ? "current iterate:" : "x:",
        format="$prefix %s",
    )
        return new(io, format)
    end
end
@deprecate DebugIterate(io::IO, long::Bool=false) DebugIterate(; io=io, long=long)
function (d::DebugIterate)(::Problem, o::Options, i::Int)
    (i > 0) && Printf.format(d.io, Printf.Format(d.format), get_iterate(o))
    return nothing
end

@doc raw"""
    DebugIteration <: DebugAction

# Constructor

    DebugIteration()

# Keyword parameters

* `format` - (`"# %-6d"`) format to print the output using an sprintf format.
* `io` – (`stdout`) default steream to print the debug to.

debug for the current iteration (prefixed with `#` by )
"""
mutable struct DebugIteration <: DebugAction
    io::IO
    format::String
    DebugIteration(; io::IO=stdout, format="# %-6d") = new(io, format)
end
@deprecate DebugIteration(io::IO) DebugIteration(; io=io)
function (d::DebugIteration)(::Problem, ::Options, i::Int)
    (i == 0) && print(d.io, "Initial ")
    (i > 0) && Printf.format(d.io, Printf.Format(d.format), i)
    return nothing
end

@doc raw"""
    DebugCost <: DebugAction

print the current cost function value, see [`get_cost`](@ref).

# Constructors
    DebugCost()

# Parameters

* `format` - (`"$prefix %f"`) format to print the output using sprintf and a prefix (see `long`).
* `io` – (`stdout`) default steream to print the debug to.
* `long` - (`false`) short form to set the format to `F(x):` (default) or `current cost: ` and the cost
"""
mutable struct DebugCost <: DebugAction
    io::IO
    format::String
    function DebugCost(;
        long::Bool=false, io::IO=stdout, format=long ? "current cost: %f" : "F(x): %f"
    )
        return new(io, format)
    end
end
@deprecate DebugCost(pre::String) DebugCost(; format="$pre %f")
@deprecate DebugCost(pre::String, io::IO) DebugCost(; format="$pre %f", io=op)
@deprecate DebugCost(long::Bool, io::IO) DebugCost(; long=long, io=io)
function (d::DebugCost)(p::Problem, o::Options, i::Int)
    (i >= 0) && Printf.format(d.io, Printf.Format(d.format), get_cost(p, get_iterate(o)))
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
function (d::DebugDivider)(::Problem, ::Options, i::Int)
    print(d.io, (i >= 0) ? d.divider : "")
    return nothing
end

@doc raw"""
    DebugEntry <: RecordAction

print a certain fields entry of type {T} during the iterates, where a `format` can be
specified how to print the entry.

# Addidtional Fields
* `field` – Symbol the entry can be accessed with within [`Options`](@ref)

# Constructor

    DebugEntry(f[, prefix="$f:", format = "$prefix %s", io=stdout])

"""
mutable struct DebugEntry <: DebugAction
    io::IO
    format::String
    field::Symbol
    function DebugEntry(f::Symbol; prefix="$f:", format="$prefix %s", io::IO=stdout)
        return new(io, format, f)
    end
end
@deprecate DebugEntry(f, prefix="$f:", io=stdout) DebugEntry(f; prefix=prefix, io=io)

function (d::DebugEntry)(::Problem, o::Options, i)
    (i >= 0) && Printf.format(d.io, Printf.Format(d.format), getfield(o, d.field))
    return nothing
end

@doc raw"""
    DebugEntryChange{T} <: DebugAction

print a certain entries change during iterates

# Additional Fields
* `print` – (`print`) function to print the result
* `prefix` – (`"Change of :Iterate"`) prefix to the print out
* `format` – (`"$prefix %e"`) format to print (uses the `prefix by default and scientific notation)
* `field` – Symbol the field can be accessed with within [`Options`](@ref)
* `distance` – function (p,o,x1,x2) to compute the change/distance between two values of the entry
* `storage` – a [`StoreOptionsAction`](@ref) to store the previous value of `:f`

# Constructors

    DebugEntryChange(f,d)

# Keyword arguments

- `io` (`stdout`) an `IOStream`
- `prefix` (`"Change of $f"`)
- `storage` (`StoreOptionsAction((f,))`) a [`StoreOptionsAction`](@ref)
- `initial_value` an initial value for the change of `o.field`.
- `format` – (`"$prefix %e"`) format to print the change

"""
mutable struct DebugEntryChange <: DebugAction
    distance::Any
    field::Symbol
    format::String
    io::IO
    storage::StoreOptionsAction
    function DebugEntryChange(
        f::Symbol,
        d;
        storage::StoreOptionsAction=StoreOptionsAction((f,)),
        prefix::String="Change of $f:",
        format::String="$prefix%s",
        io::IO=stdout,
        initial_value::T where {T}=NaN,
    )
        if !isa(initial_value, Number) || !isnan(initial_value) #set initial value
            update_storage!(storage, Dict(f => initial_value))
        end
        return new(d, f, format, io, storage)
    end
end

function (d::DebugEntryChange)(p::Problem, o::Options, i::Int)
    if i == 0
        # on init if field not present -> generate
        !has_storage(d.storage, d.field) && d.storage(p, o, i)
        return nothing
    end
    x = get_storage(d.storage, d.field)
    v = d.distance(p, o, getproperty(o, d.field), x)
    Printf.format(d.io, Printf.Format(d.format), v)
    d.storage(p, o, i)
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
    DebugWarnIfCostIncreases <: DebugAction

print a warning if the cost increases.

Note that this provides an additional warning for gradient descent
with its default constant step size.

# Constructor
    DebugWarnIfCostIncreases(warn=:Once; tol=1e-13)

Initialize the warning to warning level (`:Once`) and introduce a tolerance for the test of `1e-13`.

The `warn` level can be set to `:Once` to only warn the first time the cost increases,
to `:Always` to report an increase every time it happens, and it can be set to `:No`
to deactivate the warning, then this [`DebugAction`](@ref) is inactive.
All other symbols are handled as if they were `:Always:`
"""
mutable struct DebugWarnIfCostIncreases <: DebugAction
    # store if we need to warn – :Once, :Always, :No, where all others are handled
    # the same as :Always
    status::Symbol
    old_cost::Float64
    tol::Float64
    DebugWarnIfCostIncreases(warn::Symbol=:Once; tol=1e-13) = new(warn, Float64(Inf), tol)
end
function (d::DebugWarnIfCostIncreases)(p::Problem, o::Options, i::Int)
    if d.status !== :No
        cost = get_cost(p, get_iterate(o))
        if cost > d.old_cost + d.tol
            # Default case in Gradient Descent, include a tipp
            @warn """The cost increased.
            At iteration #$i the cost increased from $(d.old_cost) to $(cost)."""
            if o isa GradientDescentOptions && o.stepsize isa ConstantStepsize
                @warn """You seem to be running a `gradient_decent` with the default `ConstantStepsize`.
                For ease of use, this is set as the default, but might not converge.
                Maybe consider to use `ArmijoLinesearch` (if applicable) or use
                `ConstantStepsize(value)` with a `value` less than $(get_last_stepsize(p,o,i))."""
            end
            if d.status === :Once
                @warn "Further warnings will be supressed, use DebugWarnIfCostIncreases(:Always) to get all warnings."
                d.status = :No
            end
        else
            d.old_cost = min(d.old_cost, cost)
        end
    end
    return nothing
end

@doc raw"""
    DebugWarnIfCostNotFinite <: DebugAction

A debug to see when a field (value or array within the Options is or contains values
that are not finite, for example `Inf` or `Nan`.

# Constructor
    DebugWarnIfCostNotFinite(field::Symbol, warn=:Once)

Initialize the warning to warn `:Once`.

This can be set to `:Once` to only warn the first time the cost is Nan.
It can also be set to `:No` to deactivate the warning, but this makes this Action also useless.
All other symbols are handled as if they were `:Always:`
"""
mutable struct DebugWarnIfCostNotFinite <: DebugAction
    status::Symbol
    DebugWarnIfCostNotFinite(warn::Symbol=:Once) = new(warn)
end
function (d::DebugWarnIfCostNotFinite)(p::Problem, o::Options, i::Int)
    if d.status !== :No
        cost = get_cost(p, get_iterate(o))
        if !isfinite(cost)
            @warn """The cost is not finite.
            At iteration #$i the cost evaluated to $(cost)."""
            if d.status === :Once
                @warn "Further warnings will be supressed, use DebugWarnIfCostNotFinite(:Always) to get all warnings."
                d.status = :No
            end
        end
    end
    return nothing
end

@doc raw"""
    DebugWarnIfFieldNotFinite <: DebugAction

A debug to see when a field from the options is not finite, for example `Inf` or `Nan`

# Constructor
    DebugWarnIfFieldNotFinite(field::Symbol, warn=:Once)

Initialize the warning to warn `:Once`.

This can be set to `:Once` to only warn the first time the cost is Nan.
It can also be set to `:No` to deactivate the warning, but this makes this Action also useless.
All other symbols are handled as if they were `:Always:`

# Example
    DebugWaranIfFieldNotFinite(:gradient)

Creates a [`DebugAction`] to track whether the gradient does not get `Nan` or `Inf`.
"""
mutable struct DebugWarnIfFieldNotFinite <: DebugAction
    status::Symbol
    field::Symbol
    DebugWarnIfFieldNotFinite(field::Symbol, warn::Symbol=:Once) = new(warn, field)
end
function (d::DebugWarnIfFieldNotFinite)(::Problem, o::Options, i::Int)
    if d.status !== :No
        v = getproperty(o, d.field)
        if !all(isfinite.(v))
            @warn """The field o.$(d.field) is or contains values that are not finite.
            At iteration #$i it evaluated to $(v)."""
            if d.status === :Once
                @warn "Further warnings will be supressed, use DebugWaranIfFieldNotFinite(:$(d.field), :Always) to get all warnings."
                d.status = :No
            end
        end
    end
    return nothing
end

@doc raw"""
    DebugFactory(a)

given an array of `Symbol`s, `String`s [`DebugAction`](@ref)s and `Ints`

* The symbol `:Stop` creates an entry of to display the stopping criterion at the end
  (`:Stop => DebugStoppingCriterion()`), for further symbols see [`DebugActionFactory`](@ref DebugActionFactory(::Symbol))
* Tuples of a symbol and a string can be used to also specify a format, see [`DebugActionFactory`](@ref DebugActionFactory(::Tuple{Symbol,String}))
* any string creates a [`DebugDivider`](@ref)
* any [`DebugAction`](@ref) is directly included
* an Integer `k`introduces that debug is only printed every `k`th iteration

# Return value

This function returns a dictionary with an entry `:All` containing one general [`DebugAction`](@ref),
possibly a [`DebugGroup`](@ref) of entries.
It might contain an entry `:Start`, `:Step`, `:Stop` with an action (each) to specify what to do
at the start, after a step or at the end of an Algorithm, respectively. On all three occastions the `:All` action is exectued.
Note that only the `:Stop` entry is actually filled when specifying the `:Stop` symbol.

# Example

The array

```
[:Iterate, " | ", :Cost, :Stop, 10]
```

Adds a group to `:All` of three actions ([`DebugIteration`](@ref), [`DebugDivider`](@ref) with `" | "` to display, [`DebugCost`](@ref))
as a [`DebugGroup`](@ref) inside an [`DebugEvery`](@ref) to only be executed every 10th iteration.
It also adds the [`DebugStoppingCriterion`](@ref) to the `:Stop` entry of the dictionary.
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
* a `Tuple{Symbol,String}` creates a [`DebugEntry`](@ref) of that symbol where the String specifies the format.
"""
DebugActionFactory(s::String) = DebugDivider(s)
DebugActionFactory(a::A) where {A<:DebugAction} = a
"""
    DebugActionFactory(s::Symbol)

Convert certain Symbols in the `debug=[ ... ]` vector to [`DebugAction`](@ref)s
Currently the following ones are done.
Note that the Shortcut symbols should all start with a capital letter.

* `:Cost` creates a [`DebugCost`](@ref)
* `:Change` creates a [`DebugChange`](@ref)
* `:GradientChange` creates a [`DebugGradientChange`](@ref)
* `:Iterate` creates a [`DebugIterate`](@ref)
* `:Iteration` creates a [`DebugIteration`](@ref)
* `:Stepsize` creates a [`DebugStepsize`](@ref)
* `:WarnCost` creates a [`DebugWarnIfCostNotFinite`](@ref)
* `:WarnGradient` creates a [`DebugWarnIfFieldNotFinite`](@ref) for the `:gradient`.

any other symbol creates a `DebugEntry(s)` to print the entry (o.:s) from the options.
"""
function DebugActionFactory(s::Symbol)
    (s == :Cost) && return DebugCost()
    (s == :Change) && return DebugChange()
    (s == :GradientChange) && return DebugGradientChange()
    (s == :Iterate) && return DebugIterate()
    (s == :Iteration) && return DebugIteration()
    (s == :Stepsize) && return DebugStepsize()
    (s == :WarnCost) && return DebugWarnIfCostNotFinite()
    (s == :WarnGradient) && return DebugWarnIfFieldNotFinite(:gradient)
    return DebugEntry(s)
end
"""
    DebugActionFactory(t::Tuple{Symbol,String)

Convert certain Symbols in the `debug=[ ... ]` vector to [`DebugAction`](@ref)s
Currently the following ones are done, where the string in `t[2]` is passed as the
`format` the corresponding debug.
Note that the Shortcut symbols `t[1]` should all start with a capital letter.

* `:Cost` creates a [`DebugCost`](@ref)
* `:Change` creates a [`DebugChange`](@ref)
* `:GradientChange` creates a [`DebugGradientChange`](@ref)
* `:Iterate` creates a [`DebugIterate`](@ref)
* `:Iteration` creates a [`DebugIteration`](@ref)
* `:Stepsize` creates a [`DebugStepsize`](@ref)

any other symbol creates a `DebugEntry(s)` to print the entry (o.:s) from the options.
"""
function DebugActionFactory(t::Tuple{Symbol,String})
    (t[1] == :Change) && return DebugChange(; format=t[2])
    (t[1] == :GradientChange) && return DebugGradientChange(; format=t[2])
    (t[1] == :Iteration) && return DebugIteration(; format=t[2])
    (t[1] == :Iterate) && return DebugIterate(; format=t[2])
    (t[1] == :Cost) && return DebugCost(; format=t[2])
    (t[1] == :Stepsize) && return DebugStepsize(; format=t[2])
    return DebugEntry(t[1]; format=t[2])
end
