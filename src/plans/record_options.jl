@doc raw"""
    RecordAction

A `RecordAction` is a small functor to record values.
The usual call is given by `(p,o,i) -> s` that performs the record based on
a [`Problem`](@ref) `p`, [`Options`](@ref) `o` and the current iterate `i`.

By convention `i<=0` is interpreted as "For Initialization only", i.e. only
initialize internal values, but not trigger any record, the same holds for
`i=typemin(Inf)` which is used to indicate `stop`, i.e. that the record is
called from within [`stop_solver!`](@ref) which returns true afterwards.

# Fields (assumed by subtypes to exist)
* `recorded_values` an `Array` of the recorded values.
"""
abstract type RecordAction <: AbstractOptionsAction end

@doc raw"""
    RecordOptions <: Options

append to any [`Options`](@ref) the decorator with record functionality,
Internally a `Dict`ionary is kept that stores a [`RecordAction`](@ref) for
several concurrent modes using a `Symbol` as reference.
The default mode is `:Iteration`, which is used to store information that is recorded during
the iterations. RecordActions might be added to `:Start` or `:Stop` to record values at the
beginning or for the stopping time point, respectively

The original options can still be accessed using the [`get_options`](@ref) function.

# Fields
* `options` – the options that are extended by debug information
* `recordDictionary` – a `Dict{Symbol,RecordAction}` to keep track of all
  different recorded values

# Constructors
    RecordOptions(o,dR)

construct record decorated [`Options`](@ref), where `dR` can be

* a [`RecordAction`](@ref), then it is stored within the dictionary at `:Iteration`
* an `Array` of [`RecordAction`](@ref)s, then it is stored as a
  `recordDictionary`(@ref) within the dictionary at `:All`.
* a `Dict{Symbol,RecordAction}`.
"""
mutable struct RecordOptions{O<:Options,TRD<:NamedTuple} <: Options
    options::O
    recordDictionary::TRD
    function RecordOptions{O}(o::O; kwargs...) where {O<:Options}
        return new{O,typeof(values(kwargs))}(o, values(kwargs))
    end
end
function RecordOptions(o::O, dR::RecordAction) where {O<:Options}
    return RecordOptions{O}(o; Iteration=dR)
end
function RecordOptions(o::O, dR::Dict{Symbol,<:RecordAction}) where {O<:Options}
    return RecordOptions{O}(o; dR...)
end
function RecordOptions(o::O, format::Vector{<:Any}) where {O<:Options}
    return RecordOptions{O}(o; RecordFactory(get_options(o), format)...)
end
function RecordOptions(o::O, s::Symbol) where {O<:Options}
    return RecordOptions{O}(o; Iteration=RecordFactory(get_options(o), s))
end

dispatch_options_decorator(::RecordOptions) = Val(true)

@doc """
    has_record(o::Options)

check whether the [`Options`](@ref)` o` are decorated with
[`RecordOptions`](@ref)
"""
has_record(::RecordOptions) = true
has_record(o::Options) = has_record(o, dispatch_options_decorator(o))
has_record(o::Options, ::Val{true}) = has_record(o.options)
has_record(::Options, ::Val{false}) = false

@doc """
    get_record_options(o::Options)

return the [`RecordOptions`](@ref) among the decorators from the [`Options`](@ref) `o`
"""
get_record_options(o::Options) = get_record_options(o, dispatch_options_decorator(o))
get_record_options(o::Options, ::Val{true}) = get_record_options(o.options)
get_record_options(::Options, ::Val{false}) = error("No Record decoration found")
get_record_options(o::RecordOptions) = o

@doc raw"""
    get_record_action(o::Options, s::Symbol)

return the action contained in the (first) [`RecordOptions`](@ref) decorator within the [`Options`](@ref) `o`.

"""
function get_record_action(o::Options, s::Symbol=:Iteration)
    if haskey(o.recordDictionary, s)
        return o.recordDictionary[s]
    else
        error("No record known for key :$s found")
    end
end
@doc raw"""
    get_record(o::Options, [,s=:Iteration])
    get_record(o::RecordOptions, [,s=:Iteration])

return the recorded values from within the [`RecordOptions`](@ref) `o` that where
recorded with respect to the `Symbol s` as an `Array`. The default refers to
any recordings during an `:Iteration`.

When called with arbitrary [`Options`](@ref), this method looks for the
[`RecordOptions`](@ref) decorator and calls `get_record` on the decorator.
"""
function get_record(o::RecordOptions, s::Symbol=:Iteration)
    return get_record(get_record_action(o, s))
end
function get_record(o::RecordOptions, s::Symbol, i...)
    return get_record(get_record_action(o, s), i...)
end
get_record(o::Options, s::Symbol=:Iteration) = get_record(get_record_options(o), s)

@doc raw"""
    get_record(r::RecordAction)

return the recorded values stored within a [`RecordAction`](@ref) `r`.
"""
get_record(r::RecordAction, i) = r.recorded_values
get_record(r::RecordAction) = r.recorded_values

"""
    get_index(ro::RecordOptions, s::Symbol)
    ro[s]

Get the recorded values for reording type `s`, see [`get_record`](@ref) for details.

    get_index(ro::RecordOptions, s::Symbol, i...)
    ro[s, i...]

Acces the recording type of type `s` and call its [`RecordAction`](@ref) with `[i...]`.
"""
getindex(ro::RecordOptions, s::Symbol) = get_record(ro, s)
getindex(ro::RecordOptions, s::Symbol, i...) = get_record_action(ro, s)[i...]

"""
    record_or_reset!(r,v,i)

either record (`i>0` and not `Inf`) the value `v` within the [`RecordAction`](@ref) `r`
or reset (`i<0`) the internal storage, where `v` has to match the internal
value type of the corresponding Recordaction.
"""
function record_or_reset!(r::RecordAction, v, i::Int)
    if i > 0
        push!(r.recorded_values, deepcopy(v))
    elseif i < 0 # reset if negative
        r.recorded_values = Array{typeof(v),1}()
    end
end

"""
    RecordGroup <: RecordAction

group a set of [`RecordAction`](@ref)s into one action, where the internal [`RecordAction`](@ref)s
act independently, but the results can be collected in a grouped fashion, i.e. tuples per calls of this group.
The enries can be later addressed either by index or semantic Symbols

# Constructors
    RecordGroup(g::Array{<:RecordAction, 1})

construct a group consisting of an Array of [`RecordAction`](@ref)s `g`,

    RecordGroup(g, symbols)

# Examples
    r = RecordGroup([RecordIteration(), RecordCost()])

A RecordGroup to record the current iteration and the cost. The cost can then be accessed using `get_record(r,2)` or `r[2]`.

    r = RecordGroup([RecordIteration(), RecordCost()], Dict(:Cost => 2))

A RecordGroup to record the current iteration and the cost, wich can then be accesed using `get_record(:Cost)` or `r[:Cost]`.

    r = RecordGroup([RecordIteration(), :Cost => RecordCost()])

A RecordGroup identical to the previous constructor, just a little easier to use.
"""
mutable struct RecordGroup <: RecordAction
    group::Array{RecordAction,1}
    indexSymbols::Dict{Symbol,Int}
    function RecordGroup(
        g::Array{<:RecordAction,1}, symbols::Dict{Symbol,Int}=Dict{Symbol,Int}()
    )
        if length(symbols) > 0
            if maximum(values(symbols)) > length(g)
                error(
                    "Index $(maximum(values(symbols))) must not be larger than number of elements ($(length(g))) in this RecordGroup.",
                )
            end
            if minimum(values(symbols)) < 1
                error("Index $(minimum(values(symbols))) nonpositive not allowed.")
            end
        end
        return new(g, symbols)
    end
    function RecordGroup(
        records::Vector{<:Union{<:RecordAction,Pair{Symbol,<:RecordAction}}}
    )
        g = Array{RecordAction,1}()
        si = Dict{Symbol,Int}()
        for i in 1:length(records)
            if records[i] isa RecordAction
                push!(g, records[i])
            else
                push!(g, records[i].second)
                push!(si, records[i].first => i)
            end
        end
        return RecordGroup(g, si)
    end
    RecordGroup() = new(Array{RecordAction,1}(), Dict{Symbol,Int}())
end

function (d::RecordGroup)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    for ri in d.group
        ri(p, o, i)
    end
end
@doc raw"""
    get_record(r::RecordGroup)

return an array of tuples, where each tuple is a recorded set, e.g. per iteration / record call.

    get_record(r::RecordGruop, i::Int)

return an array of values corresponding to the `i`th entry in this record group

    get_record(r::RecordGruop, s::Symbol)

return an array of recorded values with respect to the `s`, see [`RecordGroup`](@ref).

    get_record(r::RecordGroup, s1::Symbol, s2::Symbol,...)

return an array of tuples, where each tuple is a recorded set corresponding to the symbols `s1, s2,...` per iteration / record call.
"""
get_record(r::RecordGroup) = [zip(get_record.(r.group)...)...]
get_record(r::RecordGroup, i) = get_record(r.group[i])
get_record(r::RecordGroup, s::Symbol) = get_record(r.group[r.indexSymbols[s]])
function get_record(r::RecordGroup, s::NTuple{N,Symbol}) where {N}
    inds = getindex.(Ref(r.indexSymbols), s)
    return [zip(get_record.([r.group[i] for i in inds])...)...]
end

@doc raw"""
    getindex(r::RecordGroup, s::Symbol)
    r[s]
    getindex(r::RecordGroup, sT::NTuple{N,Symbol})
    r[sT]
    getindex(r::RecordGroup, i)
    r[i]

return an array of recorded values with respect to the `s`, the symbols from the tuple `sT` or the index `i`.
See [`get_record`](@ref get_record(r::RecordGroup)) for details.
"""
getindex(::RecordGroup, ::Any...)
getindex(r::RecordGroup, s::Symbol) = get_record(r, s)
getindex(r::RecordGroup, s::NTuple{N,Symbol}) where {N} = get_record(r, s)
getindex(r::RecordGroup, i) = get_record(r, i)

@doc raw"""
    RecordEvery <: RecordAction

record only every $i$th iteration.
Otherwise (optionally, but activated by default) just update internal tracking
values.

This method does not perform any record itself but relies on it's childrens methods
"""
mutable struct RecordEvery <: RecordAction
    record::RecordAction
    every::Int
    alwaysUpdate::Bool
    function RecordEvery(r::RecordAction, every::Int=1, alwaysUpdate::Bool=true)
        return new(r, every, alwaysUpdate)
    end
end
function (d::RecordEvery)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    if i <= 0
        d.record(p, o, i)
    elseif (rem(i, d.every) == 0)
        d.record(p, o, i)
    elseif d.alwaysUpdate
        d.record(p, o, 0)
    end
end
get_record(r::RecordEvery) = get_record(r.record)
get_record(r::RecordEvery, i) = get_record(r.record, i)
getindex(r::RecordEvery, i) = get_record(r, i)

#
# Special single ones
#
@doc raw"""
    RecordChange <: RecordAction

debug for the amount of change of the iterate (stored in `o.x` of the [`Options`](@ref))
during the last iteration.

# Additional Fields
* `storage` a [`StoreOptionsAction`](@ref) to store (at least) `o.x` to use this
  as the last value (to compute the change)
"""
mutable struct RecordChange <: RecordAction
    recorded_values::Array{Float64,1}
    storage::StoreOptionsAction
    function RecordChange(a::StoreOptionsAction=StoreOptionsAction((:x,)))
        return new(Array{Float64,1}(), a)
    end
    function RecordChange(x0, a::StoreOptionsAction=StoreOptionsAction((:x,)))
        update_storage!(a, Dict(:x => x0))
        return new(Array{Float64,1}(), a)
    end
end
function (r::RecordChange)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    record_or_reset!(
        r,
        has_storage(r.storage, :x) ? distance(p.M, o.x, get_storage(r.storage, :x)) : 0.0,
        i,
    )
    r.storage(p, o, i)
    return r.recorded_values
end

@doc raw"""
    RecordEntry{T} <: RecordAction

record a certain fields entry of type {T} during the iterates

# Fields
* `recorded_values` – the recorded Iterates
* `field` – Symbol the entry can be accessed with within [`Options`](@ref)

"""
mutable struct RecordEntry{T} <: RecordAction
    recorded_values::Array{T,1}
    field::Symbol
    RecordEntry{T}(f::Symbol) where {T} = new(Array{T,1}(), f)
end
RecordEntry(::T, f::Symbol) where {T} = RecordEntry{T}(f)
RecordEntry(d::DataType, f::Symbol) = RecordEntry{d}(f)
function (r::RecordEntry{T})(::Problem, o::Options, i) where {T}
    return record_or_reset!(r, getfield(o, r.field), i)
end

@doc raw"""
    RecordEntryChange{T} <: RecordAction

record a certain entries change during iterates

# Additional Fields
* `recorded_values` – the recorded Iterates
* `field` – Symbol the field can be accessed with within [`Options`](@ref)
* `distance` – function (p,o,x1,x2) to compute the change/distance between two values of the entry
* `storage` – a [`StoreOptionsAction`](@ref) to store (at least) `getproperty(o, d.field)`
"""
mutable struct RecordEntryChange <: RecordAction
    recorded_values::Vector{Float64}
    field::Symbol
    distance::Any
    storage::StoreOptionsAction
    function RecordEntryChange(f::Symbol, d, a::StoreOptionsAction=StoreOptionsAction((f,)))
        return new(Float64[], f, d, a)
    end
    function RecordEntryChange(
        v::T where {T}, f::Symbol, d, a::StoreOptionsAction=StoreOptionsAction((f,))
    )
        update_storage!(a, Dict(f => v))
        return new(Float64[], f, d, a)
    end
end
function (r::RecordEntryChange)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    record_or_reset!(
        r,
        if has_storage(r.storage, r.field)
            r.distance(p, o, getfield(o, r.field), get_storage(r.storage, r.field))
        else
            0.0
        end,
        i,
    )
    return r.storage(p, o, i)
end

@doc raw"""
    RecordIterate <: RecordAction

record the iterate

# Constructors
    RecordIterate(x0)

initialize the iterate record array to the type of `x0`, e.g. your initial data.

    RecordIterate(P)

initialize the iterate record array to the data type `T`.
"""
mutable struct RecordIterate{T} <: RecordAction
    recorded_values::Array{T,1}
    RecordIterate{T}() where {T} = new(Array{T,1}())
end
RecordIterate(::T) where {T} = RecordIterate{T}()
function RecordIterate()
    return throw(
        ErrorException(
            "The iterate's data type has to be provided, i.e. RecordIterate(x0)."
        ),
    )
end

function (r::RecordIterate{T})(::Problem, o::Options, i) where {T}
    return record_or_reset!(r, o.x, i)
end

@doc raw"""
    RecordIteration <: RecordAction

record the current iteration
"""
mutable struct RecordIteration <: RecordAction
    recorded_values::Array{Int,1}
    RecordIteration() = new(Array{Int,1}())
end
function (r::RecordIteration)(::P, ::O, i::Int) where {P<:Problem,O<:Options}
    return record_or_reset!(r, i, i)
end

@doc raw"""
    RecordCost <: RecordAction

record the current cost function value, see [`get_cost`](@ref).
"""
mutable struct RecordCost <: RecordAction
    recorded_values::Array{Float64,1}
    RecordCost() = new(Array{Float64,1}())
end
function (r::RecordCost)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    return record_or_reset!(r, get_cost(p, o.x), i)
end

@doc raw"""
    RecordFactory(o::Options, a)

given an array of `Symbol`s and [`RecordAction`](@ref)s and `Ints`

* The symbol `:Cost` creates a [`RecordCost`](@ref)
* The symbol `:iteration` creates a [`RecordIteration`](@ref)
* The symbol `:Change` creates a [`RecordChange`](@ref)
* any other symbol creates a [`RecordEntry`](@ref) of the corresponding field in [`Options`](@ref)
* any [`RecordAction`](@ref) is directly included
* an semantic pair `:symbol => RecordAction` is directly included
* an Integer `k` introduces that record is only performed every `k`th iteration
"""
function RecordFactory(o::Options, a::Array{<:Any,1})
    # filter out every
    group = Array{Union{<:RecordAction,Pair{Symbol,<:RecordAction}},1}()
    for s in filter(x -> !isa(x, Int), a) # for all that are not integers or stopping criteria
        if s isa Symbol
            push!(group, s => RecordActionFactory(o, s))
        elseif s isa Pair{<:Symbol,<:RecordAction}
            push!(group, s)
        else
            push!(group, RecordActionFactory(o, s))
        end
    end
    record = RecordGroup(group)
    # filter ints
    e = filter(x -> isa(x, Int), a)
    if length(e) > 0
        record = RecordEvery(record, last(e))
    end
    return (; Iteration=record)
end
RecordFactory(o::Options, s::Symbol) = RecordActionFactory(o, s)

@doc raw"""
    RecordActionFactory(s)

create a [`RecordAction`](@ref) where

* a [`RecordAction`](@ref) is passed through
* a [`Symbol`] creates [`RecordEntry`](@ref) of that symbol, with the exceptions
  of `:Change`, `:Iterate`, `:Iteration`, and `:Cost`.
"""
RecordActionFactory(::Options, a::RecordAction) = a
RecordActionFactory(::Options, sa::Pair{Symbol,<:RecordAction}) = sa
function RecordActionFactory(o::Options, s::Symbol)
    if (s == :Change)
        return RecordChange()
    elseif (s == :Iteration)
        return RecordIteration()
    elseif (s == :Iterate)
        return RecordIterate(o.x)
    elseif (s == :Cost)
        return RecordCost()
    end
    return RecordEntry(getfield(o, s), s)
end
