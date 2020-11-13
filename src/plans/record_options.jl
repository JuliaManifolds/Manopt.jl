export RecordOptions
export RecordAction
export RecordGroup, RecordEvery
export RecordChange, RecordCost, RecordIterate, RecordIteration
export RecordEntry, RecordEntryChange
export get_record, has_record
export RecordActionFactory, RecordFactory

#
#
# record Options Decorator
#
#
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
* `recordedValues` an `Array` of the recorded values.
"""
abstract type RecordAction <: AbstractOptionsAction end

@doc raw"""
    RecordOptions <: Options

append to any [`Options`](@ref) the decorator with record functionality,
Internally a `Dict`ionary is kept that stores a [`RecordAction`](@ref) for
several occasions using a `Symbol` as reference.
The default occasion is `:All` and for example solvers join this field with
`:Start`, `:Step` and `:Stop` at the beginning, every iteration or the
end of the algorithm, respectively

The original options can still be accessed using the [`get_options`](@ref) function.

# Fields
* `options` – the options that are extended by debug information
* `recordDictionary` – a `Dict{Symbol,RecordAction}` to keep track of all
  different recorded values

# Constructors
    RecordOptions(o,dR)

construct record decorated [`Options`](@ref), where `dR` can be

* a [`RecordAction`](@ref), then it is stored within the dictionary at `:All`
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
function RecordOptions(o::O, dR::D) where {O<:Options,D<:RecordAction}
    return RecordOptions{O}(o; All=dR)
end
function RecordOptions(o::O, dR::Array{<:RecordAction,1}) where {O<:Options}
    return RecordOptions{O}(o; All=RecordGroup(dR))
end
function RecordOptions(o::O, dR::Dict{Symbol,<:RecordAction}) where {O<:Options}
    return RecordOptions{O}(o; dR...)
end
function RecordOptions(o::O, format::Vector{<:Any}) where {O<:Options}
    return RecordOptions{O}(o; RecordFactory(get_options(o), format)...)
end

dispatch_options_decorator(o::RecordOptions) = Val(true)

"""
    has_record(o)

check whether the [`Options`](@ref)` o` are decorated with
[`RecordOptions`](@ref)
"""
has_record(o::RecordOptions) = true
has_record(o::Options) = has_record(o, dispatch_options_decorator(o))
has_record(o::Options, ::Val{true}) = has_record(o.options)
has_record(o::Options, ::Val{false}) = false

# default - stored in the recordedValues field of the RecordAction
@doc raw"""
    get_record(o[,s=:Step])

return the recorded values from within the [`RecordOptions`](@ref) `o` that where
recorded with respect to the `Symbol s` as an `Array`. The default refers to
any recordings during an Iteration represented by the Symbol `:Step`
"""
function get_record(o::RecordOptions, s::Symbol=:Step)
    if haskey(o.recordDictionary, s)
        return get_record(o.recordDictionary[s])
    elseif haskey(o.recordDictionary, :All)
        return get_record(o.recordDictionary[:All])
    else
        error("No record known for key found, since neither :$s nor :All are present.")
    end
end
get_record(o::Options, s::Symbol=:Step) = get_record(o, s, dispatch_options_decorator(o))
get_record(o::Options, s, ::Val{true}) = get_record(o.options, s)
get_record(o::Options, s, ::Val{false}) = error("No Record decoration found")

@doc raw"""
    get_record(r)

return the recorded values stored within a [`RecordAction`](@ref) `r`.
"""
get_record(r::R) where {R<:RecordAction} = r.recordedValues

"""
    record_or_eset!(r,v,i)

either record (`i>0` and not `Inf`) the value `v` within the [`RecordAction`](@ref) `r`
or reset (`i<0`) the internal storage, where `v` has to match the internal
value type of the corresponding Recordaction.
"""
function record_or_eset!(r::R, v, i::Int) where {R<:RecordAction}
    if i > 0
        push!(r.recordedValues, deepcopy(v))
    elseif i < 0 && i > typemin(Int) # reset if negative but not stop indication
        r.recordedValues = Array{typeof(v),1}()
    end
end
"""
    RecordGroup <: RecordAction

group a set of [`RecordAction`](@ref)s into one action, where the internal prints
are removed by default and the resulting strings are concatenated

# Constructor
    RecordGroup(g)

construct a group consisting of an Array of [`RecordAction`](@ref)s `g`,
that are recording `en bloque`; the method does not perform any record itself,
but keeps an array of records. Accessing these yields a `Tuple` of the recorded
values per iteration
"""
mutable struct RecordGroup <: RecordAction
    group::Array{RecordAction,1}
    RecordGroup(g::Array{<:RecordAction,1}) = new(g)
    RecordGroup() = new(Array{RecordAction,1}())
end
function (d::RecordGroup)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    for ri in d.group
        ri(p, o, i)
    end
end
get_record(r::RecordGroup) = [zip(get_record.(r.group)...)...]

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
    recordedValues::Array{Float64,1}
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
    record_or_eset!(
        r,
        has_storage(r.storage, :x) ? distance(p.M, o.x, get_storage(r.storage, :x)) : 0.0,
        i,
    )
    r.storage(p, o, i)
    return r.recordedValues
end

@doc raw"""
    RecordEntry{T} <: RecordAction

record a certain fields entry of type {T} during the iterates

# Fields
* `recordedValues` – the recorded Iterates
* `field` – Symbol the entry can be accessed with within [`Options`](@ref)

"""
mutable struct RecordEntry{T} <: RecordAction
    recordedValues::Array{T,1}
    field::Symbol
    RecordEntry{T}(f::Symbol) where {T} = new(Array{T,1}(), f)
end
RecordEntry(e::T, f::Symbol) where {T} = RecordEntry{T}(f)
RecordEntry(d::DataType, f::Symbol) = RecordEntry{d}(f)
function (r::RecordEntry{T})(p::Pr, o::O, i::Int) where {T,Pr<:Problem,O<:Options}
    return record_or_eset!(r, getfield(o, r.field), i)
end

@doc raw"""
    RecordEntryChange{T} <: RecordAction

record a certain entries change during iterates

# Additional Fields
* `recordedValues` – the recorded Iterates
* `field` – Symbol the field can be accessed with within [`Options`](@ref)
* `distance` – function (p,o,x1,x2) to compute the change/distance between two values of the entry
* `storage` – a [`StoreOptionsAction`](@ref) to store (at least) `getproperty(o, d.field)`
"""
mutable struct RecordEntryChange <: RecordAction
    recordedValues::Vector{Float64}
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
    record_or_eset!(
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
    recordedValues::Array{T,1}
    RecordIterate{T}() where {T} = new(Array{T,1}())
end
RecordIterate(::T) where {T} = RecordIterate{T}()
function RecordIterate()
    return throw(ErrorException("The iterate's data type has to be provided, i.e. RecordIterate(x0)."))
end

function (r::RecordIterate{T})(::Problem, o::Options, i) where {T}
    return record_or_eset!(r, o.x, i)
end

@doc raw"""
    RecordIteration <: RecordAction

record the current iteration
"""
mutable struct RecordIteration <: RecordAction
    recordedValues::Array{Int,1}
    RecordIteration() = new(Array{Int,1}())
end
function (r::RecordIteration)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    return record_or_eset!(r, i, i)
end

@doc raw"""
    RecordCost <: RecordAction

record the current cost function value, see [`get_cost`](@ref).
"""
mutable struct RecordCost <: RecordAction
    recordedValues::Array{Float64,1}
    RecordCost() = new(Array{Float64,1}())
end
function (r::RecordCost)(p::P, o::O, i::Int) where {P<:Problem,O<:Options}
    return record_or_eset!(r, get_cost(p, o.x), i)
end

@doc raw"""
    RecordFactory(a)

given an array of `Symbol`s and [`RecordAction`](@ref)s and `Ints`

* The symbol `:Cost` creates a [`RecordCost`](@ref)
* The symbol `:iteration` creates a [`RecordIteration`](@ref)
* The symbol `:Change` creates a [`RecordChange`](@ref)
* any other symbol creates a [`RecordEntry`](@ref) of the corresponding field in [`Options`](@ref)
* any [`RecordAction`](@ref) is directly included
* an Integer `k` introduces that record is only performed every `k`th iteration
"""
function RecordFactory(o::O, a::Array{<:Any,1}) where {O<:Options}
    # filter out every
    group = Array{RecordAction,1}()
    for s in filter(x -> !isa(x, Int), a) # filter ints and stop
        push!(group, RecordActionFactory(o, s))
    end
    record = RecordGroup(group)
    # filter ints
    e = filter(x -> isa(x, Int), a)
    if length(e) > 0
        record = RecordEvery(record, last(e))
    end
    return (; All=record)
end
@doc raw"""
    RecordActionFactory(s)

create a [`RecordAction`](@ref) where

* a [`RecordAction`](@ref) is passed through
* a [`Symbol`] creates [`RecordEntry`](@ref) of that symbol, with the exceptions
  of `:Change`, `:Iterate`, `:Iteration`, and `:Cost`.
"""
RecordActionFactory(o::O, a::A) where {O<:Options,A<:RecordAction} = a
function RecordActionFactory(o::O, s::Symbol) where {O<:Options}
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
