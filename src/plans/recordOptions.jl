export RecordOptions
export RecordAction
export RecordGroup, RecordEvery
export RecordChange, RecordCost, RecordIterate, RecordIteration
export RecordEntry, RecordEntryChange
export getRecord, hasRecord
export RecordActionFactory, RecordFactory

#
#
# record Options Decorator
#
#
@doc doc"""
    RecordAction

A `RecordAction` is a small functor to record values.
The usual call is given by `(p,o,i) -> s` that performs the record based on
a [`Problem`](@ref) `p`, [`Options`](@ref) `o` and the current iterate `i`.

By convention `i<=0` is interpreted as "For Initialization only", i.e. only
initialize internal values, but not trigger any record, the same holds for
`i=typemin(Inf)` which is used to indicate `stop`, i.e. that the record is
called from within [`stopSolver!`](@ref) which returns true afterwards.

# Fields (assumed by subtypes to exist)
* `recordedValues` an `Array` of the recorded values.
""" 
abstract type RecordAction <: Action end

@doc doc"""
    RecordOptions <: Options

append to any [`Options`](@ref) the decorator with record functionality,
Internally a `Dict`ionary is kept that stores a [`RecordAction`](@ref) for
several occasions using a `Symbol` as reference.
The default occasion is `:All` and for example solvers join this field with
`:Start`, `:Step` and `:Stop` at the beginning, every iteration or the
end of the algorithm, respectively

The original options can still be accessed using the [`getOptions`](@ref) function.

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
mutable struct RecordOptions{O <: Options} <: Options
    options::O
    recordDictionary::Dict{Symbol, <: RecordAction}
    RecordOptions{O}(o::O, dR::Dict{Symbol, <: RecordAction}) where {O <: Options} = new(o,dR)
end
RecordOptions(o::O, dR::D) where {O <: Options, D <: RecordAction} = RecordOptions{O}(o,Dict(:All => dR))
RecordOptions(o::O, dR::Array{ <: RecordAction,1}) where {O <: Options} = RecordOptions{O}(o,Dict(:All => RecordGroup(dR)))
RecordOptions(o::O, dR::Dict{Symbol, <: RecordAction}) where {O <: Options} = RecordOptions{O}(o,dR)
RecordOptions(o::O, format::Array{<:Any,1}) where {O <: Options} = RecordOptions{O}(o, RecordFactory(getOptions(o),format))

@traitimpl IsOptionsDecorator{RecordOptions}

"""
    hasRecord(o)

check whether the [`Options`](@ref)` o` are decorated with
[`RecordOptions`](@ref)
"""
hasRecord(o::RecordOptions) = true
@traitfn hasRecord(o::O) where {O <: Options; IsOptionsDecorator{O}} = hasRecord(o.options)
@traitfn hasRecord(o::O) where {O <: Options; !IsOptionsDecorator{O}} = false

# default - stored in the recordedValues field of the RecordAction
@doc doc"""
    getRecord(o[,s=:Step])

return the recorded values from within the [`RecordOptions`](@ref) `o` that where
recorded with respect to the `Symbol s` as an `Array`. The default refers to
any recordings during an Iteration represented by the Symbol `:Step`
"""
function getRecord(o::RecordOptions,s::Symbol=:Step)
    if haskey(o.recordDictionary,s)
        return getRecord(o.recordDictionary[s])
    elseif haskey(o.recordDictionary,:All)
        return getRecord(o.recordDictionary[:All])
    else
        error("No record known for key found, since neither :$s nor :All are present.")
    end
end
@traitfn getRecord(o::O, s::Symbol=:Step) where {O <: Options; IsOptionsDecorator{O}} = getRecord(o.options,s)
@traitfn getRecord(o::O, s::Symbol=:Step) where {O <: Options; !IsOptionsDecorator{O}} = error("No Record decoration found")

@doc doc"""
    getRecord(r)

return the recorded values stored within a [`RecordAction`](@ref) `r`.
"""
getRecord(r::R) where {R <: RecordAction} = r.recordedValues

"""
    recordOrReset!(r,v,i)

either record (`i>0` and not `Inf`) the value `v` within the [`RecordAction`](@ref) `r`
or reset (`i<0`) the internal storage, where `v` has to match the internal
value type of the corresponding Recordaction. 
"""
function recordOrReset!(r::R,v,i::Int) where {R <: RecordAction}
    if i > 0
        push!(r.recordedValues,v)
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
function (d::RecordGroup)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    for ri in d.group
        ri(p,o,i)
    end
end
getRecord(r::RecordGroup) = [zip( getRecord.(r.group)...)...]

@doc doc"""
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
    RecordEvery(r::RecordAction,every::Int=1,alwaysUpdate::Bool=true) = new(r,every,alwaysUpdate)
end
function (d::RecordEvery)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    if i<=0
        d.record(p,o,i)
    elseif (rem(i,d.every)==0)
        d.record(p,o,i)
    elseif d.alwaysUpdate
        d.record(p,o,0)
    end
end
getRecord(r::RecordEvery) = getRecord(r.record)
#
# Special single ones
#
@doc doc"""
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
    RecordChange(a::StoreOptionsAction=StoreOptionsAction( (:x,) ) ) = new(Array{Float64,1}(),a)
    function RecordChange(x0::MPoint,
            a::StoreOptionsAction=StoreOptionsAction( (:x,) ),
        )
        updateStorage!(a,Dict(:x=>x0))
        return new(Array{Float64,1}(),a)
    end
end
function (r::RecordChange)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    recordOrReset!(r,
        hasStorage(r.storage, :x) ? distance(p.M,o.x, getStorage(r.storage,:x) ) : 0.0,
        i
    )
    r.storage(p,o,i)
    return r.recordedValues
end

@doc doc"""
    RecordEntry{T} <: RecordAction

record a certain fields entry of type {T} during the iterates

# Fields
* `recordedValues` – the recorded Iterates
* `field` – Symbol the entry can be accessed with within [`Options`](@ref)

"""
mutable struct RecordEntry{T} <: RecordAction
    recordedValues::Array{T,1}
    field::Symbol
    RecordEntry{T}(f::Symbol) where T = new(Array{T,1}(),f)
end
RecordEntry(e::T,f::Symbol) where T = RecordEntry{T}(f)
RecordEntry(d::DataType,f::Symbol) = RecordEntry{d}(f)
(r::RecordEntry{T})(p::Pr,o::O,i::Int) where {T, Pr <: Problem, O <: Options} = recordOrReset!(r, getfield(o, r.field), i)

@doc doc"""
    RecordEntryChange{T} <: RecordAction

record a certain entries change during iterates

# Additional Fields
* `recordedValues` – the recorded Iterates
* `field` – Symbol the field can be accessed with within [`Options`](@ref)
* `distance` – function (p,o,x1,x2) to compute the change/distance between two values of the entry 
* `storage` – a [`StoreOptionsAction`](@ref) to store (at least) `getproperty(o, d.field)`
"""
mutable struct RecordEntryChange <: RecordAction
    recordedValues::Array{Float64,1}
    field::Symbol
    distance::Function
    storage::StoreOptionsAction
    RecordEntryChange(
            f::Symbol,
            d::Function,
            a::StoreOptionsAction=StoreOptionsAction( (f,) )
        ) = new(Array{Float64,1}(),f,d,a)
    function RecordEntryChange(v::T where T, f::Symbol, d::Function,
            a::StoreOptionsAction=StoreOptionsAction( (f,) )
        )
        updateStorage!(a,Dict(f=>v))
        return new(Array{Float64,1}(),f, d, a)
    end
end
function (r::RecordEntryChange)(p::P,o::O,i::Int) where {P <: Problem, O <: Options}
    recordOrReset!(r,
        hasStorage(r.storage, r.field) ? r.distance(p,o, getfield(o, r.field), getStorage(r.storage, r.field) ) : 0.0,
    i)
    r.storage(p,o,i)
end

@doc doc"""
    RecordIterate <: RecordAction

record the iterate

# Constructors
    RecordIterate(x0)

initialize the iterate record array to the type of `x0`, e.g. your initial data.

    RecordIterate(P)

initialize the iterate record array to the data type `P`, where `P<:MPoint`holds.
"""
mutable struct RecordIterate{P <: MPoint} <: RecordAction
    recordedValues::Array{P,1}
    RecordIterate{P}() where {P <: MPoint} = new(Array{P,1}())
end
RecordIterate(p::P) where {P <: MPoint} = RecordIterate{P}()
RecordIterate(d::DataType) = (<:(d,MPoint)) ? RecordIterate{d}() : throw(ErrorException("Unknown manifold point (<:MPoint) DataType  $d"))
RecordIterate() = throw(ErrorException("The iterate's data type has to be provided, i.e. either RecordIterate(x0) or RecordIterate(<:MPoint)"))

(r::RecordIterate{P})(p::Pr,o::O,i::Int) where {P <: MPoint, Pr <: Problem, O <: Options} = recordOrReset!(r, o.x, i)

@doc doc"""
    RecordIteration <: RecordAction

record the current iteration
"""
mutable struct RecordIteration <: RecordAction
    recordedValues::Array{Int,1}
    RecordIteration() = new(Array{Int,1}())
end
(r::RecordIteration)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = recordOrReset!(r, i, i)
    
@doc doc"""
    RecordCost <: RecordAction

record the current cost function value, see [`getCost`](@ref).
"""
mutable struct RecordCost <: RecordAction
    recordedValues::Array{Float64,1}
    RecordCost() = new(Array{Float64,1}())
end
(r::RecordCost)(p::P,o::O,i::Int) where {P <: Problem, O <: Options} = recordOrReset!(r, getCost(p,o.x), i)


@doc doc"""
    RecordFactory(a)

given an array of `Symbol`s and [`RecordAction`](@ref)s and `Ints`

* The symbol `:Cost` creates a [`RecordCost`](@ref)
* The symbol `:iteration` creates a [`RecordIteration`](@ref)
* The symbol `:Change` creates a [`RecordChange`](@ref)
* any other symbol creates a [`RecordEntry`](@ref) of the corresponding field in [`Options`](@ref)
* any [`RecordAction`](@ref) is directly included
* an Integer `k` introduces that record is only performed every `k`th iteration
"""
function RecordFactory(o::O, a::Array{<:Any,1} ) where {O <: Options}
    # filter out every
    group = Array{RecordAction,1}()
    for s in filter(x -> !isa(x,Int), a) # filter ints and stop
        push!(group,RecordActionFactory(o,s) )
    end
    record = RecordGroup(group)
    # filter ints
    e = filter(x -> isa(x,Int),a)
    if length(e) > 0
        record = RecordEvery(record,last(e))
    end
    dictionary = Dict{Symbol,RecordAction}(:All => record)
    return dictionary
end
@doc doc"""
    RecordActionFactory(s)

create a [`RecordAction`](@ref) where

* a [`RecordAction`](@ref) is passed through
* a [`Symbol`] creates [`RecordEntry`](@ref) of that symbol, with the exceptions
  of `:Change`, `:Iterate`, `:Iteration`, and `:Cost`.
"""
RecordActionFactory(o::O,a::A) where {O <: Options, A <: RecordAction} = a
function RecordActionFactory(o::O,s::Symbol) where {O <: Options}
    if (s==:Change)
        return RecordChange()
    elseif (s==:Iteration)
        return RecordIteration()
    elseif (s==:Iterate)
        return RecordIterate(o.x)
    elseif (s==:Cost)
        return RecordCost()
    end
        return RecordEntry(getfield(o,s),s)
end