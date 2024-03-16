@doc raw"""
    RecordAction

A `RecordAction` is a small functor to record values.
The usual call is given by
`(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, i) -> s` that performs the record,
where `i` is the current iteration.

By convention `i<=0` is interpreted as "For Initialization only," so only
initialize internal values, but not trigger any record, the same holds for
`i=typemin(Inf)` which is used to indicate `stop`, that the record is
called from within [`stop_solver!`](@ref) which returns true afterwards.

# Fields (assumed by subtypes to exist)

* `recorded_values` an `Array` of the recorded values.
"""
abstract type RecordAction <: AbstractStateAction end

@doc raw"""
    RecordSolverState <: AbstractManoptSolverState

append to any [`AbstractManoptSolverState`](@ref) the decorator with record capability,
Internally a dictionary is kept that stores a [`RecordAction`](@ref) for
several concurrent modes using a `Symbol` as reference.
The default mode is `:Iteration`, which is used to store information that is recorded during
the iterations. RecordActions might be added to `:Start` or `:Stop` to record values at the
beginning or for the stopping time point, respectively

The original options can still be accessed using the [`get_state`](@ref) function.

# Fields

* `options`          the options that are extended by debug information
* `recordDictionary` a `Dict{Symbol,RecordAction}` to keep track of all
  different recorded values

# Constructors

    RecordSolverState(o,dR)

construct record decorated [`AbstractManoptSolverState`](@ref), where `dR` can be

* a [`RecordAction`](@ref), then it is stored within the dictionary at `:Iteration`
* an `Array` of [`RecordAction`](@ref)s, then it is stored as a
  `recordDictionary`(@ref) within the dictionary at `:All`.
* a `Dict{Symbol,RecordAction}`.
"""
mutable struct RecordSolverState{S<:AbstractManoptSolverState,TRD<:NamedTuple} <:
               AbstractManoptSolverState
    state::S
    recordDictionary::TRD
    function RecordSolverState{S}(s::S; kwargs...) where {S<:AbstractManoptSolverState}
        return new{S,typeof(values(kwargs))}(s, values(kwargs))
    end
end
function RecordSolverState(s::S, dR::RecordAction) where {S<:AbstractManoptSolverState}
    return RecordSolverState{S}(s; Iteration=dR)
end
function RecordSolverState(
    s::S, dR::Dict{Symbol,<:RecordAction}
) where {S<:AbstractManoptSolverState}
    return RecordSolverState{S}(s; dR...)
end
function RecordSolverState(s::S, format::Vector{<:Any}) where {S<:AbstractManoptSolverState}
    return RecordSolverState{S}(s; RecordFactory(get_state(s), format)...)
end
function RecordSolverState(s::S, symbol::Symbol) where {S<:AbstractManoptSolverState}
    return RecordSolverState{S}(s; Iteration=RecordFactory(get_state(s), symbol))
end
function status_summary(rst::RecordSolverState)
    if length(rst.recordDictionary) > 0
        return """
               $(rst.state)

               ## Record
               $(rst.recordDictionary)
               """
    else
        return "RecordSolverState($(rst.state), $(rst.recordDictionary))"
    end
end
function show(io::IO, rst::RecordSolverState)
    return print(io, status_summary(rst))
end
dispatch_state_decorator(::RecordSolverState) = Val(true)

@doc """
    has_record(s::AbstractManoptSolverState)

Indicate whether the [`AbstractManoptSolverState`](@ref)` s` are decorated with
[`RecordSolverState`](@ref)
"""
has_record(::RecordSolverState) = true
has_record(s::AbstractManoptSolverState) = _has_record(s, dispatch_state_decorator(s))
_has_record(s::AbstractManoptSolverState, ::Val{true}) = has_record(s.state)
_has_record(::AbstractManoptSolverState, ::Val{false}) = false

# pass through
function set_manopt_parameter!(rss::RecordSolverState, e::Symbol, args...)
    return set_manopt_parameter!(rss.state, e, args...)
end
function get_manopt_parameter(rss::RecordSolverState, e::Symbol, args...)
    return get_manopt_parameter(rss.state, e, args...)
end

@doc """
    get_record_state(s::AbstractManoptSolverState)

return the [`RecordSolverState`](@ref) among the decorators from the [`AbstractManoptSolverState`](@ref) `o`
"""
function get_record_state(s::AbstractManoptSolverState)
    return _get_record_state(s, dispatch_state_decorator(s))
end
function _get_record_state(s::AbstractManoptSolverState, ::Val{true})
    return get_record_state(s.state)
end
function _get_record_state(::AbstractManoptSolverState, ::Val{false})
    return error("No Record decoration found")
end
get_record_state(s::RecordSolverState) = s

@doc raw"""
    get_record_action(s::AbstractManoptSolverState, s::Symbol)

return the action contained in the (first) [`RecordSolverState`](@ref) decorator within the [`AbstractManoptSolverState`](@ref) `o`.

"""
function get_record_action(s::AbstractManoptSolverState, symbol::Symbol=:Iteration)
    if haskey(s.recordDictionary, symbol)
        return s.recordDictionary[symbol]
    else
        error("No record known for key :$s found")
    end
end
@doc raw"""
    get_record(s::AbstractManoptSolverState, [,symbol=:Iteration])
    get_record(s::RecordSolverState, [,symbol=:Iteration])

return the recorded values from within the [`RecordSolverState`](@ref) `s` that where
recorded with respect to the `Symbol symbol` as an `Array`. The default refers to
any recordings during an `:Iteration`.

When called with arbitrary [`AbstractManoptSolverState`](@ref), this method looks for the
[`RecordSolverState`](@ref) decorator and calls `get_record` on the decorator.
"""
function get_record(s::RecordSolverState, symbol::Symbol=:Iteration)
    return get_record(get_record_action(s, symbol))
end
function get_record(s::RecordSolverState, symbol::Symbol, i...)
    return get_record(get_record_action(s, symbol), i...)
end
function get_record(s::AbstractManoptSolverState, symbol::Symbol=:Iteration)
    return get_record(get_record_state(s), symbol)
end

@doc raw"""
    get_record(r::RecordAction)

return the recorded values stored within a [`RecordAction`](@ref) `r`.
"""
get_record(r::RecordAction, i) = r.recorded_values
get_record(r::RecordAction) = r.recorded_values

"""
    get_index(rs::RecordSolverState, s::Symbol)
    ro[s]

Get the recorded values for recorded type `s`, see [`get_record`](@ref) for details.

    get_index(rs::RecordSolverState, s::Symbol, i...)
    ro[s, i...]

Access the recording type of type `s` and call its [`RecordAction`](@ref) with `[i...]`.
"""
getindex(rs::RecordSolverState, s::Symbol) = get_record(rs, s)
getindex(rs::RecordSolverState, s::Symbol, i...) = get_record_action(rs, s)[i...]

"""
    record_or_reset!(r,v,i)

either record (`i>0` and not `Inf`) the value `v` within the [`RecordAction`](@ref) `r`
or reset (`i<0`) the internal storage, where `v` has to match the internal
value type of the corresponding [`RecordAction`](@ref).
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
act independently, but the results can be collected in a grouped fashion, a tuple per calls of this group.
The entries can be later addressed either by index or semantic Symbols

# Constructors
    RecordGroup(g::Array{<:RecordAction, 1})

construct a group consisting of an Array of [`RecordAction`](@ref)s `g`,

    RecordGroup(g, symbols)

# Examples
    r = RecordGroup([RecordIteration(), RecordCost()])

A RecordGroup to record the current iteration and the cost. The cost can then be accessed using `get_record(r,2)` or `r[2]`.

    r = RecordGroup([RecordIteration(), RecordCost()], Dict(:Cost => 2))

A RecordGroup to record the current iteration and the cost, which can then be accessed using `get_record(:Cost)` or `r[:Cost]`.

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
function (d::RecordGroup)(p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int)
    for ri in d.group
        ri(p, s, i)
    end
end
function status_summary(rg::RecordGroup)
    return "[ $( join(["$(status_summary(ri))" for ri in rg.group], ", ")) ]"
end
function show(io::IO, rg::RecordGroup)
    s = join(["$(ri)" for ri in rg.group], ", ")
    return print(io, "RecordGroup([$s])")
end

@doc raw"""
    get_record(r::RecordGroup)

return an array of tuples, where each tuple is a recorded set per iteration or record call.

    get_record(r::RecordGruop, i::Int)

return an array of values corresponding to the `i`th entry in this record group

    get_record(r::RecordGruop, s::Symbol)

return an array of recorded values with respect to the `s`, see [`RecordGroup`](@ref).

    get_record(r::RecordGroup, s1::Symbol, s2::Symbol,...)

return an array of tuples, where each tuple is a recorded set corresponding to the symbols `s1, s2,...` per iteration / record call.
"""
get_record(r::RecordGroup) = length(r.group) > 0 ? [zip(get_record.(r.group)...)...] : []
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
    RecordCost <: RecordAction

Record the current cost function value, see [`get_cost`](@ref).
"""
mutable struct RecordCost <: RecordAction
    recorded_values::Array{Float64,1}
    RecordCost() = new(Array{Float64,1}())
end
function (r::RecordCost)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int)
    return record_or_reset!(r, get_cost(amp, get_iterate(s)), i)
end
show(io::IO, ::RecordCost) = print(io, "RecordCost()")
status_summary(di::RecordCost) = ":Cost"

@doc raw"""
    RecordEvery <: RecordAction

record only every $i$th iteration.
Otherwise (optionally, but activated by default) just update internal tracking
values.

This method does not perform any record itself but relies on it's children's methods
"""
mutable struct RecordEvery <: RecordAction
    record::RecordAction
    every::Int
    always_update::Bool
    function RecordEvery(r::RecordAction, every::Int=1, always_update::Bool=true)
        return new(r, every, always_update)
    end
end
function (d::RecordEvery)(p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int)
    if i <= 0
        d.record(p, s, i)
    elseif (rem(i, d.every) == 0)
        d.record(p, s, i)
    elseif d.always_update
        d.record(p, s, 0)
    end
end
function show(io::IO, re::RecordEvery)
    return print(io, "RecordEvery($(re.record), $(re.every), $(re.always_update))")
end
function status_summary(re::RecordEvery)
    s = ""
    if re.record isa RecordGroup
        s = status_summary(re.record)[3:(end - 2)]
    else
        s = "$(re.record)"
    end
    return "[$s, $(re.every)]"
end
get_record(r::RecordEvery) = get_record(r.record)
get_record(r::RecordEvery, i) = get_record(r.record, i)
getindex(r::RecordEvery, i) = get_record(r, i)

#
# Special single ones
#
@doc raw"""
    RecordChange <: RecordAction

debug for the amount of change of the iterate (stored in `o.x` of the [`AbstractManoptSolverState`](@ref))
during the last iteration.

# Additional fields

* `storage` a [`StoreStateAction`](@ref) to store (at least) `o.x` to use this
  as the last value (to compute the change
* `inverse_retraction_method` - (`default_inverse_retraction_method(manifold, p)`) the
  inverse retraction to be used for approximating distance.

# Constructor

    RecordChange(M=DefaultManifold();)

with the preceding fields as keywords. For the `DefaultManifold` only the field storage is used.
Providing the actual manifold moves the default storage to the efficient point storage.
"""
mutable struct RecordChange{
    TInvRetr<:AbstractInverseRetractionMethod,TStorage<:StoreStateAction
} <: RecordAction
    recorded_values::Vector{Float64}
    storage::TStorage
    inverse_retraction_method::TInvRetr
    function RecordChange(
        M::AbstractManifold=DefaultManifold();
        storage::Union{Nothing,StoreStateAction}=nothing,
        manifold::Union{Nothing,AbstractManifold}=nothing,
        inverse_retraction_method::IRT=default_inverse_retraction_method(M),
    ) where {IRT<:AbstractInverseRetractionMethod}
        irm = inverse_retraction_method
        if !isnothing(manifold)
            @warn "The `manifold` keyword is deprecated, use the first positional argument `M`. This keyword for now sets `inverse_retracion_method`."
            irm = default_inverse_retraction_method(manifold)
        end
        if isnothing(storage)
            if M isa DefaultManifold
                storage = StoreStateAction(M; store_fields=[:Iterate])
            else
                storage = StoreStateAction(M; store_points=Tuple{:Iterate})
            end
        end
        return new{typeof(irm),typeof(storage)}(Vector{Float64}(), storage, irm)
    end
    function RecordChange(
        p,
        a::StoreStateAction=StoreStateAction([:Iterate]);
        manifold::AbstractManifold=DefaultManifold(1),
        inverse_retraction_method::IRT=default_inverse_retraction_method(
            manifold, typeof(p)
        ),
    ) where {IRT<:AbstractInverseRetractionMethod}
        update_storage!(a, Dict(:Iterate => p))
        return new{IRT,typeof(a)}(Vector{Float64}(), a, inverse_retraction_method)
    end
end
function (r::RecordChange)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int)
    M = get_manifold(amp)
    record_or_reset!(
        r,
        if has_storage(r.storage, PointStorageKey(:Iterate))
            distance(
                M,
                get_iterate(s),
                get_storage(r.storage, PointStorageKey(:Iterate)),
                r.inverse_retraction_method,
            )
        else
            0.0
        end,
        i,
    )
    r.storage(amp, s, i)
    return r.recorded_values
end
function show(io::IO, rc::RecordChange)
    return print(
        io, "RecordChange(; inverse_retraction_method=$(rc.inverse_retraction_method))"
    )
end
status_summary(rc::RecordChange) = ":Change"

@doc raw"""
    RecordEntry{T} <: RecordAction

record a certain fields entry of type {T} during the iterates

# Fields

* `recorded_values` the recorded Iterates
* `field`           Symbol the entry can be accessed with within [`AbstractManoptSolverState`](@ref)

"""
mutable struct RecordEntry{T} <: RecordAction
    recorded_values::Array{T,1}
    field::Symbol
    RecordEntry{T}(f::Symbol) where {T} = new(Array{T,1}(), f)
end
RecordEntry(::T, f::Symbol) where {T} = RecordEntry{T}(f)
RecordEntry(d::DataType, f::Symbol) = RecordEntry{d}(f)
function (r::RecordEntry{T})(
    ::AbstractManoptProblem, s::AbstractManoptSolverState, i
) where {T}
    return record_or_reset!(r, getfield(s, r.field), i)
end
function show(io::IO, di::RecordEntry)
    return print(io, "RecordEntry(:$(di.field))")
end

@doc raw"""
    RecordEntryChange{T} <: RecordAction

record a certain entries change during iterates

# Additional fields

* `recorded_values` the recorded Iterates
* `field`           Symbol the field can be accessed with within [`AbstractManoptSolverState`](@ref)
* `distance`        function (p,o,x1,x2) to compute the change/distance between two values of the entry
* `storage`         a [`StoreStateAction`](@ref) to store (at least) `getproperty(o, d.field)`
"""
mutable struct RecordEntryChange{TStorage<:StoreStateAction} <: RecordAction
    recorded_values::Vector{Float64}
    field::Symbol
    distance::Any
    storage::TStorage
    function RecordEntryChange(f::Symbol, d, a::StoreStateAction=StoreStateAction([f]))
        return new{typeof(a)}(Float64[], f, d, a)
    end
    function RecordEntryChange(v, f::Symbol, d, a::StoreStateAction=StoreStateAction([f]))
        update_storage!(a, Dict(f => v))
        return new{typeof(a)}(Float64[], f, d, a)
    end
end
function (r::RecordEntryChange)(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, i::Int
)
    value = 0.0
    if has_storage(r.storage, r.field)
        value = r.distance(
            amp, ams, getfield(ams, r.field), get_storage(r.storage, r.field)
        )
    end
    r.storage(amp, ams, i)
    return record_or_reset!(r, value, i)
end
function show(io::IO, rec::RecordEntryChange)
    return print(io, "RecordEntryChange(:$(rec.field), $(rec.distance))")
end

@doc raw"""
    RecordIterate <: RecordAction

record the iterate

# Constructors
    RecordIterate(x0)

initialize the iterate record array to the type of `x0`, which indicates the kind of iterate

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
        ErrorException("The iterate's data type has to be provided, RecordIterate(x0).")
    )
end
function (r::RecordIterate{T})(
    ::AbstractManoptProblem, s::AbstractManoptSolverState, i
) where {T}
    return record_or_reset!(r, get_iterate(s), i)
end
function show(io::IO, ri::RecordIterate)
    return print(io, "RecordIterate($(eltype(ri.recorded_values)))")
end
status_summary(di::RecordIterate) = ":Iterate"

@doc raw"""
    RecordIteration <: RecordAction

record the current iteration
"""
mutable struct RecordIteration <: RecordAction
    recorded_values::Array{Int,1}
    RecordIteration() = new(Array{Int,1}())
end
function (r::RecordIteration)(::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int)
    return record_or_reset!(r, i, i)
end
show(io::IO, ::RecordIteration) = print(io, "RecordIteration()")
status_summary(::RecordIteration) = ":Iteration"

@doc raw"""
    RecordTime <: RecordAction

record the time elapsed during the current iteration.

The three possible modes are
* `:cumulative` record times without resetting the timer
* `:iterative` record times with resetting the timer
* `:total` record a time only at the end of an algorithm (see [`stop_solver!`](@ref))

The default is `:cumulative`, and any non-listed symbol default to using this mode.

# Constructor

    RecordTime(; mode::Symbol=:cumulative)
"""
mutable struct RecordTime <: RecordAction
    recorded_values::Array{Nanosecond,1}
    start::Nanosecond
    mode::Symbol
    function RecordTime(; mode::Symbol=:cumulative)
        return new(Array{Nanosecond,1}(), Nanosecond(time_ns()), mode)
    end
end
function (r::RecordTime)(p::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int)
    # At iteration zero also reset start
    (i == 0) && (r.start = Nanosecond(time_ns()))
    t = Nanosecond(time_ns()) - r.start
    (r.mode == :iterative) && (r.start = Nanosecond(time_ns()))
    if r.mode == :total
        # only record at end (if `stop_solver` returns true)
        return record_or_reset!(r, t, (i > 0 && stop_solver!(p, s, i)) ? i : 0)
    else
        return record_or_reset!(r, t, i)
    end
end
function show(io::IO, ri::RecordTime)
    return print(io, "RecordTime(; mode=:$(ri.mode))")
end
status_summary(ri::RecordTime) = (ri.mode === :iterative ? ":IterativeTime" : ":Time")

@doc raw"""
    RecordFactory(s::AbstractManoptSolverState, a)

given an array of `Symbol`s and [`RecordAction`](@ref)s and `Ints`

* The symbol `:Cost` creates a [`RecordCost`](@ref)
* The symbol `:iteration` creates a [`RecordIteration`](@ref)
* The symbol `:Change` creates a [`RecordChange`](@ref)
* any other symbol creates a [`RecordEntry`](@ref) of the corresponding field in [`AbstractManoptSolverState`](@ref)
* any [`RecordAction`](@ref) is directly included
* an semantic pair `:symbol => RecordAction` is directly included
* an Integer `k` introduces that record is only performed every `k`th iteration
"""
function RecordFactory(s::AbstractManoptSolverState, a::Array{<:Any,1})
    # filter out every
    group = Array{Union{<:RecordAction,Pair{Symbol,<:RecordAction}},1}()
    for element in filter(x -> !isa(x, Int), a) # all non-integers/stopping-criteria
        if element isa Symbol # factory for this symbol, store in a pair
            push!(group, element => RecordActionFactory(s, element))
        elseif element isa Pair{<:Symbol,<:RecordAction} #already a generated action
            push!(group, element)
        else # process the others as elements for an action factory
            push!(group, RecordActionFactory(s, element))
        end
    end
    record = RecordGroup(group)
    # filter integer numbers
    e = filter(x -> isa(x, Int), a)
    if length(e) > 0
        record = RecordEvery(record, last(e))
    end
    return (; Iteration=record)
end
RecordFactory(s::AbstractManoptSolverState, symbol::Symbol) = RecordActionFactory(s, symbol)

@doc raw"""
    RecordActionFactory(s)

create a [`RecordAction`](@ref) where

* a [`RecordAction`](@ref) is passed through
* a [`Symbol`] creates [`RecordEntry`](@ref) of that symbol, with the exceptions
  of
  * `:Change`        to record the change of the iterates in `o.x``
  * `:Iterate`       to record the iterate
  * `:Iteration`     to record the current iteration number
  * `:Cost`          to record the current cost function value
  * `:Time`          to record the total time taken after every iteration
  * `:IterativeTime` to record the times taken for each iteration.
"""
RecordActionFactory(::AbstractManoptSolverState, a::RecordAction) = a
RecordActionFactory(::AbstractManoptSolverState, sa::Pair{Symbol,<:RecordAction}) = sa
function RecordActionFactory(s::AbstractManoptSolverState, symbol::Symbol)
    if (symbol == :Change)
        return RecordChange()
    elseif (symbol == :Iteration)
        return RecordIteration()
    elseif (symbol == :Iterate)
        return RecordIterate(get_iterate(s))
    elseif (symbol == :Cost)
        return RecordCost()
    elseif (symbol == :Time)
        return RecordTime(; mode=:cumulative)
    elseif (symbol == :IterativeTime)
        return RecordTime(; mode=:iterative)
    end
    return RecordEntry(getfield(s, symbol), symbol)
end
