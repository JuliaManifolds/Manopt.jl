@doc """
    RecordAction

A `RecordAction` is a small functor to record values.
The usual call is given by

    (amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k) -> s

that performs the record for the current problem and solver combination, and where `k` is
the current iteration.

By convention `i=0` is interpreted as "For Initialization only," so only
initialize internal values, but not trigger any record, that the record is
called from within [`stop_solver!`](@ref) which returns true afterwards.

Any negative value is interpreted as a “reset”, and should hence delete all stored recordings,
for example when reusing a `RecordAction`.
The start of a solver calls the `:Iteration` and `:Stop` dictionary entries with `-1`,
to reset those recordings.

By default any `RecordAction` is assumed to record its values in a field `recorded_values`,
an `Vector` of recorded values. See [`get_record`](@ref get_record(r::RecordAction))`(ra)`.
"""
abstract type RecordAction <: AbstractStateAction end

@doc """
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
  `recordDictionary`(@ref).
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
    return RecordSolverState{S}(s; RecordFactory(get_state(s), symbol)...)
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

"""
    set_parameter!(ams::RecordSolverState, ::Val{:Record}, args...)

Set certain values specified by `args...` into the elements of the `recordDictionary`
"""
function set_parameter!(rss::RecordSolverState, ::Val{:Record}, args...)
    for d in values(rss.recordDictionary)
        set_parameter!(d, args...)
    end
    return rss
end
# all other pass through
function set_parameter!(rss::RecordSolverState, v::Val{T}, args...) where {T}
    return set_parameter!(rss.state, v, args...)
end
function set_parameter!(rss::RecordSolverState, v::Val{:StoppingCriterion}, args...)
    return set_parameter!(rss.state, v, args...)
end
# all other pass through
function get_parameter(rss::RecordSolverState, v::Val{T}, args...) where {T}
    return get_parameter(rss.state, v, args...)
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

@doc """
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
@doc """
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

@doc """
    get_record(r::RecordAction)

return the recorded values stored within a [`RecordAction`](@ref) `r`.
"""
get_record(r::RecordAction) = r.recorded_values
get_record(r::RecordAction, k) = r.recorded_values

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
    record_or_reset!(r, v, k)

either record (`k>0` and not `Inf`) the value `v` within the [`RecordAction`](@ref) `r`
or reset (`k<0`) the internal storage, where `v` has to match the internal
value type of the corresponding [`RecordAction`](@ref).
"""
function record_or_reset!(r::RecordAction, v, k::Int)
    if k > 0
        push!(r.recorded_values, deepcopy(v))
    elseif k < 0 # reset if negative
        r.recorded_values = empty(r.recorded_values) # Reset to empty
    end
end

#
# Meta Record States
#

@doc """
    RecordEvery <: RecordAction

record only every ``k``th iteration.
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
function (re::RecordEvery)(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k::Int
)
    if k <= 0
        re.record(amp, ams, k)
    elseif (rem(k, re.every) == 0)
        re.record(amp, ams, k)
    elseif re.always_update
        re.record(amp, ams, 0)
    end
    # Set activity to activate or deactivate sub solvers
    # note that since recording is happening at the end
    # sets activity for the _next_ iteration
    set_parameter!(
        ams, :SubState, :Record, :Activity, !(k < 1) && (rem(k + 1, re.every) == 0)
    )
    return nothing
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
get_record(r::RecordEvery, k) = get_record(r.record, k)
getindex(r::RecordEvery, k) = get_record(r, k)

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
    g1 = RecordGroup([RecordIteration(), RecordCost()])

A RecordGroup to record the current iteration and the cost. The cost can then be accessed using `get_record(r,2)` or `r[2]`.

    g2 = RecordGroup([RecordIteration(), RecordCost()], Dict(:Cost => 2))

A RecordGroup to record the current iteration and the cost, which can then be accessed using `get_record(:Cost)` or `r[:Cost]`.

    g3 = RecordGroup([RecordIteration(), RecordCost() => :Cost])

A RecordGroup identical to the previous constructor, just a little easier to use.
To access all recordings of the second entry of this last `g3` you can do either `g4[2]` or `g[:Cost]`,
the first one can only be accessed by `g4[1]`, since no symbol was given here.
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
        records::Vector,# assumed: {<:Union{<:RecordAction,Pair{<:RecordAction,Symbol}, rest ignored
    )
        g = Array{RecordAction,1}()
        si = Dict{Symbol,Int}()
        for i in 1:length(records)
            if records[i] isa RecordAction
                push!(g, records[i])
            elseif records[i] isa Pair{<:RecordAction,Symbol}
                push!(g, records[i].first)
                push!(si, records[i].second => i)
            else
                error("Unrecognised element of recording $(repr(records[i])) at entry $i.")
            end
        end
        return RecordGroup(g, si)
    end
    RecordGroup() = new(Array{RecordAction,1}(), Dict{Symbol,Int}())
end
function (d::RecordGroup)(p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    for ri in d.group
        ri(p, s, k)
    end
end
function status_summary(rg::RecordGroup)
    return "[ $( join(["$(status_summary(ri))" for ri in rg.group], ", ")) ]"
end
function show(io::IO, rg::RecordGroup)
    s = join(["$(ri)" for ri in rg.group], ", ")
    return print(io, "RecordGroup([$s])")
end

@doc """
    get_record(r::RecordGroup)

return an array of tuples, where each tuple is a recorded set per iteration or record call.

    get_record(r::RecordGruop, k::Int)

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

@doc """
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

@doc """
    RecordSubsolver <: RecordAction

Record the current sub solvers recording, by calling [`get_record`](@ref)
on the sub state with

# Fields
* `records`: an array to store the recorded values
* `symbols`: arguments for [`get_record`](@ref). Defaults to just one symbol `:Iteration`, but could be set to also record the `:Stop` action.

# Constructor

    RecordSubsolver(; record=[:Iteration,], record_type=eltype([]))
"""
mutable struct RecordSubsolver{R} <: RecordAction
    recorded_values::Vector{R}
    record::Vector{Symbol}
end
function RecordSubsolver(;
    record::Union{Symbol,Vector{Symbol}}=:Iteration, record_type=eltype([])
)
    r = record isa Symbol ? [record] : record
    return RecordSubsolver{record_type}(record_type[], r)
end
function (rsr::RecordSubsolver)(
    ::AbstractManoptProblem, ams::AbstractManoptSolverState, k::Int
)
    record_or_reset!(rsr, get_record(get_sub_state(ams), rsr.record...), k)
    return nothing
end
function show(io::IO, rsr::RecordSubsolver{R}) where {R}
    return print(io, "RecordSubsolver(; record=$(rsr.record), record_type=$R)")
end
status_summary(::RecordSubsolver) = ":Subsolver"

@doc """
    RecordWhenActive <: RecordAction

record action that only records if the `active` boolean is set to true.
This can be set from outside and is for example triggered by |`RecordEvery`](@ref)
on recordings of the subsolver.
While this is for sub solvers maybe not completely necessary, recording values that
are never accessible, is not that useful.

# Fields

* `active`:        a boolean that can (de-)activated from outside to turn on/off debug
* `always_update`: whether or not to call the inner debugs with nonpositive iterates (init/reset)

# Constructor

    RecordWhenActive(r::RecordAction, active=true, always_update=true)
"""
mutable struct RecordWhenActive{R<:RecordAction} <: RecordAction
    record::R
    active::Bool
    always_update::Bool
    function RecordWhenActive(
        r::R, active::Bool=true, always_update::Bool=true
    ) where {R<:RecordAction}
        return new{R}(r, active, always_update)
    end
end

function (rwa::RecordWhenActive)(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k::Int
)
    if rwa.active
        rwa.record(amp, ams, k)
    elseif (rwa.always_update) && (k <= 0)
        rwa.record(amp, ams, k)
    end
end
function show(io::IO, rwa::RecordWhenActive)
    return print(io, "RecordWhenActive($(rwa.record), $(rwa.active), $(rwa.always_update))")
end
function status_summary(rwa::RecordWhenActive)
    return repr(rwa)
end
function set_parameter!(rwa::RecordWhenActive, v::Val, args...)
    set_parameter!(rwa.record, v, args...)
    return rwa
end
function set_parameter!(rwa::RecordWhenActive, ::Val{:Activity}, v)
    return rwa.active = v
end
get_record(r::RecordWhenActive, args...) = get_record(r.record, args...)

#
# Specific Record types
#

@doc """
    RecordCost <: RecordAction

Record the current cost function value, see [`get_cost`](@ref).

# Fields

* `recorded_values` : to store the recorded values

# Constructor

    RecordCost()
"""
mutable struct RecordCost <: RecordAction
    recorded_values::Array{Float64,1}
    RecordCost() = new(Array{Float64,1}())
end
function (r::RecordCost)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    return record_or_reset!(r, get_cost(amp, get_iterate(s)), k)
end
show(io::IO, ::RecordCost) = print(io, "RecordCost()")
status_summary(di::RecordCost) = ":Cost"

@doc """
    RecordChange <: RecordAction

debug for the amount of change of the iterate (see [`get_iterate`](@ref)`(s)` of the [`AbstractManoptSolverState`](@ref))
during the last iteration.

# Fields

* `storage`                   : a [`StoreStateAction`](@ref) to store (at least) the last
  iterate to use this as the last value (to compute the change) serving as a potential cache
  shared with other components of the solver.
$(_var(:Keyword, :inverse_retraction_method))
* `recorded_values`           : to store the recorded values

# Constructor

    RecordChange(M=DefaultManifold();
        inverse_retraction_method = default_inverse_retraction_method(M),
        storage                   = StoreStateAction(M; store_points=Tuple{:Iterate})
    )

with the previous fields as keywords. For the `DefaultManifold` only the field storage is used.
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
        inverse_retraction_method::IRT=default_inverse_retraction_method(M),
    ) where {IRT<:AbstractInverseRetractionMethod}
        irm = inverse_retraction_method
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
function (r::RecordChange)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
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
        k,
    )
    r.storage(amp, s, k)
    return r.recorded_values
end
function show(io::IO, rc::RecordChange)
    return print(
        io, "RecordChange(; inverse_retraction_method=$(rc.inverse_retraction_method))"
    )
end
status_summary(rc::RecordChange) = ":Change"

@doc """
    RecordEntry{T} <: RecordAction

record a certain fields entry of type {T} during the iterates

# Fields

* `recorded_values` : the recorded Iterates
* `field`           : Symbol the entry can be accessed with within [`AbstractManoptSolverState`](@ref)

# Constructor
    RecordEntry(::T, f::Symbol)
    RecordEntry(T::DataType, f::Symbol)

Initialize the record action to record the state field `f`, and initialize the
`recorded_values` to be a vector of element type `T`.

# Examples

* `RecordEntry(rand(M), :q)` to record the points from `M` stored in some states `s.q`
* `RecordEntry(SVDMPoint, :p)` to record the field `s.p` which takes values of type [`SVDMPoint`](@extref `Manifolds.SVDMPoint`).
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

@doc """
    RecordEntryChange{T} <: RecordAction

record a certain entries change during iterates

# Additional fields

* `recorded_values` : the recorded Iterates
* `field`           : Symbol the field can be accessed with within [`AbstractManoptSolverState`](@ref)
* `distance`        : function (p,o,x1,x2) to compute the change/distance between two values of the entry
* `storage`         : a [`StoreStateAction`](@ref) to store (at least) `getproperty(o, d.field)`

# Constructor

    RecordEntryChange(f::Symbol, d, a::StoreStateAction=StoreStateAction([f]))
"""
mutable struct RecordEntryChange{TStorage<:StoreStateAction} <: RecordAction
    recorded_values::Vector{Float64}
    field::Symbol
    distance::Any
    storage::TStorage
    function RecordEntryChange(f::Symbol, d, a::StoreStateAction=StoreStateAction([f]))
        return new{typeof(a)}(Float64[], f, d, a)
    end
end
function (r::RecordEntryChange)(
    amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k::Int
)
    value = 0.0
    if has_storage(r.storage, r.field)
        value = r.distance(
            amp, ams, getfield(ams, r.field), get_storage(r.storage, r.field)
        )
    end
    r.storage(amp, ams, k)
    return record_or_reset!(r, value, k)
end
function show(io::IO, rec::RecordEntryChange)
    return print(io, "RecordEntryChange(:$(rec.field), $(rec.distance))")
end

@doc """
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

@doc """
    RecordIteration <: RecordAction

record the current iteration
"""
mutable struct RecordIteration <: RecordAction
    recorded_values::Array{Int,1}
    RecordIteration() = new(Array{Int,1}())
end
function (r::RecordIteration)(::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int)
    return record_or_reset!(r, k, k)
end
show(io::IO, ::RecordIteration) = print(io, "RecordIteration()")
status_summary(::RecordIteration) = ":Iteration"

@doc """
    RecordStoppingReason <: RecordAction

Record reason the solver stopped, see [`get_reason`](@ref).
"""
mutable struct RecordStoppingReason <: RecordAction
    recorded_values::Vector{String}
end
RecordStoppingReason() = RecordStoppingReason(String[])
function (rsr::RecordStoppingReason)(
    ::AbstractManoptProblem, ams::AbstractManoptSolverState, k::Int
)
    s = get_reason(get_stopping_criterion(ams))
    return (length(s) > 0) && record_or_reset!(rsr, s, k)
end
show(io::IO, ::RecordStoppingReason) = print(io, "RecordStoppingReason()")
status_summary(di::RecordStoppingReason) = ":Stop"

@doc """
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
function (r::RecordTime)(p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    # At iteration zero also reset start
    (k == 0) && (r.start = Nanosecond(time_ns()))
    t = Nanosecond(time_ns()) - r.start
    (r.mode == :iterative) && (r.start = Nanosecond(time_ns()))
    if r.mode == :total
        # only record at end (if `stop_solver` returns true)
        return record_or_reset!(r, t, (k > 0 && stop_solver!(p, s, k)) ? k : 0)
    else
        return record_or_reset!(r, t, k)
    end
end
function show(io::IO, ri::RecordTime)
    return print(io, "RecordTime(; mode=:$(ri.mode))")
end
status_summary(ri::RecordTime) = (ri.mode === :iterative ? ":IterativeTime" : ":Time")

#
# Factory
#

@doc """
    RecordFactory(s::AbstractManoptSolverState, a)

Generate a dictionary of [`RecordAction`](@ref)s.

First all `Symbol`s `String`, [`RecordAction`](@ref)s and numbers are collected,
excluding `:Stop` and `:WhenActive`.
This collected vector is added to the `:Iteration => [...]` pair.
`:Stop` is added as `:StoppingCriterion` to the `:Stop => [...]` pair.
If any of these two pairs does not exist, it is pairs are created when adding the corresponding symbols

For each `Pair` of a `Symbol` and a `Vector`, the [`RecordGroupFactory`](@ref)
is called for the `Vector` and the result is added to the debug dictionary's entry
with said symbol. This is wrapped into the [`RecordWhenActive`](@ref),
when the `:WhenActive` symbol is present

# Return value

A dictionary for the different entry points where debug can happen, each containing
a [`RecordAction`](@ref) to call.

Note that upon the initialisation all dictionaries but the `:StartAlgorithm`
one are called with an `i=0` for reset.
"""
function RecordFactory(s::AbstractManoptSolverState, a::Array{<:Any,1})
    # filter out :Iteration defaults
    # filter numbers & stop & pairs (pairs handles separately, numbers at the end)
    iter_entries = filter(
        x ->
            !isa(x, Pair{Symbol,T} where {T}) && (x ∉ [:Stop, :WhenActive]) && !isa(x, Int),
        a,
    )
    # Filter pairs
    b = filter(x -> isa(x, Pair{Symbol,T} where {T}), a)
    # Push this to the :Iteration if that exists or add that pair
    i = findlast(x -> (isa(x, Pair)) && (x.first == :Iteration), b)
    if !isnothing(i)
        iter = popat!(b, i) #
        b = [b..., :Iteration => [iter.second..., iter_entries...]]
    else
        (length(iter_entries) > 0) && (b = [b..., :Iteration => iter_entries])
    end
    # Push a StoppingCriterion to `:Stop` if that exists or add such a pair
    if (:Stop in a)
        i = findlast(x -> (isa(x, Pair)) && (x.first == :Stop), b)
        if !isnothing(i)
            stop = popat!(b, i) #
            b = [b..., :Stop => [stop.second..., RecordActionFactory(s, :Stop)]]
        else # regenerate since the type of b maybe has to be changed
            b = [b..., :Stop => [RecordActionFactory(s, :Stop)]]
        end
    end
    dictionary = Dict{Symbol,RecordAction}()
    # Look for a global number -> RecordEvery
    e = filter(x -> isa(x, Int), a)
    ae = length(e) > 0 ? last(e) : 0
    # Run through all (updated) pairs
    for d in b
        dbg = RecordGroupFactory(s, d.second)
        (:WhenActive in a) && (dbg = RecordWhenActive(dbg))
        # Add RecordEvery to all but Start and Stop
        (!(d.first in [:Start, :Stop]) && (ae > 0)) && (dbg = RecordEvery(dbg, ae))
        dictionary[d.first] = dbg
    end
    return dictionary
end
RecordFactory(s::AbstractManoptSolverState, a) = RecordFactory(s, [a])
@doc """
    RecordGroupFactory(s::AbstractManoptSolverState, a)

Generate a [`RecordGroup`] of [`RecordAction`](@ref)s. The following rules are used

1. Any `Symbol` contained in `a` is passed to [`RecordActionFactory`](@ref RecordActionFactory(s::AbstractManoptSolverState, ::Symbol))
2. Any [`RecordAction`](@ref) is included as is.
Any Pair of a `RecordAction` and a symbol, that is in order `RecordCost() => :A` is handled,
that the corresponding record action can later be accessed as `g[:A]`, where `g`is the record group generated here.

If this results in more than one [`RecordAction`](@ref) a [`RecordGroup`](@ref) of these is build.

If any integers are present, the last of these is used to wrap the group in a
[`RecordEvery`](@ref)`(k)`.

If `:WhenActive` is present, the resulting Action is wrapped in [`RecordWhenActive`](@ref),
making it deactivatable by its parent solver.
"""
function RecordGroupFactory(s::AbstractManoptSolverState, a::Array{<:Any,1})
    # filter out every
    group = Array{Union{<:RecordAction,Pair{<:RecordAction,Symbol}},1}()
    for e in filter(x -> !isa(x, Int) && (x ∉ [:WhenActive]), a) # filter `Int` and Active
        if e isa Symbol # factory for this symbol, store in a pair (for better access later)
            push!(group, RecordActionFactory(s, e) => e)
        elseif e isa Pair{<:RecordAction,Symbol} #already a generated action => symbol to store at
            push!(group, e)
        else # process the others as elements for an action factory
            push!(group, RecordActionFactory(s, e))
        end
    end
    (length(group) > 1) && (record = RecordGroup(group))
    (length(group) == 1) &&
        (record = first(group) isa RecordAction ? first(group) : first(group).first)
    # filter integer numbers
    e = filter(x -> isa(x, Int), a)
    if length(e) > 0
        record = RecordEvery(record, last(e))
    end
    (:WhenActive in a) && (record = (RecordWhenActive(record)))
    return record
end
function RecordGroupFactory(
    s::AbstractManoptSolverState, symbol::Union{Symbol,<:RecordAction}
)
    return RecordActionFactory(s, symbol)
end

@doc """
    RecordActionFactory(s::AbstractManoptSolverState, a)

create a [`RecordAction`](@ref) where

* a [`RecordAction`](@ref) is passed through
* a [`Symbol`] creates
  * `:Change`        to record the change of the iterates, see [`RecordChange`](@ref)
  * `:Gradient`      to record the gradient, see [`RecordGradient`](@ref)
  * `:GradientNorm   to record the norm of the gradient, see [`RecordGradientNorm`](@ref)
  * `:Iterate`       to record the iterate
  * `:Iteration`     to record the current iteration number
  * `IterativeTime`  to record the time iteratively
  * `:Cost`          to record the current cost function value
  * `:Stepsize`      to record the current step size
  * `:Time`          to record the total time taken after every iteration
  * `:IterativeTime` to record the times taken for each iteration.

and every other symbol is passed to [`RecordEntry`](@ref), which results in recording the
field of the state with the symbol indicating the field of the solver to record.
"""
RecordActionFactory(::AbstractManoptSolverState, a::RecordAction) = a
RecordActionFactory(::AbstractManoptSolverState, sa::Pair{<:RecordAction,Symbol}) = sa
function RecordActionFactory(s::AbstractManoptSolverState, symbol::Symbol)
    (symbol == :Change) && return RecordChange()
    (symbol == :Cost) && return RecordCost()
    (symbol == :Gradient) && return RecordGradient(get_gradient(s))
    (symbol == :GradientNorm) && return RecordGradientNorm()
    (symbol == :Iterate) && return RecordIterate(get_iterate(s))
    (symbol == :Iteration) && return RecordIteration()
    (symbol == :IterativeTime) && return RecordTime(; mode=:iterative)
    (symbol == :Stepsize) && return RecordStepsize()
    (symbol == :Stop) && return RecordStoppingReason()
    (symbol == :Subsolver) && return RecordSubsolver()
    (symbol == :Time) && return RecordTime(; mode=:cumulative)
    return RecordEntry(getfield(s, symbol), symbol)
end
@doc """
    RecordActionFactory(s::AbstractManoptSolverState, t::Tuple{Symbol, T}) where {T}

create a [`RecordAction`](@ref) where

* (`:Subsolver`, s) creates a [`RecordSubsolver`](@ref) with `record=` set to the second tuple entry

For other symbol the second entry is ignored and the symbol is used to generate a [`RecordEntry`](@ref)
recording the field with the name `symbol` of `s`.
"""
function RecordActionFactory(s::AbstractManoptSolverState, t::Tuple{Symbol,T}) where {T}
    (t[1] == :Subsolver) && return RecordSubsolver(; record=t[2])
    return RecordEntry(getfield(s, t[1]), t[1])
end
