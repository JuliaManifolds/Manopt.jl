#
#
# Record Action interface

@doc """
    RecordAction

A `RecordAction` is a small functor to record values.
The usual call is given by

    (amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k) -> s

that performs the record for the current problem and solver combination, and where `k` is
the current iteration.

By convention `k=0` is interpreted as “for initialization only”: only
initialize internal values, but do not trigger any record. Note that the record is
called from within [`stop_solver!`](@ref), which returns `true` afterwards.

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

Append to any [`AbstractManoptSolverState`](@ref) the decorator with record capability.
Internally a dictionary is kept that stores a [`RecordAction`](@ref) for
several concurrent modes using a `Symbol` as reference.
The default mode is `:Iteration`, which is used to store information that is recorded during
the iterations. `RecordAction`s might be added to `:Start` or `:Stop` to record values at the
beginning or for the stopping time point, respectively.

The original state can still be accessed using the [`get_state`](@ref) function.

# Fields

* `state`:            the state that is extended by record information
* `recordDictionary`: a `NamedTuple` of [`RecordAction`](@ref)s to keep track of all
  different recorded values

# Constructors

    RecordSolverState(s, dR)

Construct a record decorated [`AbstractManoptSolverState`](@ref), where `dR` can be

* a [`RecordAction`](@ref), then it is stored within the dictionary at `:Iteration`.
* a `Dict{Symbol,RecordAction}`.
* an `Array` of [`RecordAction`](@ref)s, `Symbol`s and `String`s, which is passed to
  the [`RecordFactory`](@ref).
* a single `Symbol`, which is also passed to the [`RecordFactory`](@ref).
"""
mutable struct RecordSolverState{S <: AbstractManoptSolverState, TRD <: NamedTuple} <:
    AbstractManoptSolverState
    state::S
    recordDictionary::TRD
    function RecordSolverState{S}(s::S; kwargs...) where {S <: AbstractManoptSolverState}
        return new{S, typeof(values(kwargs))}(s, values(kwargs))
    end
end
function RecordSolverState(s::S, dR::RecordAction) where {S <: AbstractManoptSolverState}
    return RecordSolverState{S}(s; Iteration = dR)
end
function RecordSolverState(
        s::S, dR::Dict{Symbol, <:RecordAction}
    ) where {S <: AbstractManoptSolverState}
    return RecordSolverState{S}(s; dR...)
end
function RecordSolverState(s::S, format::Vector{<:Any}) where {S <: AbstractManoptSolverState}
    return RecordSolverState{S}(s; RecordFactory(get_state(s), format)...)
end
function RecordSolverState(s::S, symbol::Symbol) where {S <: AbstractManoptSolverState}
    return RecordSolverState{S}(s; RecordFactory(get_state(s), symbol)...)
end
function status_summary(rst::RecordSolverState; context::Symbol = :default)
    (context === :short) && return repr(rst)
    (context === :inline) && (return "A RecordSolverState for $(status_summary(rst.state; context = context))")
    if length(rst.recordDictionary) > 0
        return """
        $(status_summary(rst.state; context = context))

        ## Record
        $(rst.recordDictionary)
        """
    else # We indicate there is a record but no registered recordings
        return """
        $(status_summary(rst.state; context = context))

        ## Record
        No recordings registered.
        """
    end
end
# 2-argument show, used by Array show, print(obj) and repr(obj), keep it short
function Base.show(io::IO, obj::RecordSolverState)
    return print(io, "RecordSolverState($(obj.state), $(obj.recordDictionary))")
end

dispatch_state_decorator(::RecordSolverState) = Val(true)

@doc """
    has_record(s::AbstractManoptSolverState)

Indicate whether the [`AbstractManoptSolverState`](@ref) `s` is decorated with
a [`RecordSolverState`](@ref).
"""
has_record(::RecordSolverState) = true
has_record(s::AbstractManoptSolverState) = _has_record(s, dispatch_state_decorator(s))
_has_record(s::AbstractManoptSolverState, ::Val{true}) = has_record(s.state)
_has_record(::AbstractManoptSolverState, ::Val{false}) = false

"""
    set_parameter!(rss::RecordSolverState, ::Val{:Record}, args...)

Set certain values specified by `args...` into the elements of the `recordDictionary`.
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
# Resolve an ambiguity since this also exists for abstract state
function set_parameter!(rss::RecordSolverState, v::Val{:StoppingCriterion}, args...)
    return set_parameter!(rss.state, v, args...)
end
# all other pass through
function get_parameter(rss::RecordSolverState, v::Val{T}, args...) where {T}
    return get_parameter(rss.state, v, args...)
end

@doc """
    get_record_state(s::AbstractManoptSolverState)

Return the [`RecordSolverState`](@ref) among the decorators of the [`AbstractManoptSolverState`](@ref) `s`.
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
    get_record_action(s::AbstractManoptSolverState, symbol::Symbol=:Iteration)

Return the action contained in the (first) [`RecordSolverState`](@ref) decorator within the [`AbstractManoptSolverState`](@ref) `s`.
"""
function get_record_action(s::AbstractManoptSolverState, symbol::Symbol = :Iteration)
    if haskey(s.recordDictionary, symbol)
        return s.recordDictionary[symbol]
    else
        error("No record known for key :$symbol found")
    end
end
@doc """
    get_record(s::RecordSolverState[, symbol=:Iteration])

Return the recorded values from within the [`RecordSolverState`](@ref) `s` that were
recorded with respect to the `Symbol` `symbol` as an `Array`. The default refers to
any recordings during an `:Iteration`.

When called with arbitrary [`AbstractManoptSolverState`](@ref), this method looks for the
[`RecordSolverState`](@ref) decorator and calls `get_record` on the decorator.
"""
function get_record(s::RecordSolverState, symbol::Symbol = :Iteration)
    return get_record(get_record_action(s, symbol))
end
@doc """
    get_record(s::RecordSolverState[, symbol=:Iteration], i...)

Return the recorded values from within the [`RecordSolverState`](@ref) `s` that were
recorded with respect to the symbol `symbol` as an `Array`.
The default refers to any recordings during an `:Iteration`.

The following arguments `i...` can be used to access further elements of that recording,
either by an index `i` or by a further symbol addressing the recorded elements.
"""
function get_record(s::RecordSolverState, symbol::Symbol, i...)
    return get_record(get_record_action(s, symbol), i...)
end
function get_record(s::AbstractManoptSolverState, symbol::Symbol = :Iteration, i...)
    return get_record(get_record_state(s), symbol, i...)
end

@doc """
    get_record(r::RecordAction)

Return the recorded values stored within a [`RecordAction`](@ref) `r`.
"""
get_record(r::RecordAction) = r.recorded_values
get_record(r::RecordAction, k) = r.recorded_values

"""
    getindex(rs::RecordSolverState, s::Symbol)
    rs[s]

Get the recorded values for recorded type `s`, see [`get_record`](@ref) for details.

    getindex(rs::RecordSolverState, s::Symbol, i...)
    rs[s, i...]

Access the recording of type `s` and call its [`RecordAction`](@ref) with `[i...]`.
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
    return if k > 0
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

Record only every ``k``-th iteration.
Otherwise (optionally, but activated by default) just update internal tracking
values.

This method does not perform any record itself but relies on its children's methods.
"""
mutable struct RecordEvery <: RecordAction
    record::RecordAction
    every::Int
    always_update::Bool
    function RecordEvery(r::RecordAction, every::Int = 1, always_update::Bool = true)
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
        ams, Val(:SubState), Val(:Record), Val(:Activity), !(k < 1) && (rem(k + 1, re.every) == 0)
    )
    return nothing
end
function Base.show(io::IO, re::RecordEvery)
    return print(io, "RecordEvery($(re.record), $(re.every), $(re.always_update))")
end
function status_summary(re::RecordEvery; context::Symbol = :default)
    if context === :short
        s = ""
        if re.record isa RecordGroup
            s = status_summary(re.record; context = context)[2:(end - 1)]
        else
            s = "$(status_summary(re.record; context = context))"
        end
        return "[$s, $(re.every)]"
    end
    s = "every $(re.every)$(_ordinal_suffix(re.every))"
    (re.every == 1) && (s = "every")
    (context === :inline) && return "A RecordAction that records its inner action $s iteration"
    return """
    A RecordAction that records $s iteration with
    $(_MANOPT_INDENT)$(_in_str(status_summary(re.record; context = context); indent = 1))
    """
end
get_record(r::RecordEvery) = get_record(r.record)
get_record(r::RecordEvery, k) = get_record(r.record, k)
getindex(r::RecordEvery, k) = get_record(r, k)

"""
    RecordGroup <: RecordAction

Group a set of [`RecordAction`](@ref)s into one action, where the internal [`RecordAction`](@ref)s
act independently, but the results can be collected in a grouped fashion, a tuple per call of this group.
The entries can be later addressed either by index or by semantic `Symbol`s.

# Constructors

    RecordGroup(g::Array{<:RecordAction, 1})

Construct a group consisting of an `Array` of [`RecordAction`](@ref)s `g`.

    RecordGroup(g, symbols::Dict{Symbol,Int})

Additionally store a dictionary that maps `Symbol`s to indices within `g`.

# Examples

    g1 = RecordGroup([RecordIteration(), RecordCost()])

A `RecordGroup` to record the current iteration and the cost. The cost can then be accessed using `get_record(g1, 2)` or `g1[2]`.

    g2 = RecordGroup([RecordIteration(), RecordCost()], Dict(:Cost => 2))

A `RecordGroup` to record the current iteration and the cost, which can then be accessed using `get_record(g2, :Cost)` or `g2[:Cost]`.

    g3 = RecordGroup([RecordIteration(), RecordCost() => :Cost])

A `RecordGroup` identical to the previous constructor, just a little easier to use.
To access all recordings of the second entry of this last `g3` you can do either `g3[2]` or `g3[:Cost]`,
the first one can only be accessed by `g3[1]`, since no symbol was given here.
"""
mutable struct RecordGroup <: RecordAction
    group::Array{RecordAction, 1}
    indexSymbols::Dict{Symbol, Int}
    function RecordGroup(
            g::Array{<:RecordAction, 1}, symbols::Dict{Symbol, Int} = Dict{Symbol, Int}()
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
            records::Vector, # assumed: {<:Union{<:RecordAction,Pair{<:RecordAction,Symbol}, rest ignored
        )
        g = Array{RecordAction, 1}()
        si = Dict{Symbol, Int}()
        for i in 1:length(records)
            if records[i] isa RecordAction
                push!(g, records[i])
            elseif records[i] isa Pair{<:RecordAction, Symbol}
                push!(g, records[i].first)
                push!(si, records[i].second => i)
            else
                error("Unrecognized element of recording $(repr(records[i])) at entry $i.")
            end
        end
        return RecordGroup(g, si)
    end
    RecordGroup() = new(Array{RecordAction, 1}(), Dict{Symbol, Int}())
end
function (d::RecordGroup)(p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    for ri in d.group
        ri(p, s, k)
    end
    return
end
function status_summary(rg::RecordGroup; context::Symbol = :default)
    (context === :short) && (return "[$(join(["$(status_summary(ri; context = context))" for ri in rg.group], ", "))]")
    (context === :inline) && (return "A group of $(length(rg.group)) RecordActions")
    return "A group of $(length(rg.group)) RecordActions:\n $(join(["* $(status_summary(ri; context = context))" for ri in rg.group], "\n"))\n"
end
function Base.show(io::IO, rg::RecordGroup)
    s = join(["$(ri)" for ri in rg.group], ", ")
    return print(io, "RecordGroup([$s])")
end

@doc """
    get_record(r::RecordGroup)

Return an array of tuples, where each tuple is a recorded set per iteration or record call.

    get_record(r::RecordGroup, i)

Return an array of values corresponding to the ``i``-th entry in this record group.

    get_record(r::RecordGroup, s::Symbol)

Return an array of recorded values with respect to the symbol `s`, see [`RecordGroup`](@ref).

    get_record(r::RecordGroup, s1::Symbol, s2::Symbol,...)

Return an array of tuples, where each tuple is a recorded set corresponding to the symbols `s1, s2,...` per iteration / record call.
"""
get_record(r::RecordGroup) = length(r.group) > 0 ? [zip(get_record.(r.group)...)...] : []
get_record(r::RecordGroup, i) = get_record(r.group[i])
get_record(r::RecordGroup, s::Symbol) = get_record(r.group[r.indexSymbols[s]])
function get_record(r::RecordGroup, s::NTuple{N, Symbol}) where {N}
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

Return an array of recorded values with respect to the symbol `s`, the symbols from the tuple `sT` or the index `i`.
See [`get_record`](@ref get_record(r::RecordGroup)) for details.
"""
getindex(::RecordGroup, ::Any...)
getindex(r::RecordGroup, s::Symbol) = get_record(r, s)
getindex(r::RecordGroup, s::NTuple{N, Symbol}) where {N} = get_record(r, s)
getindex(r::RecordGroup, i) = get_record(r, i)

@doc """
    RecordSubsolver <: RecordAction

Record the current sub solver's recording, by calling [`get_record`](@ref)
on the sub state with the symbols stored in `record`.

# Fields

* `recorded_values`: an array to store the recorded values
* `record`: arguments for [`get_record`](@ref). Defaults to just one symbol `:Iteration`, but could be set to also record the `:Stop` action.

# Constructor

    RecordSubsolver(; record=:Iteration, record_type=eltype([]))
"""
mutable struct RecordSubsolver{R} <: RecordAction
    recorded_values::Vector{R}
    record::Vector{Symbol}
end
function RecordSubsolver(;
        record::Union{Symbol, Vector{Symbol}} = :Iteration, record_type = eltype([])
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
function Base.show(io::IO, rsr::RecordSubsolver{R}) where {R}
    return print(io, "RecordSubsolver(; record=$(rsr.record), record_type=$R)")
end
function status_summary(rsr::RecordSubsolver{R}; context::Symbol = :default) where {R}
    (context === :short) && return ":Subsolver"
    (context === :inline) && return "A RecordAction to specify something to record from each subsolver run"
    return """
    A RecordAction to record elements from each subsolver run of type $R.

    ## Recorded values
    The following recorded symbols from the sub state are recorded in every iteration of the (outer) solver.
    $(join([ "  * :$(s)" for s in rsr.record], "\n"))
    """
end

@doc """
    RecordWhenActive <: RecordAction

A record action that only records if the `active` boolean is set to true.
This can be set from outside and is for example triggered by [`RecordEvery`](@ref)
on recordings of a subsolver. While this is for sub solvers maybe not completely necessary,
recording values that are never accessible, is not that useful.

# Fields

* `active`:        a boolean that can be (de-)activated from outside to turn recording on/off
* `always_update`: whether or not to call the inner record action with nonpositive iterations (init/reset)

# Constructor

    RecordWhenActive(r::RecordAction, active=true, always_update=true)
"""
mutable struct RecordWhenActive{R <: RecordAction} <: RecordAction
    record::R
    active::Bool
    always_update::Bool
    function RecordWhenActive(
            r::R, active::Bool = true, always_update::Bool = true
        ) where {R <: RecordAction}
        return new{R}(r, active, always_update)
    end
end
function (rwa::RecordWhenActive)(
        amp::AbstractManoptProblem, ams::AbstractManoptSolverState, k::Int
    )
    return if rwa.active
        rwa.record(amp, ams, k)
    elseif (rwa.always_update) && (k <= 0)
        rwa.record(amp, ams, k)
    end
end
function Base.show(io::IO, rwa::RecordWhenActive)
    return print(io, "RecordWhenActive($(rwa.record), $(rwa.active), $(rwa.always_update))")
end
function status_summary(rwa::RecordWhenActive; context::Symbol = :default)
    (context === :short) && (return repr(rwa))
    (context === :inline) && (return "A RecordAction that only records its inner action when active (currently: $(rwa.active ? "" : "in")active)")
    return """
    Record the following only, when active (currently: $(rwa.active ? "" : "in")active)
    $(_in_str(status_summary(rwa.record; context = context), indent = 1, headers = 0))
    """
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
# Concrete Record Actions
#
