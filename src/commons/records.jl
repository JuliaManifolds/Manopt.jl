@doc """
    RecordChange <: RecordAction

record the amount of change of the iterate (see [`get_iterate`](@ref)`(s)` of the [`AbstractManoptSolverState`](@ref))
during the last iteration.

# Fields

* `storage`                   : a [`StoreStateAction`](@ref) to store (at least) the last
  iterate to use this as the last value (to compute the change) serving as a potential cache
  shared with other components of the solver.
$(_kwargs(:inverse_retraction_method; p = ""))
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
        TInvRetr <: AbstractInverseRetractionMethod, TStorage <: StoreStateAction,
    } <: RecordAction
    recorded_values::Vector{Float64}
    storage::TStorage
    inverse_retraction_method::TInvRetr
    function RecordChange(
            M::AbstractManifold = DefaultManifold();
            storage::Union{Nothing, StoreStateAction} = nothing,
            inverse_retraction_method::IRT = default_inverse_retraction_method(M),
        ) where {IRT <: AbstractInverseRetractionMethod}
        irm = inverse_retraction_method
        if isnothing(storage)
            if M isa DefaultManifold
                storage = StoreStateAction(M; store_fields = [:Iterate])
            else
                storage = StoreStateAction(M; store_points = Tuple{:Iterate})
            end
        end
        return new{typeof(irm), typeof(storage)}(Vector{Float64}(), storage, irm)
    end
    function RecordChange(
            p, a::StoreStateAction = StoreStateAction([:Iterate]);
            manifold::AbstractManifold = DefaultManifold(1),
            inverse_retraction_method::IRT = default_inverse_retraction_method(manifold, typeof(p)),
        ) where {IRT <: AbstractInverseRetractionMethod}
        update_storage!(a, Dict(:Iterate => p))
        return new{IRT, typeof(a)}(Vector{Float64}(), a, inverse_retraction_method)
    end
end
function (r::RecordChange)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    M = get_manifold(amp)
    record_or_reset!(
        r,
        if has_storage(r.storage, PointStorageKey(:Iterate))
            distance(
                M,
                get_iterate(s), get_storage(r.storage, PointStorageKey(:Iterate)),
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
function Base.show(io::IO, rc::RecordChange)
    return print(
        io, "RecordChange(; inverse_retraction_method=$(rc.inverse_retraction_method))"
    )
end
function status_summary(::RecordChange; context::Symbol = :default)
    (context === :short) && return ":Change"
    return "A RecordAction to record the change of the iterate"
end

@doc """
    RecordCost <: RecordAction

Record the current cost function value, see [`get_cost`](@ref).

# Fields

* `recorded_values` : to store the recorded values

# Constructor

    RecordCost()
"""
mutable struct RecordCost <: RecordAction
    recorded_values::Array{Float64, 1}
    RecordCost() = new(Array{Float64, 1}())
end
function (r::RecordCost)(amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    return record_or_reset!(r, get_cost(amp, get_iterate(s)), k)
end
show(io::IO, ::RecordCost) = print(io, "RecordCost()")
function status_summary(::RecordCost; context::Symbol = :default)
    (context === :short) && return ":Cost"
    return "A RecordAction to record the cost value"
end

@doc """
    RecordEntry{T} <: RecordAction

record a certain fields entry of type {T} during the iterates

# Fields

* `recorded_values` : the recorded values of the entry
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
    recorded_values::Array{T, 1}
    field::Symbol
    RecordEntry{T}(f::Symbol) where {T} = new(Array{T, 1}(), f)
end
RecordEntry(::T, f::Symbol) where {T} = RecordEntry{T}(f)
RecordEntry(d::DataType, f::Symbol) = RecordEntry{d}(f)
function (r::RecordEntry{T})(
        ::AbstractManoptProblem, s::AbstractManoptSolverState, i
    ) where {T}
    return record_or_reset!(r, getfield(s, r.field), i)
end
function Base.show(io::IO, ra::RecordEntry)
    return print(io, "RecordEntry(:$(ra.field))")
end
function status_summary(ra::RecordEntry; context::Symbol = :default)
    (context === :short) && return ":$(ra.field)"
    return "A RecordAction to record the solver state field :$(ra.field)"
end

"""
    RecordDualIterate(X)

Create a [`RecordAction`](@ref) that records the dual iterate,
a [`RecordEntry`](@ref) of the field `X` of the state.
"""
RecordDualIterate(X) = RecordEntry(X, :X)

"""
    RecordDualBaseIterate(n)

Create a [`RecordAction`](@ref) that records the dual base point,
a [`RecordEntry`](@ref) of the field `n` of the state.
"""
RecordDualBaseIterate(n) = RecordEntry(n, :n)


@doc """
    RecordEntryChange{T} <: RecordAction

record a certain entries change during iterates

# Additional fields

* `recorded_values` : the recorded change values
* `field`           : Symbol the field can be accessed with within [`AbstractManoptSolverState`](@ref)
* `distance`        : function `(amp, ams, x1, x2)` to compute the change/distance between two values of the entry
* `storage`         : a [`StoreStateAction`](@ref) to store (at least) the last value of the entry `field`

# Constructor

    RecordEntryChange(f::Symbol, d, a::StoreStateAction=StoreStateAction([f]))
"""
mutable struct RecordEntryChange{TStorage <: StoreStateAction} <: RecordAction
    recorded_values::Vector{Float64}
    field::Symbol
    distance::Any
    storage::TStorage
    function RecordEntryChange(f::Symbol, d, a::StoreStateAction = StoreStateAction([f]))
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
function Base.show(io::IO, ra::RecordEntryChange)
    return print(io, "RecordEntryChange(:$(ra.field), $(ra.distance))")
end
function status_summary(ra::RecordEntryChange; context::Symbol = :default)
    (context === :short) && return repr(ra)
    return "A RecordAction to record the solver state field's :$(ra.field) change using the function $(ra.distance)"
end


"""
    RecordDualBaseChange()

Create a [`RecordAction`](@ref) that records the dual base point change,
a [`RecordEntryChange`](@ref) of the field `n` with distance to the last value to store a value.
"""
function RecordDualBaseChange()
    return RecordEntryChange(
        :n,
        (amp, ams, x, y) -> distance(get_manifold(amp, 2), x, y, ams.inverse_retraction_method_dual),
    )
end

"""
    RecordDualChange()

Create a [`RecordAction`](@ref) that records the change of the dual iterate,
a [`RecordEntryChange`](@ref) of the field `X` with distance to the last value to store a value.
"""
function RecordDualChange()
    storage = StoreStateAction([:X, :n])
    return RecordEntryChange(
        :X,
        (amp, ams, X, X_old) -> begin
            N = get_manifold(amp, 2)
            n_old = has_storage(storage, :n) ? get_storage(storage, :n) : ams.n
            return norm(
                N, ams.n,
                vector_transport_to(N, n_old, X_old, ams.n, ams.vector_transport_method_dual) - X,
            )
        end,
        storage,
    )
end


@doc """
    RecordGradient <: RecordAction

record the gradient evaluated at the current iterate

# Constructor
    RecordGradient(X)

initialize the [`RecordAction`](@ref) to the corresponding type of the tangent vector.
"""
mutable struct RecordGradient{T} <: RecordAction
    recorded_values::Array{T, 1}
    RecordGradient{T}() where {T} = new(Array{T, 1}())
end
RecordGradient(::T) where {T} = RecordGradient{T}()
function (r::RecordGradient{T})(
        ::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    ) where {T}
    return record_or_reset!(r, get_gradient(s), k)
end
show(io::IO, ::RecordGradient{T}) where {T} = print(io, "RecordGradient($T)")
function status_summary(rg::RecordGradient; context::Symbol = :default)
    (context === :short) && return ":Gradient"
    return "A RecordAction to record the current gradient"
end
@doc """
    RecordGradientNorm{R<:Real} <: RecordAction

record the norm of the current gradient

## Constructor
    RecordGradientNorm(r::Type{<:Real}=Float64)
"""
mutable struct RecordGradientNorm{R <: Real} <: RecordAction
    recorded_values::Array{R, 1}
    RecordGradientNorm(r::Type{<:Real} = Float64) = new{r}(Array{r, 1}())
end
function (r::RecordGradientNorm)(
        mp::AbstractManoptProblem, ast::AbstractManoptSolverState, k::Int
    )
    M = get_manifold(mp)
    return record_or_reset!(r, norm(M, get_iterate(ast), get_gradient(ast)), k)
end
show(io::IO, ::RecordGradientNorm) = print(io, "RecordGradientNorm()")
function status_summary(rg::RecordGradientNorm; context::Symbol = :default)
    (context === :short) && return ":GradientNorm"
    return "A RecordAction to record the current gradient norm"
end

@doc """
    RecordIterate <: RecordAction

record the iterate

# Constructors
    RecordIterate(p0)

initialize the iterate record array to the type of `p0`, which indicates the kind of iterate

    RecordIterate(T::DataType)

initialize the iterate record array to the data type `T`.
"""
mutable struct RecordIterate{T} <: RecordAction
    recorded_values::Array{T, 1}
    RecordIterate{T}() where {T} = new(Array{T, 1}())
end
RecordIterate(::T) where {T} = RecordIterate{T}()
RecordIterate(d::DataType) = RecordIterate{d}()
function RecordIterate()
    return throw(
        ErrorException("The iterate's data type has to be provided, RecordIterate(p0).")
    )
end
function (r::RecordIterate{T})(
        ::AbstractManoptProblem, s::AbstractManoptSolverState, i
    ) where {T}
    return record_or_reset!(r, get_iterate(s), i)
end
function Base.show(io::IO, ri::RecordIterate)
    return print(io, "RecordIterate($(eltype(ri.recorded_values)))")
end
function status_summary(di::RecordIterate; context::Symbol = :default)
    (context === :short) && return ":Iterate"
    return "A RecordAction to record the current iterate"
end

@doc """
    RecordIteration <: RecordAction

record the current iteration
"""
mutable struct RecordIteration <: RecordAction
    recorded_values::Array{Int, 1}
    RecordIteration() = new(Array{Int, 1}())
end
function (r::RecordIteration)(::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int)
    return record_or_reset!(r, k, k)
end
show(io::IO, ::RecordIteration) = print(io, "RecordIteration()")
function status_summary(::RecordIteration; context::Symbol = :default)
    (context === :short) && return ":Iteration"
    return "A RecordAction to record the current iteration number"
end

"""
    RecordPrimalChange()

Create a [`RecordAction`](@ref) that records the primal value change,
a [`RecordChange`](@ref), to record the change of the iterate `p`.
"""
RecordPrimalChange() = RecordChange()

"""
    RecordPrimalIterate(p)

Create a [`RecordAction`](@ref) that records the primal point, a [`RecordIterate`](@ref) of the iterate `p`.
"""
RecordPrimalIterate(p) = RecordIterate(p)

"""
    RecordPrimalBaseChange()

Create a [`RecordAction`](@ref) that records the primal base point change,
a [`RecordEntryChange`](@ref) of the field `m` with distance to the last value to store a value.
"""
function RecordPrimalBaseChange()
    return RecordEntryChange(:m, (amp, ams, p1, p2) -> distance(get_manifold(amp, 1), p1, p2))
end

"""
    RecordPrimalBaseIterate(m)

Create a [`RecordAction`](@ref) that records the primal base point, a [`RecordEntry`](@ref) of the field `m` of the state.
"""
RecordPrimalBaseIterate(m) = RecordEntry(m, :m)


@doc """
    RecordProximalParameter{R <: Real} <: RecordAction

record the current proximal point algorithm parameter ``λ_k``, given by `s.λ(k)`
of the corresponding [`AbstractManoptSolverState`](@ref) `s`, for example the
[`CyclicProximalPointState`](@ref).

## Constructor
    RecordProximalParameter(r::Type{<:Real}=Float64)
"""
mutable struct RecordProximalParameter{R <: Real} <: RecordAction
    recorded_values::Array{R, 1}
    RecordProximalParameter(r::Type{<:Real} = Float64) = new{r}(Array{r, 1}())
end
show(io::IO, ::RecordProximalParameter{R}) where {R} = print(io, "RecordProximalParameter($R)")
function status_summary(rg::RecordProximalParameter{R}; context::Symbol = :default) where {R}
    (context === :short) && return ":ProximalParameter"
    return "A RecordAction to record the current proximal parameter (of type $R)"
end

@doc """
    RecordStepsize <: RecordAction

Record the step size.

# Constructor

    RecordStepsize(r::Type{<:Real}=Float64)
"""
mutable struct RecordStepsize{R <: Real} <: RecordAction
    recorded_values::Array{R, 1}
    RecordStepsize(r::Type{<:Real} = Float64) = new{r}(Array{r, 1}())
end
function (r::RecordStepsize)(p::AbstractManoptProblem, s::AbstractManoptSolverState, k)
    return record_or_reset!(r, get_last_stepsize(p, s, k), k)
end
show(io::IO, ::RecordStepsize{R}) where {R} = print(io, "RecordStepsize($R)")
function status_summary(rg::RecordStepsize{R}; context::Symbol = :default) where {R}
    (context === :short) && return ":Stepsize"
    return "A RecordAction to record the current stepsize (of type $R)"
end

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
function status_summary(::RecordStoppingReason; context::Symbol = :default)
    (context === :short) && return ":Stop"
    return "A RecordAction to record the stopping reason"
end

@doc """
    RecordTime <: RecordAction

record the time elapsed during the current iteration.

The three possible modes are
* `:Cumulative` record times without resetting the timer
* `:Iterative` record times with resetting the timer
* `:Total` record a time only at the end of an algorithm (see [`stop_solver!`](@ref))

The default is `:Cumulative`, and any non-listed symbol defaults to using this mode.

# Constructor

    RecordTime(; mode::Symbol=:Cumulative)
"""
mutable struct RecordTime <: RecordAction
    recorded_values::Array{Nanosecond, 1}
    start::Nanosecond
    mode::Symbol
    function RecordTime(; mode::Symbol = :Cumulative)
        return new(Array{Nanosecond, 1}(), Nanosecond(time_ns()), mode)
    end
end
function (r::RecordTime)(p::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    # At initialization and reset (k <= 0) also reset start
    (k <= 0) && (r.start = Nanosecond(time_ns()))
    t = Nanosecond(time_ns()) - r.start
    (r.mode == :Iterative) && (r.start = Nanosecond(time_ns()))
    if r.mode == :Total
        # only record at end (if `stop_solver` returns true)
        return record_or_reset!(r, t, (k > 0 && !stop_solver!(p, s, k)) ? 0 : k)
    else
        return record_or_reset!(r, t, k)
    end
end
function Base.show(io::IO, ri::RecordTime)
    return print(io, "RecordTime(; mode=:$(ri.mode))")
end
function status_summary(ri::RecordTime; context::Symbol = :default)
    (context == :short) && return (ri.mode === :Iterative ? ":IterativeTime" : ":Time")
    # Inline and Default:
    return "A RecordAction for recording times" * (ri.mode == :Iterative ? " iteratively" : ".")
end

#
# Factory
#
@doc """
    RecordFactory(s::AbstractManoptSolverState, a)

Generate a dictionary of [`RecordAction`](@ref)s.

First all `Symbol`s and [`RecordAction`](@ref)s are collected,
excluding `:Stop`, `:WhenActive` and any `Int`.
This collected vector is added to the `:Iteration => [...]` pair.
`:Stop` is added as a [`RecordStoppingReason`](@ref) to the `:Stop => [...]` pair.
If any of these two pairs does not exist, it is created when adding the corresponding entries.

For each `Pair` of a `Symbol` and a `Vector`, the [`RecordGroupFactory`](@ref)
is called for the `Vector` and the result is added to the record dictionary's entry
with said symbol. This is wrapped into the [`RecordWhenActive`](@ref),
when the `:WhenActive` symbol is present.

If an `Int` `k` is present, all entries but `:Start` and `:Stop` are wrapped
into a [`RecordEvery`](@ref)`(k)`.

# Return value

A dictionary for the different entry points where recording can happen, each containing
a [`RecordAction`](@ref) to call.

Note that upon the initialization all dictionaries but the `:Start`
one are called with a `k=-1` for reset.
"""
function RecordFactory(s::AbstractManoptSolverState, a::Array{<:Any, 1})
    # filter out :Iteration defaults
    # filter numbers & stop & pairs (pairs handles separately, numbers at the end)
    iter_entries = filter(
        x ->
        !isa(x, Pair{Symbol, T} where {T}) && (x ∉ [:Stop, :WhenActive]) && !isa(x, Int),
        a,
    )
    # Filter pairs
    b = filter(x -> isa(x, Pair{Symbol, T} where {T}), a)
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
    dictionary = Dict{Symbol, RecordAction}()
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

Generate a [`RecordGroup`](@ref) of [`RecordAction`](@ref)s. The following rules are used

1. Any `Symbol` contained in `a` is passed to [`RecordActionFactory`](@ref RecordActionFactory(s::AbstractManoptSolverState, ::Symbol))
2. Any [`RecordAction`](@ref) is included as is.
Any Pair of a `RecordAction` and a symbol, that is in order `RecordCost() => :A` is handled,
that the corresponding record action can later be accessed as `g[:A]`, where `g` is the record group generated here.

If this results in more than one [`RecordAction`](@ref) a [`RecordGroup`](@ref) of these is build.

If any integers are present, the last of these is used to wrap the group in a
[`RecordEvery`](@ref)`(k)`.

If `:WhenActive` is present, the resulting Action is wrapped in [`RecordWhenActive`](@ref),
making it deactivatable by its parent solver.
"""
function RecordGroupFactory(s::AbstractManoptSolverState, a::Array{<:Any, 1})
    # filter out every
    group = Array{Union{<:RecordAction, Pair{<:RecordAction, Symbol}}, 1}()
    for e in filter(x -> !isa(x, Int) && (x ∉ [:WhenActive]), a) # filter `Int` and Active
        if e isa Symbol # factory for this symbol, store in a pair (for better access later)
            push!(group, RecordActionFactory(s, e) => e)
        elseif e isa Pair{<:RecordAction, Symbol} #already a generated action => symbol to store at
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
        s::AbstractManoptSolverState, symbol::Union{Symbol, <:RecordAction}
    )
    return RecordActionFactory(s, symbol)
end

@doc """
    RecordActionFactory(s::AbstractManoptSolverState, a)

create a [`RecordAction`](@ref) where

* a [`RecordAction`](@ref) is passed through
* a `Symbol` creates
  * `:Change`        to record the change of the iterates, see [`RecordChange`](@ref)
  * `:Cost`          to record the current cost function value
  * `:Gradient`      to record the gradient, see [`RecordGradient`](@ref)
  * `:GradientNorm`  to record the norm of the gradient, see [`RecordGradientNorm`](@ref)
  * `:Iterate`       to record the iterate
  * `:Iteration`     to record the current iteration number
  * `:IterativeTime` to record the times taken for each iteration.
  * `:ProximalParameter` to record the proximal parameter, see [`RecordProximalParameter`](@ref)
  * `:Stepsize`      to record the current step size
  * `:Stop`          to record the reason the solver stopped, see [`RecordStoppingReason`](@ref)
  * `:Subsolver`     to record the sub solver's record, see [`RecordSubsolver`](@ref)
  * `:Time`          to record the total time taken after every iteration

and every other symbol is passed to [`RecordEntry`](@ref), which results in recording the
field of the state with the symbol indicating the field of the solver to record.
"""
RecordActionFactory(::AbstractManoptSolverState, a::RecordAction) = a
RecordActionFactory(::AbstractManoptSolverState, sa::Pair{<:RecordAction, Symbol}) = sa
function RecordActionFactory(s::AbstractManoptSolverState, symbol::Symbol)
    (symbol == :Change) && return RecordChange()
    (symbol == :Cost) && return RecordCost()
    (symbol == :Gradient) && return RecordGradient(get_gradient(s))
    (symbol == :GradientNorm) && return RecordGradientNorm()
    (symbol == :Iterate) && return RecordIterate(get_iterate(s))
    (symbol == :Iteration) && return RecordIteration()
    (symbol == :IterativeTime) && return RecordTime(; mode = :Iterative)
    (symbol == :ProximalParameter) && return RecordProximalParameter()
    (symbol == :Stepsize) && return RecordStepsize()
    (symbol == :Stop) && return RecordStoppingReason()
    (symbol == :Subsolver) && return RecordSubsolver()
    (symbol == :Time) && return RecordTime(; mode = :Cumulative)
    return RecordEntry(getfield(s, symbol), symbol)
end
@doc """
    RecordActionFactory(s::AbstractManoptSolverState, t::Tuple{Symbol, T}) where {T}

create a [`RecordAction`](@ref) where

* (`:Subsolver`, s) creates a [`RecordSubsolver`](@ref) with `record=` set to the second tuple entry

For any other symbol the second entry is ignored and the symbol is used to generate a [`RecordEntry`](@ref)
recording the field with the name `symbol` of `s`.
"""
function RecordActionFactory(s::AbstractManoptSolverState, t::Tuple{Symbol, T}) where {T}
    (t[1] == :Subsolver) && return RecordSubsolver(; record = t[2])
    return RecordEntry(getfield(s, t[1]), t[1])
end
