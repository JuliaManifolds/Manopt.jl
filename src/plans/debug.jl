@doc raw"""
    DebugAction

A `DebugAction` is a small functor to print/issue debug output.
The usual call is given by `(amp::AbstractManoptProblem, ams::AbstractManoptSolverState, i) -> s`,
where `i` is the current iterate.

By convention `i=0` is interpreted as "For Initialization only", i.e. only debug
info that prints initialization reacts, `i<0` triggers updates of variables
internally but does not trigger any output. Finally `typemin(Int)` is used
to indicate a call from [`stop_solver!`](@ref) that returns true afterwards.

# Fields (assumed by subtypes to exist)
* `print` method to perform the actual print. Can for example be set to a file export,
or to @info. The default is the `print` function on the default `Base.stdout`.
"""
abstract type DebugAction <: AbstractStateAction end

@doc raw"""
    DebugSolverState <: AbstractManoptSolverState

The debug options append to any options a debug functionality, i.e. they act as
a decorator pattern. Internally a `Dict`ionary is kept that stores a
[`DebugAction`](@ref) for several occasions using a `Symbol` as reference.
The default occasion is `:All` and for example solvers join this field with
`:Start`, `:Step` and `:Stop` at the beginning, every iteration or the
end of the algorithm, respectively

The original options can still be accessed using the [`get_state`](@ref) function.

# Fields (defaults in brackets)
* `options` – the options that are extended by debug information
* `debugDictionary` – a `Dict{Symbol,DebugAction}` to keep track of Debug for different actions

# Constructors
    DebugSolverState(o,dA)

construct debug decorated options, where `dD` can be
* a [`DebugAction`](@ref), then it is stored within the dictionary at `:All`
* an `Array` of [`DebugAction`](@ref)s, then it is stored as a
  `debugDictionary` within `:All`.
* a `Dict{Symbol,DebugAction}`.
* an Array of Symbols, String and an Int for the [`DebugFactory`](@ref)
"""
mutable struct DebugSolverState{S<:AbstractManoptSolverState} <: AbstractManoptSolverState
    state::S
    debugDictionary::Dict{Symbol,<:DebugAction}
    function DebugSolverState{S}(
        st::S, dA::Dict{Symbol,<:DebugAction}
    ) where {S<:AbstractManoptSolverState}
        return new(st, dA)
    end
end
function DebugSolverState(st::S, dD::D) where {S<:AbstractManoptSolverState,D<:DebugAction}
    return DebugSolverState{S}(st, Dict(:All => dD))
end
function DebugSolverState(
    st::S, dD::Array{<:DebugAction,1}
) where {S<:AbstractManoptSolverState}
    return DebugSolverState{S}(st, Dict(:All => DebugGroup(dD)))
end
function DebugSolverState(
    st::S, dD::Dict{Symbol,<:DebugAction}
) where {S<:AbstractManoptSolverState}
    return DebugSolverState{S}(st, dD)
end
function DebugSolverState(
    st::S, format::Array{<:Any,1}
) where {S<:AbstractManoptSolverState}
    return DebugSolverState{S}(st, DebugFactory(format))
end
function status_summary(dst::DebugSolverState)
    if length(dst.debugDictionary) > 0
        s = ""
        if length(dst.debugDictionary) == 1 && first(keys(dst.debugDictionary)) === :All
            s = "\n    $(status_summary(dst.debugDictionary[:All]))"
        else
            for (k, v) in dst.debugDictionary
                s = "$s\n    :$k = $(status_summary(v))"
            end
        end
        return """
               $(dst.state)

               ## Debug$s"""
    else # for length 1 the group is equivvalent to the summary of the single state
        return status_summary(dst.state)
    end
end
function show(io::IO, dst::DebugSolverState)
    return print(io, status_summary(dst))
end
dispatch_state_decorator(::DebugSolverState) = Val(true)

#
# Meta Debug Actions
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
function (d::DebugGroup)(p::AbstractManoptProblem, st::AbstractManoptSolverState, i)
    for di in d.group
        di(p, st, i)
    end
end
function status_summary(dg::DebugGroup)
    return "[ $( join(["$(status_summary(di))" for di in dg.group], ", ")) ]"
end
function show(io::IO, dg::DebugGroup)
    s = join(["$(di)" for di in dg.group], ", ")
    return print(io, "DebugGroup([$s])")
end

@doc raw"""
    DebugEvery <: DebugAction

evaluate and print debug only every $i$th iteration. Otherwise no print is performed.
Whether internal variables are updates is determined by `always_update`.

This method does not perform any print itself but relies on it's childrens print.

# Constructor

    DebugEvery(d::DebugAction, every=1, always_update=true)

Initialise the DebugEvery.
"""
mutable struct DebugEvery <: DebugAction
    debug::DebugAction
    every::Int
    always_update::Bool
    function DebugEvery(d::DebugAction, every::Int=1, always_update::Bool=true)
        return new(d, every, always_update)
    end
end
function (d::DebugEvery)(p::AbstractManoptProblem, st::AbstractManoptSolverState, i)
    if (rem(i, d.every) == 0)
        d.debug(p, st, i)
    elseif d.always_update
        d.debug(p, st, -1)
    end
end
function show(io::IO, de::DebugEvery)
    return print(io, "DebugEvery($(de.debug), $(de.every), $(de.always_update))")
end
function status_summary(de::DebugEvery)
    s = ""
    if de.debug isa DebugGroup
        s = status_summary(de.debug)[3:(end - 2)]
    else
        s = "$(de.debug)"
    end
    return "[$s, $(de.every)]"
end
#
# Special single ones
#
@doc raw"""
    DebugChange(M=DefaultManifold())

debug for the amount of change of the iterate (stored in `get_iterate(o)` of the [`AbstractManoptSolverState`](@ref))
during the last iteration. See [`DebugEntryChange`](@ref) for the general case

# Keyword Parameters
* `storage` – (`StoreStateAction( [:Gradient] )`) – (eventually shared) the storage of the previous action
* `prefix` – (`"Last Change:"`) prefix of the debug output (ignored if you set `format`)
* `io` – (`stdout`) default steream to print the debug to.
* `format` - ( `"$prefix %f"`) format to print the output using an sprintf format.
* `inverse_retraction_method` - (`default_inverse_retraction_method(M)`) the inverse retraction to be
  used for approximating distance.
"""
mutable struct DebugChange{IR<:AbstractInverseRetractionMethod} <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    inverse_retraction_method::IR
    function DebugChange(
        M::AbstractManifold=DefaultManifold();
        storage::Union{Nothing,StoreStateAction}=nothing,
        io::IO=stdout,
        prefix::String="Last Change: ",
        format::String="$(prefix)%f",
        manifold::Union{Nothing,AbstractManifold}=nothing,
        invretr::Union{Nothing,AbstractInverseRetractionMethod}=nothing,
        inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
            M
        ),
    )
        irm = inverse_retraction_method
        # Deprecated, remove in Manopt 0.5
        if !isnothing(manifold)
            @warn "The `manifold` keyword is deprecated, use the first positional argument `M`. This keyword for now sets `inverse_retracion_method`."
            irm = default_inverse_retraction_method(manifold)
        end
        if !isnothing(invretr)
            @warn "invretr keyword is deprecated, use `inverse_retraction_method`, which this one overrides for now."
            irm = invretr
        end
        if isnothing(storage)
            if M isa DefaultManifold
                storage = StoreStateAction(M; store_fields=[:Iterate])
            else
                storage = StoreStateAction(M; store_points=Tuple{:Iterate})
            end
        end
        return new{typeof(irm)}(io, format, storage, irm)
    end
end
function (d::DebugChange)(mp::AbstractManoptProblem, st::AbstractManoptSolverState, i)
    M = get_manifold(mp)
    (i > 0) && Printf.format(
        d.io,
        Printf.Format(d.format),
        distance(
            M,
            get_iterate(st),
            get_storage(d.storage, PointStorageKey(:Iterate)),
            d.inverse_retraction_method,
        ),
    )
    d.storage(mp, st, i)
    return nothing
end
function show(io::IO, dc::DebugChange)
    return print(
        io,
        "DebugChange(; format=\"$(dc.format)\", inverse_retraction=$(dc.inverse_retraction_method))",
    )
end
status_summary(dc::DebugChange) = "(:Change, \"$(dc.format)\")"

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
function (d::DebugCost)(p::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int)
    (i >= 0) && Printf.format(d.io, Printf.Format(d.format), get_cost(p, get_iterate(st)))
    return nothing
end
show(io::IO, di::DebugCost) = print(io, "DebugCost(; format=\"$(di.format)\")")
status_summary(di::DebugCost) = "(:Cost, \"$(di.format)\")"

@doc raw"""
    DebugDivider <: DebugAction

print a small `div`ider (default `" | "`).

# Constructor
    DebugDivider(div,print)

"""
mutable struct DebugDivider{TIO<:IO} <: DebugAction
    io::TIO
    divider::String
    DebugDivider(divider=" | "; io::IO=stdout) = new{typeof(io)}(io, divider)
end
function (d::DebugDivider)(::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int)
    if i >= 0 && !isempty(d.divider)
        print(d.io, d.divider)
    end
    return nothing
end
show(io::IO, di::DebugDivider) = print(io, "DebugDivider(; divider=\"$(di.divider)\")")
status_summary(di::DebugDivider) = "\"$(di.divider)\""

@doc raw"""
    DebugEntry <: DebugAction

print a certain fields entry of type {T} during the iterates, where a `format` can be
specified how to print the entry.

# Addidtional Fields
* `field` – Symbol the entry can be accessed with within [`AbstractManoptSolverState`](@ref)

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
function (d::DebugEntry)(::AbstractManoptProblem, st::AbstractManoptSolverState, i)
    (i >= 0) && Printf.format(d.io, Printf.Format(d.format), getfield(st, d.field))
    return nothing
end
function show(io::IO, di::DebugEntry)
    return print(io, "DebugEntry(:$(di.field); format=\"$(di.format)\")")
end

@doc raw"""
    DebugEntryChange{T} <: DebugAction

print a certain entries change during iterates

# Additional Fields
* `print` – (`print`) function to print the result
* `prefix` – (`"Change of :Iterate"`) prefix to the print out
* `format` – (`"$prefix %e"`) format to print (uses the `prefix by default and scientific notation)
* `field` – Symbol the field can be accessed with within [`AbstractManoptSolverState`](@ref)
* `distance` – function (p,o,x1,x2) to compute the change/distance between two values of the entry
* `storage` – a [`StoreStateAction`](@ref) to store the previous value of `:f`

# Constructors

    DebugEntryChange(f,d)

# Keyword arguments

* `io` (`stdout`) an `IOStream`
* `prefix` (`"Change of $f"`)
* `storage` (`StoreStateAction((f,))`) a [`StoreStateAction`](@ref)
* `initial_value` an initial value for the change of `o.field`.
* `format` – (`"$prefix %e"`) format to print the change

"""
mutable struct DebugEntryChange <: DebugAction
    distance::Any
    field::Symbol
    format::String
    io::IO
    storage::StoreStateAction
    function DebugEntryChange(
        f::Symbol,
        d;
        storage::StoreStateAction=StoreStateAction([f]),
        prefix::String="Change of $f:",
        format::String="$prefix%s",
        io::IO=stdout,
        initial_value::Any=NaN,
    )
        if !isa(initial_value, Number) || !isnan(initial_value) #set initial value
            update_storage!(storage, Dict(f => initial_value))
        end
        return new(d, f, format, io, storage)
    end
end
function (d::DebugEntryChange)(
    p::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int
)
    if i == 0
        # on init if field not present -> generate
        !has_storage(d.storage, d.field) && d.storage(p, st, i)
        return nothing
    end
    x = get_storage(d.storage, d.field)
    v = d.distance(p, st, getproperty(st, d.field), x)
    Printf.format(d.io, Printf.Format(d.format), v)
    d.storage(p, st, i)
    return nothing
end
function show(io::IO, dec::DebugEntryChange)
    return print(
        io, "DebugEntryChange(:$(dec.field), $(dec.distance); format=\"$(dec.format)\")"
    )
end

@doc raw"""
    DebugGradientChange()

debug for the amount of change of the gradient (stored in `get_gradient(o)` of the [`AbstractManoptSolverState`](@ref) `o`)
during the last iteration. See [`DebugEntryChange`](@ref) for the general case

# Keyword Parameters
* `storage` – (`StoreStateAction( (:Gradient,) )`) – (eventually shared) the storage of the previous action
* `prefix` – (`"Last Change:"`) prefix of the debug output (ignored if you set `format`)
* `io` – (`stdout`) default steream to print the debug to.
* `format` - ( `"$prefix %f"`) format to print the output using an sprintf format.
"""
mutable struct DebugGradientChange{VTR<:AbstractVectorTransportMethod} <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    vector_transport_method::VTR
    function DebugGradientChange(
        M::AbstractManifold=DefaultManifold();
        storage::Union{Nothing,StoreStateAction}=nothing,
        io::IO=stdout,
        prefix::String="Last Change: ",
        format::String="$(prefix)%f",
        vector_transport_method::VTR=default_vector_transport_method(M),
    ) where {VTR<:AbstractVectorTransportMethod}
        if isnothing(storage)
            if M isa DefaultManifold
                storage = StoreStateAction(M; store_fields=[:Iterate, :Gradient])
            else
                storage = StoreStateAction(
                    M; store_points=[:Iterate], store_vectors=[:Gradient]
                )
            end
        end
        return new{VTR}(io, format, storage, vector_transport_method)
    end
end
function (d::DebugGradientChange)(
    pm::AbstractManoptProblem, st::AbstractManoptSolverState, i
)
    if i > 0
        M = get_manifold(pm)
        p_old = get_storage(d.storage, PointStorageKey(:Iterate))
        X_old = get_storage(d.storage, VectorStorageKey(:Gradient))
        p = get_iterate(st)
        X = get_gradient(st)
        l = norm(
            M, p, X - vector_transport_to(M, p_old, X_old, p, d.vector_transport_method)
        )
        Printf.format(d.io, Printf.Format(d.format), l)
    end
    d.storage(pm, st, i)
    return nothing
end
function show(io::IO, dgc::DebugGradientChange)
    return print(
        io,
        "DebugGradientChange(; format=\"$(dgc.format)\", vector_transport_method=$(dgc.vector_transport_method))",
    )
end
status_summary(di::DebugGradientChange) = "(:GradientChange, \"$(di.format)\")"

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
        prefix=long ? "current iterate:" : "p:",
        format="$prefix %s",
    )
        return new(io, format)
    end
end
function (d::DebugIterate)(::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int)
    (i > 0) && Printf.format(d.io, Printf.Format(d.format), get_iterate(st))
    return nothing
end
function show(io::IO, di::DebugIterate)
    return print(io, "DebugIterate(; format=\"$(di.format)\")")
end
status_summary(di::DebugIterate) = "(:Iterate, \"$(di.format)\")"

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
function (d::DebugIteration)(::AbstractManoptProblem, ::AbstractManoptSolverState, i::Int)
    (i == 0) && print(d.io, "Initial ")
    (i > 0) && Printf.format(d.io, Printf.Format(d.format), i)
    return nothing
end
show(io::IO, di::DebugIteration) = print(io, "DebugIteration(; format=\"$(di.format)\")")
status_summary(di::DebugIteration) = "(:Iteration, \"$(di.format)\")"

@doc raw"""
    DebugMessages <: DebugAction

An [`AbstractManoptSolverState`](@ref) or one of its substeps like a
[`Stepsize`](@ref) might generate warnings throughout their compuations.
This debug can be used to `:print` them display them as `:info` or `:warnings` or even `:error`,
depending on the message type.

# Constructor
    DebugMessages(mode=:Info; io::IO=stdout)

Initialize the messages debug to a certain `mode`. Available modes are
* `:Error` – issue the messages as an error and hence stop at any issue occuring
* `:Info` – issue the messages as an `@info`
* `:Print` – print messages to the steam `io`.
* `:Warning` – issue the messages as a warning
"""
mutable struct DebugMessages <: DebugAction
    io::IO
    mode::Symbol
    DebugMessages(mode::Symbol=:Info; io::IO=stdout) = new(io, mode)
end
function (d::DebugMessages)(::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int)
    msg = get_message(st)
    (i < 0 || length(msg) == 0) && (return nothing)
    (d.mode == :Warning) && (@warn msg; return nothing)
    (d.mode == :Error) && (@error msg; return nothing)
    (d.mode == :Print) && (print(d.io, msg); return nothing)
    #(d.mode == :Info) &&
    (@info msg) # Default
    return nothing
end
show(io::IO, ::DebugMessages) = print(io, "DebugMessages()")
function status_summary(::DebugMessages)
    (mode == :Warning) && return ":WarningMessages"
    (mode == :Info) && return ":InfoMessages"
    (mode == :Error) && return ":ErrorMessages"
    return ":Messages"
end

@doc raw"""
    DebugStoppingCriterion <: DebugAction

print the Reason provided by the stopping criterion. Usually this should be
empty, unless the algorithm stops.
"""
mutable struct DebugStoppingCriterion <: DebugAction
    io::IO
    DebugStoppingCriterion(; io::IO=stdout) = new(io)
end
function (d::DebugStoppingCriterion)(
    ::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int
)
    print(d.io, (i >= 0 || i == typemin(Int)) ? get_reason(st) : "")
    return nothing
end
show(io::IO, ::DebugStoppingCriterion) = print(io, "DebugStoppingCriterion()")
status_summary(::DebugStoppingCriterion) = ":Stop"

@doc raw"""
    DebugTime()

Measure time and print the intervals. Using `start=true` you can start the timer on construction,
for example to measure the runtime of an algorithm overall (adding)

The measured time is rounded using the given `time_accuracy` and printed after [canonicalization](https://docs.julialang.org/en/v1/stdlib/Dates/#Dates.canonicalize).

# Keyword Parameters

* `prefix` – (`"Last Change:"`) prefix of the debug output (ignored if you set `format`)
* `io` – (`stdout`) default steream to print the debug to.
* `format` - ( `"$prefix %s"`) format to print the output using an sprintf format, where `%s` is the canonicalized time`.
* `mode` – (`:cumulative`) whether to display the total time or reset on every call using `:iterative`.
* `start` – (`false`) indicate whether to start the timer on creation or not. Otherwise it might only be started on firsr call.
* `time_accuracy` – (`Millisecond(1)`) round the time to this period before printing the canonicalized time
"""
mutable struct DebugTime <: DebugAction
    io::IO
    format::String
    last_time::Nanosecond
    time_accuracy::Period
    mode::Symbol
    function DebugTime(;
        start=false,
        io::IO=stdout,
        prefix::String="time spent:",
        format::String="$(prefix) %s",
        mode::Symbol=:cumulative,
        time_accuracy::Period=Millisecond(1),
    )
        return new(io, format, Nanosecond(start ? time_ns() : 0), time_accuracy, mode)
    end
end
function (d::DebugTime)(::AbstractManoptProblem, ::AbstractManoptSolverState, i)
    if i == 0 || d.last_time == Nanosecond(0) # init
        d.last_time = Nanosecond(time_ns())
    else
        t = time_ns()
        p = Nanosecond(t) - d.last_time
        Printf.format(
            d.io, Printf.Format(d.format), canonicalize(round(p, d.time_accuracy))
        )
    end
    if d.mode == :iterative
        d.last_time = Nanosecond(time_ns())
    end
    return nothing
end
function show(io::IO, di::DebugTime)
    return print(io, "DebugTime(; format=\"$(di.format)\", mode=:$(di.mode))")
end
function status_summary(di::DebugTime)
    if di.mode === :iterative
        return "(:IterativeTime, \"$(di.format)\")"
    end
    return "(:Time, \"$(di.format)\")"
end
"""
    reset!(d::DebugTime)

reset the internal time of a [`DebugTime`](@ref), that is start from now again.
"""
function reset!(d::DebugTime)
    d.last_time = Nanosecond(time_ns())
    return d
end
"""
    stop!(d::DebugTime)

stop the reset the internal time of a [`DebugTime`](@ref), that is set the time to 0 (undefined)
"""
function stop!(d::DebugTime)
    d.last_time = Nanosecond(0)
    return d
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
function (d::DebugWarnIfCostIncreases)(
    p::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int
)
    (i < 0) && (return nothing)
    if d.status !== :No
        cost = get_cost(p, get_iterate(st))
        if cost > d.old_cost + d.tol
            # Default case in Gradient Descent, include a tipp
            @warn """The cost increased.
            At iteration #$i the cost increased from $(d.old_cost) to $(cost)."""
            if st isa GradientDescentState && st.stepsize isa ConstantStepsize
                @warn """You seem to be running a `gradient_descent` with a `ConstantStepsize`.
                Maybe consider to use `ArmijoLinesearch` (if applicable) or use
                `ConstantStepsize(value)` with a `value` less than $(get_last_stepsize(p,st,i))."""
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
function show(io::IO, di::DebugWarnIfCostIncreases)
    return print(io, "DebugWarnIfCostIncreases(; tol=\"$(di.tol)\")")
end

@doc raw"""
    DebugWarnIfCostNotFinite <: DebugAction

A debug to see when a field (value or array within the AbstractManoptSolverState is or contains values
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
function (d::DebugWarnIfCostNotFinite)(
    p::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int
)
    if d.status !== :No
        cost = get_cost(p, get_iterate(st))
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
show(io::IO, ::DebugWarnIfCostNotFinite) = print(io, "DebugWarnIfCostNotFinite()")
status_summary(::DebugWarnIfCostNotFinite) = ":WarnCost"

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
    DebugWaranIfFieldNotFinite(:Gradient)

Creates a [`DebugAction`] to track whether the gradient does not get `Nan` or `Inf`.
"""
mutable struct DebugWarnIfFieldNotFinite <: DebugAction
    status::Symbol
    field::Symbol
    function DebugWarnIfFieldNotFinite(field::Symbol=:Gradient, warn::Symbol=:Once)
        return new(warn, field)
    end
end
function (d::DebugWarnIfFieldNotFinite)(
    ::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int
)
    if d.status !== :No
        if d.field == :Iterate
            v = get_iterate(st)
            s = "The iterate"
        elseif d.field == :Gradient
            v = get_gradient(st)
            s = "The gradient"
        else
            v = getproperty(st, d.field)
            s = "The field s.$(d.field)"
        end
        if !all(isfinite.(v))
            @warn """$s is or contains values that are not finite.
            At iteration #$i it evaluated to $(v)."""
            if d.status === :Once
                @warn "Further warnings will be supressed, use DebugWaranIfFieldNotFinite(:$(d.field), :Always) to get all warnings."
                d.status = :No
            end
        end
    end
    return nothing
end
function show(io::IO, dw::DebugWarnIfFieldNotFinite)
    return print(io, "DebugWarnIfFieldNotFinite(:$(dw.field))")
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
* `:GradientNorm` creates a [`DebugGradientNorm`](@ref)
* `:Iterate` creates a [`DebugIterate`](@ref)
* `:Iteration` creates a [`DebugIteration`](@ref)
* `:IterativeTime` creates a [`DebugTime`](@ref)`(:Iterative)`
* `:Stepsize` creates a [`DebugStepsize`](@ref)
* `:WarnCost` creates a [`DebugWarnIfCostNotFinite`](@ref)
* `:WarnGradient` creates a [`DebugWarnIfFieldNotFinite`](@ref) for the `::Gradient`.
* `:Time` creates a [`DebugTime`](@ref)
* `:WarningMessages`creates a [`DebugMessages`](@ref)`(:Warning)`
* `:InfoMessages`creates a [`DebugMessages`](@ref)`(:Info)`
* `:ErrorMessages`creates a [`DebugMessages`](@ref)`(:Error)`
* `:Messages`creates a [`DebugMessages`](@ref)`()` (i.e. the same as `:InfoMessages`)

any other symbol creates a `DebugEntry(s)` to print the entry (o.:s) from the options.
"""
function DebugActionFactory(s::Symbol)
    (s == :Cost) && return DebugCost()
    (s == :Change) && return DebugChange()
    (s == :GradientChange) && return DebugGradientChange()
    (s == :GradientNorm) && return DebugGradientNorm()
    (s == :Iterate) && return DebugIterate()
    (s == :Iteration) && return DebugIteration()
    (s == :Stepsize) && return DebugStepsize()
    (s == :WarnCost) && return DebugWarnIfCostNotFinite()
    (s == :WarnGradient) && return DebugWarnIfFieldNotFinite(:Gradient)
    (s == :Time) && return DebugTime()
    (s == :IterativeTime) && return DebugTime(; mode=:Iterative)
    # Messages
    (s == :WarningMessages) && return DebugMessages(:Warning)
    (s == :InfoMessages) && return DebugMessages(:Info)
    (s == :ErrorMessages) && return DebugMessages(:Error)
    (s == :Messages) && return DebugMessages()
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
* `:Time` creates a [`DebugTime`](@ref)
* `:IterativeTime` creates a [`DebugTime`](@ref)`(:Iterative)`

any other symbol creates a `DebugEntry(s)` to print the entry (o.:s) from the options.
"""
function DebugActionFactory(t::Tuple{Symbol,String})
    (t[1] == :Change) && return DebugChange(; format=t[2])
    (t[1] == :GradientChange) && return DebugGradientChange(; format=t[2])
    (t[1] == :Iteration) && return DebugIteration(; format=t[2])
    (t[1] == :Iterate) && return DebugIterate(; format=t[2])
    (t[1] == :GradientNorm) && return DebugGradientNorm(; format=t[2])
    (t[1] == :Cost) && return DebugCost(; format=t[2])
    (t[1] == :Stepsize) && return DebugStepsize(; format=t[2])
    (t[1] == :Time) && return DebugTime(; format=t[2])
    (t[1] == :IterativeTime) && return DebugTime(; mode=:Iterative, format=t[2])
    return DebugEntry(t[1]; format=t[2])
end
