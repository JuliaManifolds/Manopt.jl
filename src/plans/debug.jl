@doc raw"""
    DebugAction

A `DebugAction` is a small functor to print/issue debug output. The usual call is given by
`(p::AbstractManoptProblem, s::AbstractManoptSolverState, i) -> s`, where `i` is
the current iterate.

By convention `i=0` is interpreted as "For Initialization only," only debug
info that prints initialization reacts, `i<0` triggers updates of variables
internally but does not trigger any output.

# Fields (assumed by subtypes to exist)
* `print` method to perform the actual print. Can for example be set to a file export,
or to @info. The default is the `print` function on the default `Base.stdout`.
"""
abstract type DebugAction <: AbstractStateAction end

@doc raw"""
    DebugSolverState <: AbstractManoptSolverState

The debug state appends debug to any state, they act as a decorator pattern.
Internally a dictionary is kept that stores a [`DebugAction`](@ref) for several occasions
using a `Symbol` as reference.

The original options can still be accessed using the [`get_state`](@ref) function.

# Fields

* `options`:         the options that are extended by debug information
* `debugDictionary`: a `Dict{Symbol,DebugAction}` to keep track of Debug for different actions

# Constructors
    DebugSolverState(o,dA)

construct debug decorated options, where `dD` can be
* a [`DebugAction`](@ref), then it is stored within the dictionary at `:Iteration`
* an `Array` of [`DebugAction`](@ref)s.
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
    return DebugSolverState{S}(st, Dict(:Iteration => dD))
end
function DebugSolverState(
    st::S, dD::Array{<:DebugAction,1}
) where {S<:AbstractManoptSolverState}
    return DebugSolverState{S}(st, Dict(:Iteration => DebugGroup(dD)))
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

"""
    set_manopt_parameter!(ams::DebugSolverState, ::Val{:Debug}, args...)

Set certain values specified by `args...` into the elements of the `debugDictionary`
"""
function set_manopt_parameter!(dss::DebugSolverState, ::Val{:Debug}, args...)
    for d in values(dss.debugDictionary)
        set_manopt_parameter!(d, args...)
    end
    return dss
end
# all other pass through
function set_manopt_parameter!(dss::DebugSolverState, v::Val{T}, args...) where {T}
    return set_manopt_parameter!(dss.state, v, args...)
end
# all other pass through
function get_manopt_parameter(dss::DebugSolverState, v::Val{T}, args...) where {T}
    return get_manopt_parameter(dss.state, v, args...)
end

function status_summary(dst::DebugSolverState)
    if length(dst.debugDictionary) > 0
        s = ""
        for (k, v) in dst.debugDictionary
            s = "$s\n    :$k = $(status_summary(v))"
        end
        return "$(dst.state)\n\n## Debug$s"
    else # for length 1 the group is equivalent to the summary of the single state
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
mutable struct DebugGroup{D<:DebugAction} <: DebugAction
    group::Vector{D}
end
function (d::DebugGroup)(p::AbstractManoptProblem, st::AbstractManoptSolverState, i)
    for di in d.group
        di(p, st, i)
    end
end
function status_summary(dg::DebugGroup)
    str = join(["$(status_summary(di))" for di in dg.group], ", ")
    return "[ $str ]"
end
function show(io::IO, dg::DebugGroup)
    s = join(["$(di)" for di in dg.group], ", ")
    return print(io, "DebugGroup([$s])")
end
function set_manopt_parameter!(dg::DebugGroup, v::Val, args...)
    for di in dg.group
        set_manopt_parameter!(di, v, args...)
    end
    return dg
end
function set_manopt_parameter!(dg::DebugGroup, e::Symbol, args...)
    set_manopt_parameter!(dg, Val(e), args...)
    return dg
end

@doc raw"""
    DebugEvery <: DebugAction

evaluate and print debug only every $i$th iteration. Otherwise no print is performed.
Whether internal variables are updates is determined by `always_update`.

This method does not perform any print itself but relies on it's children's print.

It also sets the subsolvers active parameter, see |`DebugWhenActive`}(#ref).
Here, the `activattion_offset` can be used to specify whether it refers to _this_ iteration,
the `i`th, when this call is _before_ the iteration, then the offset should be 0,
for the _next_ iteration, that is if this is called _after_ an iteration, it has to be set to 1.
Since usual debug is happening after the iteration, 1 is the default.

# Constructor

    DebugEvery(d::DebugAction, every=1, always_update=true, activation_offset=1)
"""
mutable struct DebugEvery <: DebugAction
    debug::DebugAction
    every::Int
    always_update::Bool
    activation_offset::Int
    function DebugEvery(
        d::DebugAction, every::Int=1, always_update::Bool=true; activation_offset=1
    )
        return new(d, every, always_update, activation_offset)
    end
end
function (d::DebugEvery)(p::AbstractManoptProblem, st::AbstractManoptSolverState, i)
    if (rem(i, d.every) == 0)
        d.debug(p, st, i)
    elseif d.always_update
        d.debug(p, st, -1)
    end
    # set activity for this iterate in subsolvers
    set_manopt_parameter!(
        st,
        :SubState,
        :Debug,
        :Activity,
        !(i < 1) && (rem(i + d.activation_offset, d.every) == 0),
    )
    return nothing
end
function show(io::IO, de::DebugEvery)
    return print(
        io,
        "DebugEvery($(de.debug), $(de.every), $(de.always_update); activation_offset=$(de.activation_offset))",
    )
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
function set_manopt_parameter!(de::DebugEvery, e::Symbol, args...)
    set_manopt_parameter!(de, Val(e), args...)
    return de
end
function set_manopt_parameter!(de::DebugEvery, args...)
    set_manopt_parameter!(de.debug, args...)
    return de
end

#
# Special single ones
#
@doc raw"""
    DebugChange(M=DefaultManifold())

debug for the amount of change of the iterate (stored in `get_iterate(o)` of the [`AbstractManoptSolverState`](@ref))
during the last iteration. See [`DebugEntryChange`](@ref) for the general case

# Keyword parameters

* `storage`:                   (`StoreStateAction( [:Gradient] )` storage of the previous action
* `prefix`:                    (`"Last Change:"`) prefix of the debug output (ignored if you set `format`)
* `io`:                        (`stdout`) default stream to print the debug to.
* `format`:                    ( `"$prefix %f"`) format to print the output.
* `inverse_retraction_method`: (`default_inverse_retraction_method(M)`) the inverse retraction
  to be used for approximating distance.
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
        "DebugChange(; format=\"$(escape_string(dc.format))\", inverse_retraction=$(dc.inverse_retraction_method))",
    )
end
status_summary(dc::DebugChange) = "(:Change, \"$(escape_string(dc.format))\")"

@doc raw"""
    DebugCost <: DebugAction

print the current cost function value, see [`get_cost`](@ref).

# Constructors
    DebugCost()

# Parameters

* `format`: (`"$prefix %f"`) format to print the output
* `io`:     (`stdout`) default stream to print the debug to.
* `long`:   (`false`) short form to set the format to `f(x):` (default) or `current cost: ` and the cost
"""
mutable struct DebugCost <: DebugAction
    io::IO
    format::String
    function DebugCost(;
        long::Bool=false, io::IO=stdout, format=long ? "current cost: %f" : "f(x): %f"
    )
        return new(io, format)
    end
end
function (d::DebugCost)(p::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int)
    (i >= 0) && Printf.format(d.io, Printf.Format(d.format), get_cost(p, get_iterate(st)))
    return nothing
end
function show(io::IO, di::DebugCost)
    return print(io, "DebugCost(; format=\"$(escape_string(di.format))\")")
end
status_summary(di::DebugCost) = "(:Cost, \"$(escape_string(di.format))\")"

@doc raw"""
    DebugDivider <: DebugAction

print a small divider (default `" | "`).

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
function show(io::IO, di::DebugDivider)
    return print(io, "DebugDivider(; divider=\"$(escape_string(di.divider))\")")
end
status_summary(di::DebugDivider) = "\"$(escape_string(di.divider))\""

@doc raw"""
    DebugEntry <: DebugAction

print a certain fields entry during the iterates, where a `format` can be specified
how to print the entry.

# Additional fields

* `field`: symbol the entry can be accessed with within [`AbstractManoptSolverState`](@ref)

# Constructor

    DebugEntry(f; prefix="$f:", format = "$prefix %s", io=stdout)

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
    return print(io, "DebugEntry(:$(di.field); format=\"$(escape_string(di.format))\")")
end

"""
    DebugFeasibility <: DebugAction

Display information about the feasibility of the current iterate

# Fields
* `atol`:   absolute tolerance for when either equality or inequality constraints are counted as violated
* `format`: a vector of symbols and string formatting the output
* `io`:     default stream to print the debug to.

The following symbols are filled with values

* `:Feasbile` display true or false depending on whether the iterate is feasible
* `:FeasbileEq` display `=` or `≠` equality constraints are fulfilled or not
* `:FeasbileInEq` display `≤` or `≰` inequality constraints are fulfilled or not
* `:NumEq` display the number of equality constraints infeasible
* `:NumEqNz` display the number of equality constraints infeasible if exists
* `:NumIneq` display the number of inequality constraints infeasible
* `:NumIneqNz` display the number of inequality constraints infeasible if exists
* `:TotalEq` display the sum of how much the equality constraints are violated
* `:TotalInEq` display the sum of how much the inequality constraints are violated

format to print the output.

# Constructor

DebugFeasibility(
    format=["feasible: ", :Feasible];
    io::IO=stdout,
    atol=1e-13
)

"""
mutable struct DebugFeasibility <: DebugAction
    atol::Float64
    format::Vector{Union{String,Symbol}}
    io::IO
    function DebugFeasibility(format=["feasible: ", :Feasible]; io::IO=stdout, atol=1e-13)
        return new(atol, format, io)
    end
end
function (d::DebugFeasibility)(
    mp::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int
)
    s = ""
    p = get_iterate(st)
    eqc = get_equality_constraint(mp, p, :)
    eqc_nz = eqc[abs.(eqc) .> d.atol]
    ineqc = get_inequality_constraint(mp, p, :)
    ineqc_pos = ineqc[ineqc .> d.atol]
    feasible = (length(eqc_nz) == 0) && (length(ineqc_pos) == 0)
    n_eq = length(eqc_nz)
    n_ineq = length(ineqc_pos)
    for f in d.format
        (f isa String) && (s *= f)
        (f === :Feasible) && (s *= feasible ? "Yes" : "No")
        (f === :FeasibleEq) && (s *= n_eq == 0 ? "=" : "≠")
        (f === :FeasibleIneq) && (s *= n_ineq == 0 ? "≤" : "≰")
        (f === :NumEq) && (s *= "$(n_eq)")
        (f === :NumEqNz) && (s *= n_eq == 0 ? "" : "$(n_eq)")
        (f === :NumIneq) && (s *= "$(n_ineq)")
        (f === :NumIneqNz) && (s *= n_ineq == 0 ? "" : "$(n_ineq)")
        (f === :TotalEq) && (s *= "$(sum(abs.(eqc_nz);init=0.0))")
        (f === :TotalInEq) && (s *= "$(sum(ineq_pos;init=0.0))")
    end
    print(d.io, (k > 0) ? s : "")
    return nothing
end
function show(io::IO, d::DebugFeasibility)
    sf = "[" * (join([e isa String ? "\"$e\"" : ":$e" for e in d.format], ", ")) * "]"
    return print(io, "DebugFeasibility($sf; atol=$(d.atol))")
end
function status_summary(d::DebugFeasibility)
    sf = "[" * (join([e isa String ? "\"$e\"" : ":$e" for e in d.format], ", ")) * "]"
    return "(:Feasibility, $sf)"
end

@doc raw"""
    DebugIfEntry <: DebugAction

Issue a warning, info, or error if a certain field does _not_ pass a the `check`.

The `message` is printed in this case. If it contains a `@printf` argument identifier,
that one is filled with the value of the `field`.
That way you can print the value in this case as well.

# Fields

* `io`:    an `IO` stream
* `check`: a function that takes the value of the `field` as input and returns a boolean
* `field`: symbol the entry can be accessed with within [`AbstractManoptSolverState`](@ref)
* `msg`:   if the `check` fails, this message is displayed
* `type`: symbol specifying the type of display, possible values `:print`, `: warn`, `:info`, `:error`,
            where `:print` prints to `io`.

# Constructor

    DebugEntry(field, check=(>(0)); type=:warn, message=":$f is nonnegative", io=stdout)

"""
mutable struct DebugIfEntry{F} <: DebugAction
    io::IO
    check::F
    field::Symbol
    msg::String
    type::Symbol
    function DebugIfEntry(
        f::Symbol, check::F=(>(0)); type=:warn, message=":$f nonpositive.", io::IO=stdout
    ) where {F}
        return new{F}(io, check, f, message, type)
    end
end
function (d::DebugIfEntry)(::AbstractManoptProblem, st::AbstractManoptSolverState, i)
    if (i >= 0) && (!d.check(getfield(st, d.field)))
        format = Printf.Format(d.msg)
        msg = !('%' ∈ d.msg) ? d.msg : Printf.format(format, getfield(st, d.field))
        d.type === :warn && (@warn "$(msg)")
        d.type === :info && (@info "$(msg)")
        d.type === :error && error(msg)
        d.type === :print && print(d.io, msg)
    end
    return nothing
end
function show(io::IO, di::DebugIfEntry)
    return print(io, "DebugIfEntry(:$(di.field), $(di.check); type=:$(di.type))")
end

@doc raw"""
    DebugEntryChange{T} <: DebugAction

print a certain entries change during iterates

# Additional fields

* `print`:    (`print`) function to print the result
* `prefix`:   (`"Change of :Iterate"`) prefix to the print out
* `format`:   (`"$prefix %e"`) format to print (uses the `prefix by default and scientific notation)
* `field`:    Symbol the field can be accessed with within [`AbstractManoptSolverState`](@ref)
* `distance`: function (p,o,x1,x2) to compute the change/distance between two values of the entry
* `storage`:  a [`StoreStateAction`](@ref) to store the previous value of `:f`

# Constructors

    DebugEntryChange(f,d)

# Keyword arguments

* `io`:             (`stdout`) an `IOStream`
* `prefix`:         (`"Change of $f"`)
* `storage`:        (`StoreStateAction((f,))`) a [`StoreStateAction`](@ref)
* `initial_value`: an initial value for the change of `o.field`.
* `format`:         (`"$prefix %e"`) format to print the change
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
        io,
        "DebugEntryChange(:$(dec.field), $(dec.distance); format=\"$(escape_string(dec.format))\")",
    )
end

@doc raw"""
    DebugGradientChange()

debug for the amount of change of the gradient (stored in `get_gradient(o)` of the [`AbstractManoptSolverState`](@ref) `o`)
during the last iteration. See [`DebugEntryChange`](@ref) for the general case

# Keyword parameters

* `storage`: (`StoreStateAction( (:Gradient,) )`) storage of the action for previous data
* `prefix`:  (`"Last Change:"`) prefix of the debug output (ignored if you set `format`)
* `io`:      (`stdout`) default stream to print the debug to.
* `format`:  ( `"$prefix %f"`) format to print the output
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
        "DebugGradientChange(; format=\"$(escape_string(dgc.format))\", vector_transport_method=$(dgc.vector_transport_method))",
    )
end
function status_summary(di::DebugGradientChange)
    return "(:GradientChange, \"$(escape_string(di.format))\")"
end

@doc raw"""
    DebugIterate <: DebugAction

debug for the current iterate (stored in `get_iterate(o)`).

# Constructor
    DebugIterate()

# Parameters

* `io`:        (`stdout`) default stream to print the debug to.
* `format`:    (`"$prefix %s"`) format how to print the current iterate
* `long`:      (`false`) whether to have a long (`"current iterate:"`) or a short (`"p:"`) prefix
* `prefix`     (see `long` for default) set a prefix to be printed before the iterate
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
    return print(io, "DebugIterate(; format=\"$(escape_string(di.format))\")")
end
status_summary(di::DebugIterate) = "(:Iterate, \"$(escape_string(di.format))\")"

@doc raw"""
    DebugIteration <: DebugAction

# Constructor

    DebugIteration()

# Keyword parameters

* `format`: (`"# %-6d"`) format to print the output
* `io`:     (`stdout`) default stream to print the debug to.

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
function show(io::IO, di::DebugIteration)
    return print(io, "DebugIteration(; format=\"$(escape_string(di.format))\")")
end
status_summary(di::DebugIteration) = "(:Iteration, \"$(escape_string(di.format))\")"

@doc raw"""
    DebugMessages <: DebugAction

An [`AbstractManoptSolverState`](@ref) or one of its sub steps like a
[`Stepsize`](@ref) might generate warnings throughout their computations.
This debug can be used to `:print` them display them as `:info` or `:warnings` or even `:error`,
depending on the message type.

# Constructor

    DebugMessages(mode=:Info, warn=:Once; io::IO=stdout)

Initialize the messages debug to a certain `mode`. Available modes are

* `:Error`:   issue the messages as an error and hence stop at any issue occurring
* `:Info`:    issue the messages as an `@info`
* `:Print`:   print messages to the steam `io`.
* `:Warning`: issue the messages as a warning

The `warn` level can be set to `:Once` to only display only the first message,
to `:Always` to report every message, one can set it to `:No`,
to deactivate this, then this [`DebugAction`](@ref) is inactive.
All other symbols are handled as if they were `:Always:`
"""
mutable struct DebugMessages <: DebugAction
    io::IO
    mode::Symbol
    status::Symbol
    function DebugMessages(mode::Symbol=:Info, warn::Symbol=:Once; io::IO=stdout)
        return new(io, mode, warn)
    end
end
function (d::DebugMessages)(::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int)
    if d.status !== :No
        msg = get_message(st)
        (i < 0 || length(msg) == 0) && (return nothing)
        (d.mode == :Warning) && (@warn msg)
        (d.mode == :Error) && (@error msg)
        (d.mode == :Print) && (print(d.io, msg))
        (d.mode == :Info) && (@info msg)
        if d.status === :Once
            @warn "Further warnings will be suppressed, use DebugMessages(:$(d.mode), :Always) to get all warnings."
            d.status = :No
        end
    end
    return nothing
end
show(io::IO, d::DebugMessages) = print(io, "DebugMessages(:$(d.mode), :$(d.status))")
function status_summary(d::DebugMessages)
    (d.mode == :Warning) && return "(:WarningMessages, :$(d.status))"
    (d.mode == :Error) && return "(:ErrorMessages, :$(d.status))"
    # default
    # (d.mode == :Info) && return "(:InfoMessages, $(d.status)"
    return "(:Messages, :$(d.status))"
end

@doc raw"""
    DebugStoppingCriterion <: DebugAction

print the Reason provided by the stopping criterion. Usually this should be
empty, unless the algorithm stops.

# Fields

* `prefix`: (`""`) format to print the output
* `io`:     (`stdout`) default stream to print the debug to.

# Constructor

DebugStoppingCriterion(prefix = ""; io::IO=stdout)

"""
mutable struct DebugStoppingCriterion <: DebugAction
    io::IO
    prefix::String
    DebugStoppingCriterion(prefix=""; io::IO=stdout) = new(io, prefix)
end
function (d::DebugStoppingCriterion)(
    ::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int
)
    print(d.io, (i > 0) ? "$(d.prefix)$(get_reason(st))" : "")
    return nothing
end
function show(io::IO, c::DebugStoppingCriterion)
    s = length(c.prefix) > 0 ? "\"$(c.prefix)\"" : ""
    return print(io, "DebugStoppingCriterion($s)")
end
function status_summary(c::DebugStoppingCriterion)
    return length(c.prefix) == 0 ? ":Stop" : "(:Stop, \"$(c.prefix)\")"
end

@doc raw"""
    DebugWhenActive <: DebugAction

evaluate and print debug only if the active boolean is set.
This can be set from outside and is for example triggered by [`DebugEvery`](@ref)
on debugs on the subsolver.

This method does not perform any print itself but relies on it's children's prints.

For now, the main interaction is with [`DebugEvery`](@ref) which might activate or
deactivate this debug

# Fields

* `active`:        a boolean that can (de-)activated from outside to turn on/off debug
* `always_update`: whether or not to call the order debugs with iteration `<=0` inactive state

# Constructor

    DebugWhenActive(d::DebugAction, active=true, always_update=true)
"""
mutable struct DebugWhenActive{D<:DebugAction} <: DebugAction
    debug::D
    active::Bool
    always_update::Bool
    function DebugWhenActive(
        d::D, active::Bool=true, always_update::Bool=true
    ) where {D<:DebugAction}
        return new{D}(d, active, always_update)
    end
end
function (dwa::DebugWhenActive)(p::AbstractManoptProblem, st::AbstractManoptSolverState, i)
    if dwa.active
        dwa.debug(p, st, i)
    elseif (i <= 0) && (dwa.always_update)
        dwa.debug(p, st, i)
    end
end
function show(io::IO, dwa::DebugWhenActive)
    return print(io, "DebugWhenActive($(dwa.debug), $(dwa.active), $(dwa.always_update))")
end
function status_summary(dwa::DebugWhenActive)
    return repr(dwa)
end
function set_manopt_parameter!(dwa::DebugWhenActive, v::Val, args...)
    set_manopt_parameter!(dwa.debug, v, args...)
    return dwa
end
function set_manopt_parameter!(dwa::DebugWhenActive, ::Val{:Activity}, v)
    return dwa.active = v
end

@doc raw"""
    DebugTime()

Measure time and print the intervals. Using `start=true` you can start the timer on construction,
for example to measure the runtime of an algorithm overall (adding)

The measured time is rounded using the given `time_accuracy` and printed after [canonicalization](https://docs.julialang.org/en/v1/stdlib/Dates/#Dates.canonicalize).

# Keyword parameters

* `io`:            (`stdout`) default stream to print the debug to.
* `format`:        ( `"$prefix %s"`) format to print the output, where `%s` is the canonicalized time`.
* `mode`:          (`:cumulative`) whether to display the total time or reset on every call using `:iterative`.
* `prefix`:        (`"Last Change:"`) prefix of the debug output (ignored if you set `format`)
* `start`:         (`false`) indicate whether to start the timer on creation or not. Otherwise it might only be started on first call.
* `time_accuracy`: (`Millisecond(1)`) round the time to this period before printing the canonicalized time
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
    return print(
        io, "DebugTime(; format=\"$(escape_string(di.format))\", mode=:$(di.mode))"
    )
end
function status_summary(di::DebugTime)
    if di.mode === :iterative
        return "(:IterativeTime, \"$(escape_string(di.format))\")"
    end
    return "(:Time, \"$(escape_string(di.format))\")"
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

#
# Debugs that warn about something
#
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
            @warn """
            The cost increased.
            At iteration #$i the cost increased from $(d.old_cost) to $(cost).
            """
            if st isa GradientDescentState && st.stepsize isa ConstantStepsize
                @warn """
                You seem to be running a `gradient_descent` with a `ConstantStepsize`.
                Maybe consider to use `ArmijoLinesearch` (if applicable) or use
                `ConstantStepsize(value)` with a `value` less than $(get_last_stepsize(p,st,i)).
                """
            end
            if d.status === :Once
                @warn "Further warnings will be suppressed, use DebugWarnIfCostIncreases(:Always) to get all warnings."
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
            At iteration #$i the cost evaluated to $(cost).
            """
            if d.status === :Once
                @warn "Further warnings will be suppressed, use DebugWarnIfCostNotFinite(:Always) to get all warnings."
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
            @warn """
            $s is or contains values that are not finite.
            At iteration #$i it evaluated to $(v).
            """
            if d.status === :Once
                @warn "Further warnings will be suppressed, use DebugWaranIfFieldNotFinite(:$(d.field), :Always) to get all warnings."
                d.status = :No
            end
        end
    end
    return nothing
end
function show(io::IO, dw::DebugWarnIfFieldNotFinite)
    return print(io, "DebugWarnIfFieldNotFinite(:$(dw.field), :$(dw.status))")
end

@doc raw"""
    DebugWarnIfGradientNormTooLarge{T} <: DebugAction

A debug to warn when an evaluated gradient at the current iterate is larger than
(a factor times) the maximal (recommended) stepsize at the current iterate.

# Constructor

    DebugWarnIfGradientNormTooLarge(factor::T=1.0, warn=:Once)

Initialize the warning to warn `:Once`.

This can be set to `:Once` to only warn the first time the cost is Nan.
It can also be set to `:No` to deactivate the warning, but this makes this Action also useless.
All other symbols are handled as if they were `:Always:`

# Example
    DebugWaranIfFieldNotFinite(:Gradient)

Creates a [`DebugAction`] to track whether the gradient does not get `Nan` or `Inf`.
"""
mutable struct DebugWarnIfGradientNormTooLarge{T} <: DebugAction
    status::Symbol
    factor::T
    function DebugWarnIfGradientNormTooLarge(factor::T=1.0, warn::Symbol=:Once) where {T}
        return new{T}(warn, factor)
    end
end
function (d::DebugWarnIfGradientNormTooLarge)(
    mp::AbstractManoptProblem, st::AbstractManoptSolverState, i::Int
)
    if d.status !== :No
        M = get_manifold(mp)
        p = get_iterate(st)
        X = get_gradient(st)
        Xn = norm(M, p, X)
        p_inj = d.factor * max_stepsize(M, p)
        if Xn > p_inj
            @warn """At iteration #$i
            the gradient norm ($Xn) is larger that $(d.factor) times the injectivity radius $(p_inj) at the current iterate.
            """
            if d.status === :Once
                @warn "Further warnings will be suppressed, use DebugWarnIfGradientNormTooLarge($(d.factor), :Always) to get all warnings."
                d.status = :No
            end
        end
    end
    return nothing
end
function show(io::IO, d::DebugWarnIfGradientNormTooLarge)
    return print(io, "DebugWarnIfGradientNormTooLarge($(d.factor), :$(d.status))")
end

#
# Convenience constructors using Symbols
#
@doc raw"""
    DebugFactory(a::Vector)

Generate a dictionary of [`DebugAction`](@ref)s.

First all `Symbol`s `String`, [`DebugAction`](@ref)s and numbers are collected,
excluding `:Stop` and `:WhenActive`.
This collected vector is added to the `:Iteration => [...]` pair.
`:Stop` is added as `:StoppingCriterion` to the `:Stop => [...]` pair.
If necessary, these pairs are created

For each `Pair` of a `Symbol` and a `Vector`, the [`DebugGroupFactory`](@ref)
is called for the `Vector` and the result is added to the debug dictionary's entry
with said symbol. This is wrapped into the [`DebugWhenActive`](@ref),
when the `:WhenActive` symbol is present

# Return value

A dictionary for the different enrty points where debug can happen, each containing
a [`DebugAction`](@ref) to call.

Note that upon the initialisation all dictionaries but the `:StartAlgorithm`
one are called with an `i=0` for reset.

# Examples

1. Providing a simple vector of symbols, numbers and strings like

    [:Iterate, " | ", :Cost, :Stop, 10]

Adds a group to :Iteration of three actions ([`DebugIteration`](@ref), [`DebugDivider`](@ref)`(" | "),  and[`DebugCost`](@ref))
as a [`DebugGroup`](@ref) inside an [`DebugEvery`](@ref) to only be executed every 10th iteration.
It also adds the [`DebugStoppingCriterion`](@ref) to the `:EndAlgorhtm` entry of the dictionary.

2. The same can also be written a bit more precise as

    DebugFactory([:Iteration => [:Iterate, " | ", :Cost, 10], :Stop])

3. We can even make the stoping criterion concrete and pass Actions directly,
  for example explicitly Making the stop more concrete, we get

    DebugFactory([:Iteration => [:Iterate, " | ", DebugCost(), 10], :Stop => [:Stop]])
"""
function DebugFactory(a::Vector{<:Any})
    entries = filter(x -> !isa(x, Pair) && (x ∉ [:Stop, :WhenActive]) && !isa(x, Int), a)
    # Filter pairs
    b = filter(x -> isa(x, Pair), a)
    # Push this to the `:Iteration` if that exists or add that pair
    i = findlast(x -> (isa(x, Pair)) && (x.first == :Iteration), b)
    if !isnothing(i)
        item = popat!(b, i) #
        b = [b..., :Iteration => [item.second..., entries...]]
    else
        (length(entries) > 0) && (b = [b..., :Iteration => entries])
    end
    # Push a StoppingCriterion to `:Stop` if that exists or add such a pair
    if (:Stop in a)
        i = findlast(x -> (isa(x, Pair)) && (x.first == :Stop), b)
        if !isnothing(i)
            stop = popat!(b, i) #
            b = [b..., :Stop => [stop.second..., DebugActionFactory(:Stop)]]
        else # regenerate since the type of b might change
            b = [b..., :Stop => [DebugActionFactory(:Stop)]]
        end
    end
    dictionary = Dict{Symbol,DebugAction}()
    # Look for a global number -> DebugEvery
    e = filter(x -> isa(x, Int), a)
    ae = length(e) > 0 ? last(e) : 0
    # Run through all (updated) pairs
    for d in b
        offset = d.first === :BeforeIteration ? 0 : 1
        debug = DebugGroupFactory(d.second; activation_offset=offset)
        (:WhenActive in a) && (debug = DebugWhenActive(debug))
        # Add DebugEvery to all but Start and Stop
        (!(d.first in [:Start, :Stop]) && (ae > 0)) && (debug = DebugEvery(debug, ae))
        dictionary[d.first] = debug
    end
    return dictionary
end

@doc raw"""
   DebugGroupFactory(a::Vector)

Generate a [`DebugGroup`] of [`DebugAction`](@ref)s. The following rules are used

1. Any `Symbol` is passed to [`DebugActionFactory`](@ref DebugActionFactory(::Symbol))
2. Any `(Symbol, String)` generates similar actions as in 1., but the string is used for `format=``,
  see [`DebugActionFactory`](@ref DebugActionFactory(::Tuple{Symbol,String}))
3. Any `String` is passed to `DebugActionFactory(d::String)`](@ref)`
4. Any [`DebugAction`](@ref) is included as is.

If this results in more than one [`DebugAction`](@ref) a [`DebugGroup`](@ref) of these is build.

If any integers are present, the last of these is used to wrap the group in a
[`DebugEvery`](@ref)`(k)`.

If `:WhenActive` is present, the resulting Action is wrapped in [`DebugWhenActive`](@ref),
making it deactivatable by its parent solver.
"""
function DebugGroupFactory(a::Vector; activation_offset=1)
    group = DebugAction[]
    for d in filter(x -> !isa(x, Int) && (x ∉ [:WhenActive]), a) # filter Integers & Active
        push!(group, DebugActionFactory(d))
    end
    l = length(group)
    (l == 0) && return DebugDivider("")
    if l == 1
        debug = first(group)
    else
        debug = DebugGroup(group)
    end
    # filter numbers, find last
    e = filter(x -> isa(x, Int), a)
    if length(e) > 0
        debug = DebugEvery(debug, last(e); activation_offset=activation_offset)
    end
    (:WhenActive in a) && (debug = (DebugWhenActive(debug)))
    return debug
end
DebugGroupFactory(a; kwargs...) = DebugGroupFactory([a]; kwargs...)

@doc raw"""
    DebugActionFactory(s)

create a [`DebugAction`](@ref) where

* a `String`yields the corresponding divider
* a [`DebugAction`](@ref) is passed through
* a [`Symbol`] creates [`DebugEntry`](@ref) of that symbol, with the exceptions
  of `:Change`, `:Iterate`, `:Iteration`, and `:Cost`.
* a `Tuple{Symbol,String}` creates a [`DebugEntry`](@ref) of that symbol where the String specifies the format.
"""
DebugActionFactory(d::String) = DebugDivider(d)
DebugActionFactory(a::A) where {A<:DebugAction} = a

"""
    DebugActionFactory(s::Symbol)

Convert certain Symbols in the `debug=[ ... ]` vector to [`DebugAction`](@ref)s
Currently the following ones are done.
Note that the Shortcut symbols should all start with a capital letter.

* `:Cost` creates a [`DebugCost`](@ref)
* `:Change` creates a [`DebugChange`](@ref)
* `:Gradient` creates a [`DebugGradient`](@ref)
* `:GradientChange` creates a [`DebugGradientChange`](@ref)
* `:GradientNorm` creates a [`DebugGradientNorm`](@ref)
* `:Iterate` creates a [`DebugIterate`](@ref)
* `:Iteration` creates a [`DebugIteration`](@ref)
* `:IterativeTime` creates a [`DebugTime`](@ref)`(:Iterative)`
* `:Stepsize` creates a [`DebugStepsize`](@ref)
* `:Stop` creates a [`StoppingCriterion`](@ref)`()`
* `:WarnCost` creates a [`DebugWarnIfCostNotFinite`](@ref)
* `:WarnGradient` creates a [`DebugWarnIfFieldNotFinite`](@ref) for the `::Gradient`.
* `:WarnBundle` creates a [`DebugWarnIfLagrangeMultiplierIncreases`](@ref)
* `:Time` creates a [`DebugTime`](@ref)
* `:WarningMessages` creates a [`DebugMessages`](@ref)`(:Warning)`
* `:InfoMessages` creates a [`DebugMessages`](@ref)`(:Info)`
* `:ErrorMessages` creates a [`DebugMessages`](@ref)`(:Error)`
* `:Messages` creates a [`DebugMessages`](@ref)`()` (the same as `:InfoMessages`)

any other symbol creates a `DebugEntry(s)` to print the entry (o.:s) from the options.
"""
function DebugActionFactory(d::Symbol)
    (d == :Cost) && return DebugCost()
    (d == :Change) && return DebugChange()
    (d == :Gradient) && return DebugGradient()
    (d == :GradientChange) && return DebugGradientChange()
    (d == :GradientNorm) && return DebugGradientNorm()
    (d == :Iterate) && return DebugIterate()
    (d == :Iteration) && return DebugIteration()
    (d == :Feasibility) && return DebugFeasibility()
    (d == :Stepsize) && return DebugStepsize()
    (d == :Stop) && return DebugStoppingCriterion()
    (d == :WarnBundle) && return DebugWarnIfLagrangeMultiplierIncreases()
    (d == :WarnCost) && return DebugWarnIfCostNotFinite()
    (d == :WarnGradient) && return DebugWarnIfFieldNotFinite(:Gradient)
    (d == :Time) && return DebugTime()
    (d == :IterativeTime) && return DebugTime(; mode=:Iterative)
    # Messages
    (d == :WarningMessages) && return DebugMessages(:Warning)
    (d == :InfoMessages) && return DebugMessages(:Info)
    (d == :ErrorMessages) && return DebugMessages(:Error)
    (d == :Messages) && return DebugMessages()
    # all other symbols try to display the entry of said symbol
    return DebugEntry(d)
end
"""
    DebugActionFactory(t::Tuple{Symbol,String)

Convert certain Symbols in the `debug=[ ... ]` vector to [`DebugAction`](@ref)s
Currently the following ones are done, where the string in `t[2]` is passed as the
`format` the corresponding debug.
Note that the Shortcut symbols `t[1]` should all start with a capital letter.

* `:Change` creates a [`DebugChange`](@ref)
* `:Cost` creates a [`DebugCost`](@ref)
* `:Gradient` creates a [`DebugGradient`](@ref)
* `:GradientChange` creates a [`DebugGradientChange`](@ref)
* `:GradientNorm` creates a [`DebugGradientNorm`](@ref)
* `:Iterate` creates a [`DebugIterate`](@ref)
* `:Iteration` creates a [`DebugIteration`](@ref)
* `:Stepsize` creates a [`DebugStepsize`](@ref)
* `:Stop` creates a [`DebugStoppingCriterion`](@ref)
* `:Time` creates a [`DebugTime`](@ref)
* `:IterativeTime` creates a [`DebugTime`](@ref)`(:Iterative)`

any other symbol creates a `DebugEntry(s)` to print the entry (o.:s) from the options.
"""
function DebugActionFactory(t::Tuple{Symbol,Any})
    (t[1] == :Change) && return DebugChange(; format=t[2])
    (t[1] == :Cost) && return DebugCost(; format=t[2])
    (t[1] == :Feasibility) && return DebugFeasibility(t[2])
    (t[1] == :Gradient) && return DebugGradient(; format=t[2])
    (t[1] == :GradientChange) && return DebugGradientChange(; format=t[2])
    (t[1] == :GradientNorm) && return DebugGradientNorm(; format=t[2])
    (t[1] == :Iteration) && return DebugIteration(; format=t[2])
    (t[1] == :Iterate) && return DebugIterate(; format=t[2])
    (t[1] == :IterativeTime) && return DebugTime(; mode=:Iterative, format=t[2])
    (t[1] == :Stepsize) && return DebugStepsize(; format=t[2])
    (t[1] == :Stop) && return DebugStoppingCriterion(t[2])
    (t[1] == :Time) && return DebugTime(; format=t[2])
    ((t[1] == :Messages) || (t[1] == :InfoMessages)) && return DebugMessages(:Info, t[2])
    (t[1] == :WarningMessages) && return DebugMessages(:Warning, t[2])
    (t[1] == :ErrorMessages) && return DebugMessages(:error, t[2])
    return DebugEntry(t[1]; format=t[2])
end
