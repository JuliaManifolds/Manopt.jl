@doc """
    DebugCallback <: DebugAction

Debug for a simple callback function, mainly for compatibility to other solvers and if
a user already has a callback function or functor available

The expected format of the is that it is a function with signature `(problem, state, iteration) -> nothing`
A simple callback of the signature `() -> nothing` can be specified by `simple=true`. In this case the callback is wrapped in a function of the generic form

!!! note
    This is for now an internal struct, since its name might still change before
    it is made public. The functionality with the factory (`callback=f`) will still work,
    but this debug actions name might still change its name in the future.

# Constructor

    DebugCallback(callback; simple=false)
"""
struct DebugCallback{CB} <: DebugAction
    callback::CB
    function DebugCallback(callback; simple::Bool = false)
        _cb = simple ? (problem, state, k) -> callback() : callback
        return new{typeof(_cb)}(_cb)
    end
end
function (d::DebugCallback)(
        problem::AbstractManoptProblem, state::AbstractManoptSolverState, k
    )
    d.callback(problem, state, k)
    return nothing
end
function show(io::IO, dc::DebugCallback{CB}) where {CB}
    return print(io, "DebugCallback($(dc.callback))")
end
function status_summary(dc::DebugCallback; context::Symbol = :default)
    (context === :short) && return "$(dc.callback)"
    # inline and default
    return "A DebugAction with a callback that calls $(dc.callback)"
end

@doc """
    DebugChange(M=DefaultManifold(); kwargs...)

debug for the amount of change of the iterate (stored in `get_iterate(o)` of the [`AbstractManoptSolverState`](@ref))
during the last iteration. See [`DebugEntryChange`](@ref) for the general case

# Keyword parameters

* `storage=`[`StoreStateAction`](@ref)`( [:Gradient] )` storage of the previous action
* `prefix="Last Change:"`: prefix of the debug output (ignored if you set `format`)
* `io=stdout`: default stream to print the debug to.
$(_kwargs(:inverse_retraction_method))

the inverse retraction
  to be used for approximating distance.
"""
mutable struct DebugChange{IR <: AbstractInverseRetractionMethod} <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    inverse_retraction_method::IR
    function DebugChange(
            M::AbstractManifold = DefaultManifold();
            storage::Union{Nothing, StoreStateAction} = nothing,
            io::IO = stdout, prefix::String = "Last Change: ", format::String = "$(prefix)%f",
            inverse_retraction_method::AbstractInverseRetractionMethod = default_inverse_retraction_method(M),
        )
        irm = inverse_retraction_method
        if isnothing(storage)
            if M isa DefaultManifold
                storage = StoreStateAction(M; store_fields = [:Iterate])
            else
                storage = StoreStateAction(M; store_points = Tuple{:Iterate})
            end
        end
        return new{typeof(irm)}(io, format, storage, irm)
    end
end
function (d::DebugChange)(mp::AbstractManoptProblem, st::AbstractManoptSolverState, k)
    M = get_manifold(mp)
    (k > 0) && Printf.format(
        d.io,
        Printf.Format(d.format),
        distance(
            M, get_iterate(st), get_storage(d.storage, PointStorageKey(:Iterate)),
            d.inverse_retraction_method,
        ),
    )
    d.storage(mp, st, k)
    return nothing
end
function show(io::IO, dc::DebugChange)
    return print(
        io,
        "DebugChange(; format=\"$(escape_string(dc.format))\", inverse_retraction=$(dc.inverse_retraction_method))",
    )
end
function status_summary(dc::DebugChange; context::Symbol = :default)
    (context === :short) && (return "(:Change, \"$(escape_string(dc.format))\")")
    # Inline and Default
    return "A DebugAction to print the change of the iterate from one iteration to the next with format “$(escape_string(dc.format))”"
end
@doc """
    DebugCost <: DebugAction

print the current cost function value, see [`get_cost`](@ref).

# Constructors
    DebugCost()

# Parameters

* `format="\$prefix %f"`: format to print the output
* `io=stdout`: default stream to print the debug to.
* `long=false`: short form to set the format to `f(x):` (default) or `current cost: ` and the cost
* `at_init=true`: whether to print also at initialization
"""
mutable struct DebugCost <: DebugAction
    io::IO
    format::String
    at_init::Bool
    function DebugCost(;
            long::Bool = false, io::IO = stdout, format = long ? "current cost: %f" : "f(x): %f",
            at_init::Bool = true,
        )
        return new(io, format, at_init)
    end
end
function (d::DebugCost)(p::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int)
    (k >= (d.at_init ? 0 : 1)) && Printf.format(d.io, Printf.Format(d.format), get_cost(p, get_iterate(st)))
    return nothing
end
function show(io::IO, di::DebugCost)
    return print(io, "DebugCost(; format=\"$(escape_string(di.format))\", at_init=$(di.at_init))")
end
function status_summary(di::DebugCost; context::Symbol = :default)
    (context === :short) && return "(:Cost, \"$(escape_string(di.format))\")"
    # inline & default
    return "A DebugAction printing the current cost value"
end

@doc """
    DebugDivider <: DebugAction

print a small divider (default `" | "`).

# Constructor
    DebugDivider(div, io=stdout, at_init=true)

"""
mutable struct DebugDivider{TypeIO <: IO} <: DebugAction
    io::TypeIO
    divider::String
    at_init::Bool
    DebugDivider(divider = " | "; io::IO = stdout, at_init::Bool = true) = new{typeof(io)}(io, divider, at_init)
end
function (d::DebugDivider)(::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int)
    if k >= (d.at_init ? 0 : 1) && !isempty(d.divider)
        print(d.io, d.divider)
    end
    return nothing
end
function show(io::IO, di::DebugDivider)
    return print(io, "DebugDivider(; divider=\"$(escape_string(di.divider))\", at_init=$(di.at_init))")
end
function status_summary(di::DebugDivider; context::Symbol = :default)
    (context === :short) && (return "\"$(escape_string(di.divider))\"")
    # inline and default
    return "A DebugAction printing the String “$(escape_string(di.divider))” as a divider"
end
# A global constant for empty debugs
_EMPTY_DIVIDER = DebugDivider("")

"""
    DebugDualChange(opts...)

Print the change of the dual variable.

This is similar to [`DebugChange`](@ref) (see their constructors for details), but uses a
different calculation of the change, since the dual variable lives in (possibly different)
tangent spaces.
"""
mutable struct DebugDualChange <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    at_init::Bool
    function DebugDualChange(;
            storage::StoreStateAction = StoreStateAction([:X, :n]),
            io::IO = stdout, prefix = "Dual Change: ", format = "$prefix%s", at_init::Bool = false,
        )
        return new(io, format, storage)
    end
    function DebugDualChange(
            values::Tuple{T, P};
            storage::StoreStateAction = StoreStateAction([:X, :n]),
            io::IO = stdout, prefix = "Dual Change: ", format = "$prefix%s",
        ) where {P, T}
        update_storage!(
            storage, Dict{Symbol, Any}(k => v for (k, v) in zip((:X, :n), values))
        )
        return new(io, format, storage)
    end
end
function (d::DebugDualChange)(
        tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, k::Int
    )
    N = get_manifold(tmp, 2)
    if all(has_storage.(Ref(d.storage), [:X, :n])) && k > 0 # all values stored
        #fetch
        X_old = get_storage(d.storage, :X)
        n_old = get_storage(d.storage, :n)
        v = norm(
            N, apds.n,
            vector_transport_to(
                N, n_old, X_old, apds.n, apds.vector_transport_method_dual
            ) - apds.X,
        )
        (k >= (d.at_init ? 0 : 1)) && Printf.format(d.io, Printf.Format(d.format), v)
    end
    return d.storage(tmp, apds, k)
end
function show(io::IO, ddc::DebugDualChange)
    return print(io, "DebugDualChange(; io = ", ddc.io, ", format =\"$(escape_string(ddc.format))\", at_init=$(ddc.at_init))")
end
function status_summary(ddc::DebugDualChange; context::Symbol = :default)
    (context === :short) && return repr(ddc)
    return "A DebugAction to print the change of the dual variable with format \"$(escape_string(ddc.format))\""
end

@doc """
    DebugDualResidual <: DebugAction

A Debug action to print the dual residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:Iterate`, `:X` and `:n`.

# Constructor
DebugDualResidual(; kwargs...)

# Keyword warguments

* `io=stdout`: stream to perform the debug to
* `format="\$prefix%s"`: format to print the dual residual, using the
* `prefix="Dual Residual: "`: short form to just set the prefix
* `storage` (a new [`StoreStateAction`](@ref)) to store values for the debug.
"""
mutable struct DebugDualResidual <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    at_init::Bool
    function DebugDualResidual(;
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "Dual Residual: ", format = "$prefix%s", at_init::Bool = false,
        )
        return new(io, format, storage, at_init)
    end
    function DebugDualResidual(
            initial_values::Tuple{P, T, Q};
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "Dual Residual: ", format = "$prefix%s", at_init::Bool = false,
        ) where {P, T, Q}
        update_storage!(
            storage, Dict(k => v for (k, v) in zip((:Iterate, :X, :n), initial_values))
        )
        return new(io, format, storage, at_init)
    end
end
function (d::DebugDualResidual)(
        tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, k::Int
    )
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    apdmo = get_objective(tmp)
    if all(has_storage.(Ref(d.storage), [:Iterate, :X, :n])) && (k >= (d.at_init ? 0 : 1)) # all values stored
        #fetch
        p_old = get_storage(d.storage, :Iterate)
        X_old = get_storage(d.storage, :X)
        n_old = get_storage(d.storage, :n)
        Printf.format(
            d.io, Printf.Format(d.format),
            dual_residual(M, N, apdmo, apds, p_old, X_old, n_old),
        )
    end
    return d.storage(tmp, apds, k)
end
function show(io::IO, d::DebugDualResidual)
    return print(io, "DebugDualResidual(; io = ", d.io, ", format=\"$(escape_string(d.format))\", at_init=$(d.at_init))")
end
function status_summary(d::DebugDualResidual; context::Symbol = :default)
    (context === :short) && return repr(d)
    return "A DebugAction to print the dual residual with format \"$(escape_string(d.format))\""
end

@doc """
    DebugEntry <: DebugAction

print a certain fields entry during the iterates, where a `format` can be specified
how to print the entry.

# Additional fields

* `field`: symbol the entry can be accessed with within [`AbstractManoptSolverState`](@ref)
* `at_init`: whether to print also at initialization

# Constructor

    DebugEntry(f; prefix="\$f:", format = "\$prefix %s", io=stdout, at_init=true)
"""
mutable struct DebugEntry <: DebugAction
    io::IO
    format::String
    field::Symbol
    at_init::Bool
    function DebugEntry(f::Symbol; prefix = "$f:", format = "$prefix %s", io::IO = stdout, at_init::Bool = true)
        return new(io, format, f, at_init)
    end
end
function (d::DebugEntry)(::AbstractManoptProblem, st::AbstractManoptSolverState, k)
    (k >= (d.at_init ? 0 : 1)) && Printf.format(d.io, Printf.Format(d.format), getfield(st, d.field))
    return nothing
end
function show(io::IO, di::DebugEntry)
    return print(io, "DebugEntry(:$(di.field); format=\"$(escape_string(di.format))\", at_init=$(di.at_init))")
end
function status_summary(di::DebugEntry; context::Symbol = :default)
    (context === :short) && return "(:$(di.field), format=\"$(escape_string(di.format))\")"
    return "A DebugAction to print the field :$(di.field) of the solver state with format \"$(escape_string(di.format))\""
end

"""
    DebugDualIterate(e)

Print the dual variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set display `o.X`.
"""
DebugDualIterate(opts...; kwargs...) = DebugEntry(:X, opts...; kwargs...)

"""
    DebugDualBaseIterate(io::IO=stdout)

Print the dual base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set to display the field `n` of the state.
"""
DebugDualBaseIterate(; kwargs...) = DebugEntry(:n; kwargs...)


@doc """
    DebugEntryChange{T} <: DebugAction

print a certain entries change during iterates

# Additional fields

* `print`:    function to print the result
* `prefix`:   prefix to the print out
* `format`:   format to print (uses the `prefix` by default and scientific notation)
* `field`:    Symbol the field can be accessed with within [`AbstractManoptSolverState`](@ref)
* `distance`: function (p,o,x1,x2) to compute the change/distance between two values of the entry
* `storage`:  a [`StoreStateAction`](@ref) to store the previous value of `:f`

# Constructors

    DebugEntryChange(f,d)

## Keyword arguments

* `io=stdout`:                      an `IOStream` used for the debug
* `prefix="Change of \$f"`:          the prefix
* `storage=StoreStateAction((f,))`: a [`StoreStateAction`](@ref)
* `initial_value=NaN`:              an initial value for the change of `o.field`.
* `format="\$prefix %e"`:            format to print the change
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
            storage::StoreStateAction = StoreStateAction([f]),
            prefix::String = "Change of \$f:",
            format::String = "$prefix%s",
            io::IO = stdout,
            initial_value::Any = NaN,
        )
        if !isa(initial_value, Number) || !isnan(initial_value) #set initial value
            update_storage!(storage, Dict(f => initial_value))
        end
        return new(d, f, format, io, storage)
    end
end
function (d::DebugEntryChange)(
        p::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int
    )
    if k == 0
        # on init if field not present -> generate
        !has_storage(d.storage, d.field) && d.storage(p, st, k)
        return nothing
    end
    x = get_storage(d.storage, d.field)
    v = d.distance(p, st, getproperty(st, d.field), x)
    (k > 0) && Printf.format(d.io, Printf.Format(d.format), v)
    d.storage(p, st, k)
    return nothing
end
function show(io::IO, dec::DebugEntryChange)
    return print(
        io,
        "DebugEntryChange(:$(dec.field), $(dec.distance); format=\"$(escape_string(dec.format))\")",
    )
end
function status_summary(d::DebugEntryChange; context::Symbol = :default)
    (context === :short) && return repr(d)
    return "A DebugAction that prints the change of the entry :$(d.field) of the solver state in format “$(escape_string(d.format))”"
end

"""
    DebugDualBaseChange(; storage=StoreStateAction([:n]), kwargs...)

Print the change of the dual base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on the field `n` of the state.
"""
function DebugDualBaseChange(;
        storage::StoreStateAction = StoreStateAction([:n]), prefix = "Dual Base Change:", kwargs...
    )
    return DebugEntryChange(
        :n, (p, o, x, y) -> distance(get_manifold(p, 2), x, y, o.inverse_retraction_method_dual);
        storage = storage, prefix = prefix, kwargs...,
    )
end

"""
    DebugPrimalBaseIterate()

Print the primal base variable by using [`DebugEntry`](@ref),
see their constructors for detail.
This method is further set to display the field `m` of the state.
"""
DebugPrimalBaseIterate(opts...; kwargs...) = DebugEntry(:m, opts...; kwargs...)

"""
    DebugPrimalBaseChange(opts...; prefix="Primal Base Change:", kwargs...)

Print the change of the primal base variable by using [`DebugEntryChange`](@ref),
see their constructors for detail, on the field `m` of the state.
"""
function DebugPrimalBaseChange(opts...; prefix = "Primal Base Change:", kwargs...)
    return DebugEntryChange(
        :m, (p, o, x, y) -> distance(get_manifold(p, 1), x, y),
        opts...; prefix = prefix, kwargs...,
    )
end

"""
    DebugFeasibility <: DebugAction

Display information about the feasibility of the current iterate

# Fields
* `format`: a vector of symbols and string formatting the output
* `io`:     default stream to print the debug to.
* `at_init`: whether to print also at initialization

The following symbols are filled with values

* `:Feasible` display true or false depending on whether the iterate is feasible
* `:FeasibleEq` display `=` or `≠` equality constraints are fulfilled or not
* `:FeasibleIneq` display `≤` or `≰` inequality constraints are fulfilled or not
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
    at_init::Bool=true,
)

"""
mutable struct DebugFeasibility <: DebugAction
    format::Vector{Union{String, Symbol}}
    io::IO
    at_init::Bool
    function DebugFeasibility(format = ["feasible: ", :Feasible]; io::IO = stdout, atol = NaN, at_init::Bool = true)
        isnan(atol) || (@warn "Providing atol= directly to DebugFeasibility is deprecated. Use the keyword for the ConstrainedObjective instead. The value provided here ($(atol)) is ignored")
        return new(format, io, at_init)
    end
end
function (d::DebugFeasibility)(
        mp::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int
    )
    s = ""
    cmo = get_objective(mp, true) #Unwrap to get the constrained objective.
    p = get_iterate(st)
    eqc = get_equality_constraint(mp, p, :)
    eqc_nz = eqc[abs.(eqc) .> cmo.atol]
    ineqc = get_inequality_constraint(mp, p, :)
    ineqc_pos = ineqc[ineqc .> cmo.atol]
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
        (f === :TotalEq) && (s *= "$(sum(abs.(eqc_nz); init = 0.0))")
        (f === :TotalInEq) && (s *= "$(sum(ineqc_pos; init = 0.0))")
    end
    print(d.io, (k >= (d.at_init ? 0 : 1)) ? s : "")
    return nothing
end
function show(io::IO, d::DebugFeasibility)
    sf = "[" * (join([e isa String ? "\"$e\"" : ":$e" for e in d.format], ", ")) * "]"
    return print(io, "DebugFeasibility($sf, at_init=$(d.at_init))")
end
function status_summary(d::DebugFeasibility; context::Symbol = :default)
    sf = "[" * (join([e isa String ? "\"$e\"" : ":$e" for e in d.format], ", ")) * "]"
    (context === :short) && (return "(:Feasibility, $sf)")
    # inline and Default
    return "A DebugAction printing Feasibility information of the current iterate, namely $sf"
end

@doc """
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
* `at_init`: whether to print also at initialization

# Constructor

    DebugIfEntry(field, check=(>(0)); type=:warn, message=":\$f is nonnegative", io=stdout, at_init=true)

"""
mutable struct DebugIfEntry{F} <: DebugAction
    io::IO
    check::F
    field::Symbol
    msg::String
    type::Symbol
    at_init::Bool
    function DebugIfEntry(
            f::Symbol, check::F = (>(0)); type = :warn, message = ":\$f nonpositive.", io::IO = stdout, at_init::Bool = true
        ) where {F}
        return new{F}(io, check, f, message, type, at_init)
    end
end
function (d::DebugIfEntry)(::AbstractManoptProblem, st::AbstractManoptSolverState, k)
    if (k >= (d.at_init ? 0 : 1)) && (!d.check(getfield(st, d.field)))
        format = Printf.Format(d.msg)
        msg = !('%' ∈ d.msg) ? d.msg : Printf.format(format, getfield(st, d.field))
        d.type === :warn && (@warn "$(msg)")
        d.type === :info && (@info "$(msg)")
        d.type === :error && error(msg)
        d.type === :print && print(d.io, msg)
    end
    return nothing
end
function show(io::IO, d::DebugIfEntry)
    return print(io, "DebugIfEntry(:$(d.field), $(d.check); type=:$(d.type), at_init=$(d.at_init))")
end
function status_summary(d::DebugIfEntry; context::Symbol = :Default)
    (context === :short) && (return repr(d))
    # Inline and default
    return "A DebugAction printing the entry :$(d.field) of the solver state if $(d.check) of that field is true, in format “$(escape_string(d.msg))” as $(d.type)"
end

@doc """
    DebugGradient <: DebugAction

debug for the gradient evaluated at the current iterate

# Constructors
    DebugGradient(; long=false, prefix= , format= "\$prefix%s", io=stdout, at_init=false)

display the short (`false`) or long (`true`) default text for the gradient,
or set the `prefix` manually. Alternatively the complete format can be set.
"""
mutable struct DebugGradient <: DebugAction
    io::IO
    format::String
    at_init::Bool
    function DebugGradient(;
            long::Bool = false,
            prefix = long ? "Gradient: " : "grad f(p):",
            format = "$prefix%s",
            io::IO = stdout,
            at_init::Bool = false,
        )
        return new(io, format, at_init)
    end
end
function (d::DebugGradient)(::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int)
    (k < (d.at_init ? 0 : 1)) && return nothing
    Printf.format(d.io, Printf.Format(d.format), get_gradient(s))
    return nothing
end
function Base.show(io::IO, dg::DebugGradient)
    return print(io, "DebugGradient(; format=\"$(dg.format)\", at_init=$(dg.at_init))")
end
function status_summary(dg::DebugGradient; context::Symbol = :default)
    (context === :short) && (return "(:Gradient, \"$(dg.format)\")")
    return "A DebugAction to print the gradient at the current iterate “$(dg.format)”"
end

@doc """
    DebugGradientChange()

debug for the amount of change of the gradient (stored in `get_gradient(o)` of the [`AbstractManoptSolverState`](@ref) `o`)
during the last iteration. See [`DebugEntryChange`](@ref) for the general case

# Keyword parameters

* `storage=`[`StoreStateAction`](@ref)`( (:Gradient,) )`: storage of the action for previous data
* `prefix="Last Change:"`: prefix of the debug output (ignored if you set `format`:
* `io=stdout`: default stream to print the debug to.
* `format="\$prefix %f"`: format to print the output
"""
mutable struct DebugGradientChange{VTR <: AbstractVectorTransportMethod} <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    vector_transport_method::VTR
    function DebugGradientChange(
            M::AbstractManifold = DefaultManifold();
            storage::Union{Nothing, StoreStateAction} = nothing,
            io::IO = stdout,
            prefix::String = "Last Change: ",
            format::String = "$(prefix)%f",
            vector_transport_method::VTR = default_vector_transport_method(M),
        ) where {VTR <: AbstractVectorTransportMethod}
        if isnothing(storage)
            if M isa DefaultManifold
                storage = StoreStateAction(M; store_fields = [:Iterate, :Gradient])
            else
                storage = StoreStateAction(
                    M; store_points = [:Iterate], store_vectors = [:Gradient]
                )
            end
        end
        return new{VTR}(io, format, storage, vector_transport_method)
    end
end
function (d::DebugGradientChange)(
        pm::AbstractManoptProblem, st::AbstractManoptSolverState, k
    )
    if k > 0
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
    d.storage(pm, st, k)
    return nothing
end
function show(io::IO, dgc::DebugGradientChange)
    return print(
        io,
        "DebugGradientChange(; format=\"$(escape_string(dgc.format))\", vector_transport_method=$(dgc.vector_transport_method))",
    )
end
function status_summary(di::DebugGradientChange; context::Symbol = :Default)
    (context === :short) && (return "(:GradientChange, \"$(escape_string(di.format))\")")
    # Inline and default
    return "A DebugAction printing the change of the gradient with format “$(escape_string(di.format))”"
end

@doc """
    DebugGradientNorm <: DebugAction

debug for gradient evaluated at the current iterate.

# Constructors
    DebugGradientNorm([long=false, format= "\$prefix%s", io=stdout, at_init=true])

display the short (`false`) or long (`true`) default text for the gradient norm.

    DebugGradientNorm(prefix[, p=print])

display the a `prefix` in front of the gradient norm.
"""
mutable struct DebugGradientNorm <: DebugAction
    io::IO
    format::String
    at_init::Bool
    function DebugGradientNorm(;
            long::Bool = false,
            prefix = long ? "Norm of the Gradient: " : "|grad f(p)|:",
            format = "$prefix%s", io::IO = stdout, at_init::Bool = true,
        )
        return new(io, format, at_init)
    end
end
function (d::DebugGradientNorm)(
        mp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    (k < (d.at_init ? 0 : 1)) && return nothing
    Printf.format(
        d.io,
        Printf.Format(d.format),
        norm(get_manifold(mp), get_iterate(s), get_gradient(s)),
    )
    return nothing
end
function Base.show(io::IO, dgn::DebugGradientNorm)
    return print(io, "DebugGradientNorm(; format=\"$(dgn.format)\", at_init=$(dgn.at_init))")
end
function status_summary(dgn::DebugGradientNorm; context::Symbol = :default)
    (context === :short) && return "(:GradientNorm, \"$(dgn.format)\")"
    return "A debug action to display the gradient norm (format. \"$(dgn.format)\")"
end

@doc """
    DebugIterate <: DebugAction

debug for the current iterate (stored in `get_iterate(o)`).

# Constructor
    DebugIterate(; kwargs...)

# Keyword arguments

* `io=stdout`:           default stream to print the debug to.
* `format="\$prefix %s"`: format how to print the current iterate
* `long=false`:          whether to have a long (`"current iterate:"`) or a short (`"p:"`) prefix default
* `prefix`:              (see `long` for default) set a prefix to be printed before the iterate
* `at_init=true`:        whether to print also at initialization
"""
mutable struct DebugIterate <: DebugAction
    io::IO
    format::String
    at_init::Bool
    function DebugIterate(;
            io::IO = stdout,
            long::Bool = false,
            prefix = long ? "current iterate:" : "p:",
            format = "$prefix %s",
            at_init::Bool = false,
        )
        return new(io, format, at_init)
    end
end
function (d::DebugIterate)(::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int)
    (k >= (d.at_init ? 0 : 1)) && Printf.format(d.io, Printf.Format(d.format), get_iterate(st))
    return nothing
end
function show(io::IO, di::DebugIterate)
    return print(io, "DebugIterate(; format=\"$(escape_string(di.format))\", at_init=$(di.at_init))")
end
function status_summary(di::DebugIterate; context::Symbol = :default)
    (context === :short) && (return "(:Iterate, \"$(escape_string(di.format))\")")
    # Inline and default
    return "A DebugAction printing the current iterate in format “$(escape_string(di.format))”"
end

@doc """
    DebugIteration <: DebugAction

# Constructor

    DebugIteration()

# Keyword parameters

* `format="# %-6d"`: format to print the output
* `io=stdout`: default stream to print the debug to.

debug for the current iteration (prefixed with `#` by )
"""
mutable struct DebugIteration <: DebugAction
    io::IO
    format::String
    DebugIteration(; io::IO = stdout, format = "# %-6d") = new(io, format)
end
function (d::DebugIteration)(::AbstractManoptProblem, ::AbstractManoptSolverState, k::Int)
    (k == 0) && print(d.io, "Initial ")
    (k > 0) && Printf.format(d.io, Printf.Format(d.format), k)
    return nothing
end
function show(io::IO, di::DebugIteration)
    return print(io, "DebugIteration(; format=\"$(escape_string(di.format))\")")
end
function status_summary(di::DebugIteration; context::Symbol = :default)
    (context === :short) && return "(:Iteration, \"$(escape_string(di.format))\")"
    # Inline and default
    return "A DebugAction that prints the current iteration number in format “$(escape_string(di.format))”"
end
@doc """
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
    function DebugMessages(mode::Symbol = :Info, warn::Symbol = :Once; io::IO = stdout)
        return new(io, mode, warn)
    end
end
function (d::DebugMessages)(::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int)
    if d.status !== :No
        msg = get_message(st)
        (k < 0 || length(msg) == 0) && (return nothing)
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
function status_summary(d::DebugMessages; context::Symbol = :default)
    if context === :short
        s = ":Messages"
        (d.mode == :Warning) && (s = ":WarningMessages")
        (d.mode == :Error) && (s = ":ErrorMessages")
        (d.mode == :Info) && (s = ":InfoMessages")
        return d.status === :No ? s : "($s, :$(d.status))"
    end
    # Inline and default
    m = "a $(d.mode == :Warning ? "warning " : (d.mode == :Error ? "error " : ""))message"
    s = d.status === :No ? " (inactive)" : (d.status === :Once ? " once" : "")
    return "A DebugAction printing messages collected during the last iteration as $(m)$(s)."
end

"""
    DebugPrimalChange(opts...)

Print the change of the primal variable by using [`DebugChange`](@ref),
see their constructors for detail.
"""
function DebugPrimalChange(;
        storage::StoreStateAction = StoreStateAction([:Iterate]), prefix = "Primal Change: ", kwargs...,
    )
    return DebugChange(; storage = storage, prefix = prefix, kwargs...)
end


@doc """
    DebugPrimalDualResidual <: DebugAction

A Debug action to print the primal dual residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:Iterate`, `:X` and `:n`.

# Constructor

    DebugPrimalDualResidual()

with the keywords

# Keyword warguments

* `io=stdout`: stream to perform the debug to
* `format="\$prefix%s"`: format to print the dual residual, using the
* `prefix="PD Residual: "`: short form to just set the prefix
* `storage` (a new [`StoreStateAction`](@ref)) to store values for the debug.
"""
mutable struct DebugPrimalDualResidual <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    at_init::Bool
    function DebugPrimalDualResidual(;
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "PD Residual: ", format = "$prefix%s", at_init::Bool = false,
        )
        return new(io, format, storage, at_init)
    end
    function DebugPrimalDualResidual(
            values::Tuple{P, T, Q};
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "PD Residual: ", format = "$prefix%s", at_init::Bool = false,
        ) where {P, Q, T}
        update_storage!(storage, Dict(k => v for (k, v) in zip((:Iterate, :X, :n), values)))
        return new(io, format, storage, at_init)
    end
end
function (d::DebugPrimalDualResidual)(
        tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, k::Int
    )
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    apdmo = get_objective(tmp)
    if all(has_storage.(Ref(d.storage), [:Iterate, :X, :n])) && (k >= (d.at_init ? 0 : 1)) # all values stored
        #fetch
        p_old = get_storage(d.storage, :Iterate)
        X_old = get_storage(d.storage, :X)
        n_old = get_storage(d.storage, :n)
        v = primal_residual(M, N, apdmo, apds, p_old, X_old, n_old) + dual_residual(tmp, apds, p_old, X_old, n_old)
        Printf.format(d.io, Printf.Format(d.format), v / manifold_dimension(M))
    end
    return d.storage(tmp, apds, k)
end
function show(io::IO, d::DebugPrimalDualResidual)
    return print(io, "DebugPrimalDualResidual(; io = ", d.io, ", format=\"$(escape_string(d.format))\", at_init=$(d.at_init))")
end
function status_summary(d::DebugPrimalDualResidual; context::Symbol = :default)
    (context === :short) && return repr(d)
    return "A DebugAction to print the primal dual residual with format \"$(escape_string(d.format))\""
end

"""
    DebugPrimalIterate(opts...;kwargs...)

Print the change of the primal variable by using [`DebugIterate`](@ref),
see their constructors for detail.
"""
DebugPrimalIterate(opts...; kwargs...) = DebugIterate(opts...; kwargs...)


@doc """
    DebugPrimalResidual <: DebugAction

A Debug action to print the primal residual.
The constructor accepts a printing function and some (shared) storage, which
should at least record `:Iterate`, `:X` and `:n`.

# Constructor

    DebugPrimalResidual(; kwargs...)

# Keyword warguments

* `io=stdout`: stream to perform the debug to
* `format="\$prefix%s"`: format to print the dual residual, using the
* `prefix="Primal Residual: "`: short form to just set the prefix
* `storage` (a new [`StoreStateAction`](@ref)) to store values for the debug.
"""
mutable struct DebugPrimalResidual <: DebugAction
    io::IO
    format::String
    storage::StoreStateAction
    at_init::Bool
    function DebugPrimalResidual(;
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "Primal Residual: ", format = "$prefix%s", at_init::Bool = false
        )
        return new(io, format, storage, at_init)
    end
    function DebugPrimalResidual(
            values::Tuple{P, T, Q};
            storage::StoreStateAction = StoreStateAction([:Iterate, :X, :n]),
            io::IO = stdout, prefix = "Primal Residual: ", format = "$prefix%s", at_init::Bool = false,
        ) where {P, T, Q}
        update_storage!(storage, Dict(k => v for (k, v) in zip((:Iterate, :X, :n), values)))
        return new(io, format, storage, at_init)
    end
end
function (d::DebugPrimalResidual)(
        tmp::TwoManifoldProblem, apds::AbstractPrimalDualSolverState, k::Int
    )
    M = get_manifold(tmp, 1)
    N = get_manifold(tmp, 2)
    apdmo = get_objective(tmp)
    if all(has_storage.(Ref(d.storage), [:Iterate, :X, :n])) && (k >= (d.at_init ? 0 : 1)) # all values stored
        #fetch
        p_old = get_storage(d.storage, :Iterate)
        X_old = get_storage(d.storage, :X)
        n_old = get_storage(d.storage, :n)
        Printf.format(
            d.io, Printf.Format(d.format),
            primal_residual(M, N, apdmo, apds, p_old, X_old, n_old),
        )
    end
    return d.storage(tmp, apds, k)
end
function show(io::IO, d::DebugPrimalResidual)
    return print(io, "DebugPrimalResidual(; io = ", d.io, ", format=\"$(escape_string(d.format))\", at_init=$(d.at_init))")
end
function status_summary(d::DebugPrimalResidual; context::Symbol = :default)
    (context === :short) && return repr(d)
    return "A DebugAction to print the primal residual with format \"$(escape_string(d.format))\""
end

@doc """
    DebugProximalParameter <: DebugAction

print the current iterates proximal point algorithm parameter given by
[`AbstractManoptSolverState`](@ref)s `o.λ`.
"""
mutable struct DebugProximalParameter <: DebugAction
    io::IO
    format::String
    at_init::Bool
    function DebugProximalParameter(;
            long::Bool = false,
            prefix = long ? "Proximal Map Parameter λ(i):" : "λ:",
            format = "$prefix%s",
            io::IO = stdout,
            at_init::Bool = true,
        )
        return new(io, format, at_init)
    end
end
function Base.show(io::IO, d::DebugProximalParameter)
    return print(
        io, "DebugGradientChange(; io = ", d.io, ", format=\"$(escape_string(d.format))\", at_init = $(d.at_init))",
    )
end
function status_summary(d::DebugProximalParameter; context::Symbol = :Default)
    (context === :short) && (return "(:ProxParameter, \"$(escape_string(d.format))\")")
    # Inline and default
    return "A DebugAction printing the proximal parameter as “$(escape_string(d.format))”"
end


@doc """
    DebugStepsize <: DebugAction

debug for the current step size.

# Constructors
    DebugStepsize(;long=false,prefix="step size:", format="\$prefix%s", io=stdout, at_init=true)

display the a `prefix` in front of the step size.
"""
mutable struct DebugStepsize <: DebugAction
    io::IO
    format::String
    at_init::Bool
    function DebugStepsize(;
            at_init::Bool = true, io::IO = stdout,
            long::Bool = false, prefix = long ? "step size:" : "s:", format = "$prefix%s",
        )
        return new(io, format, at_init)
    end
end
function (d::DebugStepsize)(
        p::P, s::O, k::Int
    ) where {P <: AbstractManoptProblem, O <: AbstractGradientSolverState}
    (k < (d.at_init ? 0 : 1)) && return nothing
    Printf.format(d.io, Printf.Format(d.format), get_last_stepsize(p, s, k))
    return nothing
end
function Base.show(io::IO, ds::DebugStepsize)
    return print(io, "DebugStepsize(; format=\"$(escape_string(ds.format))\", at_init=$(ds.at_init))")
end
function status_summary(ds::DebugStepsize; context::Symbol = :default)
    (context === :short) && return "(:Stepsize, \"$(escape_string(ds.format))\")"
    return "A DebugAction that prints the current step size to $(ds.io) in format “$(escape_string(ds.format))”"
end

@doc """
    DebugStoppingCriterion <: DebugAction

print the Reason provided by the stopping criterion. Usually this should be
empty, unless the algorithm stops.

# Fields

* `prefix=""`: format to print the output
* `io=stdout`: default stream to print the debug to.

# Constructor

DebugStoppingCriterion(prefix = ""; io::IO=stdout)

"""
mutable struct DebugStoppingCriterion <: DebugAction
    io::IO
    prefix::String
    DebugStoppingCriterion(prefix = ""; io::IO = stdout) = new(io, prefix)
end
function (d::DebugStoppingCriterion)(
        ::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int
    )
    print(d.io, (k > 0) ? "$(d.prefix)$(get_reason(st))" : "")
    return nothing
end
function show(io::IO, c::DebugStoppingCriterion)
    s = length(c.prefix) > 0 ? "\"$(c.prefix)\"" : ""
    return print(io, "DebugStoppingCriterion($s)")
end
function status_summary(c::DebugStoppingCriterion; context::Symbol = :default)
    (context === :short) && (return length(c.prefix) == 0 ? ":Stop" : "(:Stop, \"$(c.prefix)\")")
    # Inline and default
    return "A DebugAction printing the reason why a solver has stopped."
end

@doc """
    DebugWarnIfLagrangeMultiplierIncreases <: DebugAction

print a warning if the Lagrange parameter based value ``-ξ`` of the bundle method increases.

# Constructor

    DebugWarnIfLagrangeMultiplierIncreases(warn=:Once; tol=1e2)

Initialize the warning to warning level (`:Once`) and introduce a tolerance for the test of `1e2`.

The `warn` level can be set to `:Once` to only warn the first time the cost increases,
to `:Always` to report an increase every time it happens, and it can be set to `:No`
to deactivate the warning, then this [`DebugAction`](@ref) is inactive.
All other symbols are handled as if they were `:Always`.
"""
mutable struct DebugWarnIfLagrangeMultiplierIncreases <: DebugAction
    status::Symbol
    old_value::Float64
    tol::Float64
    function DebugWarnIfLagrangeMultiplierIncreases(warn::Symbol = :Once; tol = 1.0e2)
        return new(warn, Float64(Inf), tol)
    end
end
function show(io::IO, d::DebugWarnIfLagrangeMultiplierIncreases)
    m = (d.status === :No ? "" : ":$(d.status)")
    return print(io, "DebugWarnIfLagrangeMultiplierIncreases($(m); tol=\"$(d.tol)\")")
end
function status_summary(d::DebugWarnIfLagrangeMultiplierIncreases; context::Symbol = :default)
    (context === :short) && return repr(d)
    m = (d.status === :Once) ? "once" : (d.status === :No ? "(inactive)" : "")
    return "a DebugAction warning if the lagange multiplier increases in an iteration $m."
end


@doc """
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
mutable struct DebugWhenActive{D <: DebugAction} <: DebugAction
    debug::D
    active::Bool
    always_update::Bool
    function DebugWhenActive(
            d::D, active::Bool = true, always_update::Bool = true
        ) where {D <: DebugAction}
        return new{D}(d, active, always_update)
    end
end
function (dwa::DebugWhenActive)(p::AbstractManoptProblem, st::AbstractManoptSolverState, k)
    return if dwa.active
        dwa.debug(p, st, k)
    elseif (k < 0) && (dwa.always_update)
        dwa.debug(p, st, k)
    end
end
function show(io::IO, dwa::DebugWhenActive)
    return print(io, "DebugWhenActive($(dwa.debug), $(dwa.active), $(dwa.always_update))")
end
function status_summary(dwa::DebugWhenActive; context::Symbol = :default)
    (context === :short) && (return repr(dwa))
    (context === :inline) && return "A DebugAction only printing its internal criterion ($(status_summary(dwa.debug; context = context))) when active (currently: $(dwa.active))"
    return """
    a DebugActin only printing its internal DebugAction when activated

    ## DebugAction
    $(status_summary(dwa.debug; context = context))$(dwa.always_update ? "\nwhich is always updated for negative iteration numbers still." : "")

    ## Current activity
    $(dwa.active ? "active" : "inactive") – use `set_parameter!(debug_action, :Activity, $(!dwa.active))` to toggle
    """
end
function set_parameter!(dwa::DebugWhenActive, v::Val, args...)
    set_parameter!(dwa.debug, v, args...)
    return dwa
end
function set_parameter!(dwa::DebugWhenActive, ::Val{:Activity}, v)
    return dwa.active = v
end

@doc """
    DebugTime()

Measure time and print the intervals. Using `start=true` you can start the timer on construction,
for example to measure the runtime of an algorithm overall (adding)

The measured time is rounded using the given `time_accuracy` and printed after [canonicalization](https://docs.julialang.org/en/v1/stdlib/Dates/#Dates.canonicalize).

# Keyword parameters

* `io=stdout`:             default stream to print the debug to.
* `format="\$prefix %s"`:   format to print the output, where `%s` is the canonicalized time.
* `mode=:cumulative`:      whether to display the total time or reset on every call using `:iterative`.
* `prefix="Last Change:"`: prefix of the debug output (ignored if you set `format`:
* `start=false`:           indicate whether to start the timer on creation or not.
   Otherwise it might only be started on first call.
* `time_accuracy=Millisecond(1)`: round the time to this period before printing the canonicalized time
"""
mutable struct DebugTime <: DebugAction
    io::IO
    format::String
    last_time::Nanosecond
    time_accuracy::Period
    mode::Symbol
    function DebugTime(;
            start = false,
            io::IO = stdout,
            prefix::String = "time spent:",
            format::String = "$(prefix) %s",
            mode::Symbol = :cumulative,
            time_accuracy::Period = Millisecond(1),
        )
        return new(io, format, Nanosecond(start ? time_ns() : 0), time_accuracy, mode)
    end
end
function (d::DebugTime)(::AbstractManoptProblem, ::AbstractManoptSolverState, k)
    if k == 0 || d.last_time == Nanosecond(0) # init
        d.last_time = Nanosecond(time_ns())
    elseif k > 0
        t = time_ns()
        p = Nanosecond(t) - d.last_time
        Printf.format(
            d.io, Printf.Format(d.format), canonicalize(round(p, d.time_accuracy))
        )
        if d.mode == :iterative
            d.last_time = Nanosecond(time_ns())
        end
    end
    return nothing
end
function show(io::IO, di::DebugTime)
    return print(
        io, "DebugTime(; format=\"$(escape_string(di.format))\", mode=:$(di.mode))"
    )
end
function status_summary(di::DebugTime; context::Symbol = :default)
    if context == :short
        if di.mode === :iterative
            return "(:IterativeTime, \"$(escape_string(di.format))\")"
        end
        return "(:Time, \"$(escape_string(di.format))\")"
    end
    # Default and inline
    return "a DebugActin to print time per step $(di.mode === :iterative ? "iteratively" : "cumulatively")"
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
@doc """
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
    DebugWarnIfCostIncreases(warn::Symbol = :Once; tol = 1.0e-13) = new(warn, Float64(Inf), tol)
end
function (d::DebugWarnIfCostIncreases)(
        p::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int
    )
    if d.status !== :No
        cost = get_cost(p, get_iterate(st))
        if cost > d.old_cost + d.tol
            @warn """
            The cost increased.
            At iteration #$k the cost increased from $(d.old_cost) to $(cost).
            """
            if st isa GradientDescentState && st.stepsize isa ConstantStepsize
                @warn """
                You seem to be running a `gradient_descent` with a `ConstantStepsize`.
                Maybe consider to use `ArmijoLinesearch` (if applicable) or use
                `ConstantLength(value)` with a `value` less than $(get_last_stepsize(p, st, k)).
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
function show(io::IO, d::DebugWarnIfCostIncreases)
    m = (d.status === :No ? "" : ":$(d.status)")
    return print(io, "DebugWarnIfCostIncreases($(m); tol=\"$(d.tol)\")")
end
function status_summary(d::DebugWarnIfCostIncreases; context::Symbol = :default)
    (context === :short) && return repr(d)
    m = (d.status === :Once) ? "once" : (d.status === :No ? "(inactive)" : "")
    return "A DebugAction warning if the cost increases in an iteration $m."
end

@doc """
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
    DebugWarnIfCostNotFinite(warn::Symbol = :Once) = new(warn)
end
function (d::DebugWarnIfCostNotFinite)(
        p::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int
    )
    if d.status !== :No
        cost = get_cost(p, get_iterate(st))
        if !isfinite(cost)
            @warn """The cost is not finite.
            At iteration #$k the cost evaluated to $(cost).
            """
            if d.status === :Once
                @warn "Further warnings will be suppressed, use DebugWarnIfCostNotFinite(:Always) to get all warnings."
                d.status = :No
            end
        end
    end
    return nothing
end
show(io::IO, d::DebugWarnIfCostNotFinite) = print(io, "DebugWarnIfCostNotFinite(:$(d.status))")
function status_summary(d::DebugWarnIfCostNotFinite; context::Symbol = :default)
    (context == :short) && (return ":WarnCost")
    # Default and inline
    s = ""
    (d.status === :Once) && (s = " It will only warn once.")
    (d.status === :No) && (s = " It either has warned already or was deactivated by setting its status to `:No`.")
    return "A DebugAction to issue a warning when the cost is no longer finite.$s"
end

@doc """
    DebugWarnIfFieldNotFinite <: DebugAction

A debug to see when a field from the options is not finite, for example `Inf` or `Nan`

# Constructor
    DebugWarnIfFieldNotFinite(field::Symbol, warn=:Once)

Initialize the warning to warn `:Once`.

This can be set to `:Once` to only warn the first time the cost is Nan.
It can also be set to `:No` to deactivate the warning, but this makes this Action also useless.
All other symbols are handled as if they were `:Always:`

# Example
    DebugWarnIfFieldNotFinite(:Gradient)

Creates a [`DebugAction`] to track whether the gradient does not get `Nan` or `Inf`.
"""
mutable struct DebugWarnIfFieldNotFinite <: DebugAction
    status::Symbol
    field::Symbol
    function DebugWarnIfFieldNotFinite(field::Symbol = :Gradient, warn::Symbol = :Once)
        return new(warn, field)
    end
end
function (d::DebugWarnIfFieldNotFinite)(
        ::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int
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
            At iteration #$k it evaluated to $(v).
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
function status_summary(dw::DebugWarnIfFieldNotFinite; context::Symbol = :default)
    (context == :short) && (return repr(dw))
    # Default and inline
    s = ""
    (dw.status === :Once) && (s = " It will only warn once.")
    (dw.status === :No) && (s = " It either has warned already or was deactivated by setting its status to `:No`.")
    return "A DebugAction to warn if the field “:$(dw.field)” is or has entries that are not finite.$s"
end
@doc """
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
    DebugWarnIfFieldNotFinite(:Gradient)

Creates a [`DebugAction`] to track whether the gradient does not get `Nan` or `Inf`.
"""
mutable struct DebugWarnIfGradientNormTooLarge{T} <: DebugAction
    status::Symbol
    factor::T
    function DebugWarnIfGradientNormTooLarge(factor::T = 1.0, warn::Symbol = :Once) where {T}
        return new{T}(warn, factor)
    end
end
function (d::DebugWarnIfGradientNormTooLarge)(
        mp::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int
    )
    if d.status !== :No
        M = get_manifold(mp)
        p = get_iterate(st)
        X = get_gradient(st)
        Xn = norm(M, p, X)
        p_inj = d.factor * max_stepsize(M, p)
        if Xn > p_inj
            @warn """At iteration #$k
            the gradient norm ($Xn) is larger than $(d.factor) times the injectivity radius $(p_inj) at the current iterate.
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
    # only print status if active
    m = (d.status === :No ? "" : ", :$(d.status)")
    return print(io, "DebugWarnIfGradientNormTooLarge($(d.factor)$(m))")
end
function status_summary(d::DebugWarnIfGradientNormTooLarge; context::Symbol = :default)
    (context === :short) && return repr(d)
    m = (d.status === :Once) ? " once" : (d.status === :No ? " (inactive)" : "")
    return "A DebugAction warning if the gradient norm gets larger than the maximal stepsize$m."
end

@doc """
    DebugWarnIfStepsizeCollapsed <: DebugAction

print a warning if the backtracking stopped because the stepsize fell below a given threshold.
This threshold is specified by the `stop_when_stepsize_less` field.

# Constructor

    DebugWarnIfStepsizeCollapsed(tol::T=1e-8,warn=:Once;)

Initialize the warning to warning level (`:Once`) with a tolerance for `stop_when_stepsize_less` set to `tol` (1e-8).

The `warn` level can be set to `:Once` to only warn the first time the cost increases,
to `:Always` to report an increase every time it happens, and it can be set to `:No`
to deactivate the warning, then this [`DebugAction`](@ref) is inactive.
All other symbols are handled as if they were `:Always`
"""
mutable struct DebugWarnIfStepsizeCollapsed{T} <: DebugAction
    status::Symbol
    stop_when_stepsize_less::T
    function DebugWarnIfStepsizeCollapsed(tol::T = 1.0e-8, warn::Symbol = :Once) where {T}
        return new{T}(warn, tol)
    end
end
function (d::DebugWarnIfStepsizeCollapsed)(
        amp::AbstractManoptProblem, st::AbstractManoptSolverState, k::Int
    )
    (k == 0) && (return nothing)
    if d.status !== :No
        if get_last_stepsize(amp, st, k) ≤ d.stop_when_stepsize_less
            @warn "Backtracking stopped because the stepsize fell below the threshold $(d.stop_when_stepsize_less)."
            if d.status === :Once
                @warn "Further warnings will be suppressed, use DebugWarnIfLagrangeMultiplierIncreases(:Always) to get all warnings."
                d.status = :No
            end
        end
    end
    return nothing
end
function show(io::IO, d::DebugWarnIfStepsizeCollapsed)
    m = (d.status === :No ? "" : ", :$(d.status)")
    return print(io, "DebugWarnIfStepsizeCollapsed($(d.stop_when_stepsize_less)$(m))")
end
function status_summary(d::DebugWarnIfStepsizeCollapsed; context::Symbol = :default)
    (context === :short) && return repr(d)
    m = (d.status === :Once) ? " once" : (d.status === :No ? " (inactive)" : "")
    return "A DebugAction warning if the step size collapses (below $(d.stop_when_stepsize_less))$m."
end
#
# Convenience constructors using Symbols
#
@doc """
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

A dictionary for the different entry points where debug can happen, each containing
a [`DebugAction`](@ref) to call.

Note that upon the initialization all dictionaries but the `:StartAlgorithm`
one are called with an `i=0` for reset.

# Examples

1. Providing a simple vector of symbols, numbers and strings like

   ```
   [:Iterate, " | ", :Cost, :Stop, 10]
   ```

   Adds a group to :Iteration of three actions ([`DebugIteration`](@ref), [`DebugDivider`](@ref)`(" | ")`, and [`DebugCost`](@ref))
   as a [`DebugGroup`](@ref) inside an [`DebugEvery`](@ref) to only be executed every 10th iteration.
   It also adds the [`DebugStoppingCriterion`](@ref) to the `:EndAlgorithm` entry of the dictionary.

2. The same can also be written a bit more precise as

   ```
   DebugFactory([:Iteration => [:Iterate, " | ", :Cost, 10], :Stop])
   ```

3. We can even make the stopping criterion concrete and pass Actions directly,
   for example explicitly Making the stop more concrete, we get

   ```
   DebugFactory([:Iteration => [:Iterate, " | ", DebugCost(), 10], :Stop => [:Stop]])
   ```
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
    dictionary = Dict{Symbol, DebugAction}()
    # Look for a global number -> DebugEvery
    e = filter(x -> isa(x, Int), a)
    ae = length(e) > 0 ? last(e) : 0
    # Run through all (updated) pairs
    for d in b
        offset = d.first === :BeforeIteration ? 0 : 1
        debug = DebugGroupFactory(d.second; activation_offset = offset)
        (:WhenActive in a) && (debug = DebugWhenActive(debug))
        # Add DebugEvery to all but Start and Stop
        (!(d.first in [:Start, :Stop]) && (ae > 0)) && (debug = DebugEvery(debug, ae))
        dictionary[d.first] = debug
    end
    return dictionary
end

@doc """
    DebugGroupFactory(a::Vector)

Generate a [`DebugGroup`](@ref) of [`DebugAction`](@ref)s. The following rules are used

1. Any `Symbol` is passed to [`DebugActionFactory`](@ref DebugActionFactory(::Symbol))
2. Any `(Symbol, String)` generates similar actions as in 1., but the string is used for `format=`,
   see [`DebugActionFactory`](@ref DebugActionFactory(::Tuple{Symbol,String}))
3. Any `String` is passed to [`DebugActionFactory`](@ref DebugActionFactory(d::String))
4. Any `Function` generates a [`DebugCallback`](@ref).
5. Any [`DebugAction`](@ref) is included as is.

If this results in more than one [`DebugAction`](@ref) a [`DebugGroup`](@ref) of these is build.

If any integers are present, the last of these is used to wrap the group in a
[`DebugEvery`](@ref)`(k)`.

If `:WhenActive` is present, the resulting Action is wrapped in [`DebugWhenActive`](@ref),
making it deactivatable by its parent solver.
"""
function DebugGroupFactory(a::Vector; activation_offset = 1)
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
        debug = DebugEvery(debug, last(e); activation_offset = activation_offset)
    end
    (:WhenActive in a) && (debug = (DebugWhenActive(debug)))
    return debug
end
DebugGroupFactory(a; kwargs...) = DebugGroupFactory([a]; kwargs...)

@doc """
    DebugActionFactory(s)

create a [`DebugAction`](@ref) where

* a `String`yields the corresponding divider
* a [`DebugAction`](@ref) is passed through
* a [`Symbol`] creates [`DebugEntry`](@ref) of that symbol, with the exceptions
  of `:Change`, `:Iterate`, `:Iteration`, and `:Cost`.
* a `Tuple{Symbol,String}` creates a [`DebugEntry`](@ref) of that symbol where the String specifies the format.
* a `<:Function` creates a [`DebugCallback`](@ref) with the function as callback.
"""
DebugActionFactory(d::String) = DebugDivider(d)
DebugActionFactory(a::A) where {A <: DebugAction} = a
# Deprecated
function DebugActionFactory(f::F) where {F <: Function}
    @warn """
        the `DebugCallback` struct is deprecated. Passing functions to `debug = `
        will no longer word in the next release. Use
        `callbacks = [:Step => [...]]` to add your callback to the (end of)
        an iteration step
    """
    return DebugCallback(f)
end
"""
    DebugActionFactory(s::Symbol)

Convert certain Symbols in the `debug=[ ... ]` vector to [`DebugAction`](@ref)s
Currently the following ones are done.
Note that the Shortcut symbols should all start with a capital letter.

* `:Cost` creates a [`DebugCost`](@ref)
* `:Change` creates a [`DebugChange`](@ref)
* `:Feasibility` creates a [`DebugFeasibility`](@ref)
* `:Gradient` creates a [`DebugGradient`](@ref)
* `:GradientChange` creates a [`DebugGradientChange`](@ref)
* `:GradientNorm` creates a [`DebugGradientNorm`](@ref)
* `:Iterate` creates a [`DebugIterate`](@ref)
* `:Iteration` creates a [`DebugIteration`](@ref)
* `:IterativeTime` creates a [`DebugTime`](@ref)`(:Iterative)`
* `:ProxParameter` creates a [`DebugProximalParameter`](@ref)`()`
* `:Stepsize` creates a [`DebugStepsize`](@ref)
* `:Stop` creates a [`StoppingCriterion`](@ref)`()`
* `:Time` creates a [`DebugTime`](@ref)
* `:WarnStepsize` creates a [`DebugWarnIfStepsizeCollapsed`](@ref)
* `:WarnBundle` creates a [`DebugWarnIfLagrangeMultiplierIncreases`](@ref)
* `:WarnCost` creates a [`DebugWarnIfCostNotFinite`](@ref)
* `:WarnGradient` creates a [`DebugWarnIfFieldNotFinite`](@ref) for the `::Gradient`.
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
    (d == :ProxParameter) && return DebugProximalParameter()
    (d == :Stepsize) && return DebugStepsize()
    (d == :Stop) && return DebugStoppingCriterion()
    (d == :WarnStepsize) && return DebugWarnIfStepsizeCollapsed()
    (d == :WarnBundle) && return DebugWarnIfLagrangeMultiplierIncreases()
    (d == :WarnCost) && return DebugWarnIfCostNotFinite()
    (d == :WarnGradient) && return DebugWarnIfFieldNotFinite(:Gradient)
    (d == :Time) && return DebugTime()
    (d == :IterativeTime) && return DebugTime(; mode = :Iterative)
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
* `:Feasibility` creates a [`DebugFeasibility`](@ref)
* `:Gradient` creates a [`DebugGradient`](@ref)
* `:GradientChange` creates a [`DebugGradientChange`](@ref)
* `:GradientNorm` creates a [`DebugGradientNorm`](@ref)
* `:Iterate` creates a [`DebugIterate`](@ref)
* `:Iteration` creates a [`DebugIteration`](@ref)
* `:ProxParameter` creates a [`DebugProximalParameter`](@ref)
* `:Stepsize` creates a [`DebugStepsize`](@ref)
* `:Stop` creates a [`DebugStoppingCriterion`](@ref)
* `:Time` creates a [`DebugTime`](@ref)
* `:IterativeTime` creates a [`DebugTime`](@ref)`(:Iterative)`

any other symbol creates a `DebugEntry(s)` to print the entry (o.:s) from the options.
"""
function DebugActionFactory(t::Tuple{Symbol, Any})
    (t[1] == :Change) && return DebugChange(; format = t[2])
    (t[1] == :Cost) && return DebugCost(; format = t[2])
    (t[1] == :Feasibility) && return DebugFeasibility(t[2])
    (t[1] == :Gradient) && return DebugGradient(; format = t[2])
    (t[1] == :GradientChange) && return DebugGradientChange(; format = t[2])
    (t[1] == :GradientNorm) && return DebugGradientNorm(; format = t[2])
    (t[1] == :Iteration) && return DebugIteration(; format = t[2])
    (t[1] == :Iterate) && return DebugIterate(; format = t[2])
    (t[1] == :IterativeTime) && return DebugTime(; mode = :Iterative, format = t[2])
    (t[1] == :ProxParameter) && return DebugProximalParameter(; format = t[2])
    (t[1] == :Stepsize) && return DebugStepsize(; format = t[2])
    (t[1] == :Stop) && return DebugStoppingCriterion(t[2])
    (t[1] == :Time) && return DebugTime(; format = t[2])
    ((t[1] == :Messages) || (t[1] == :InfoMessages)) && return DebugMessages(:Info, t[2])
    (t[1] == :WarningMessages) && return DebugMessages(:Warning, t[2])
    (t[1] == :ErrorMessages) && return DebugMessages(:Error, t[2])
    return DebugEntry(t[1]; format = t[2])
end
