"""
    StopWhenGradientMappingNormLess <: StoppingCriterion

A stopping criterion based on the gradient mapping norm for proximal gradient methods.

# Fields

$(_fields(:at_iteration))
$(_fields(:last_change))
* `threshold`: the threshold for the change to check (run under to stop)

# Constructor

    StopWhenGradientMappingNormLess(ε)

Create a stopping criterion with threshold `ε` for the gradient mapping for the [`proximal_gradient_method`](@ref).
That is, this criterion indicates to stop when the gradient mapping has a norm less than `ε`.
The gradient mapping G_λ(p) is defined as -(1/λ) * log_p(T_λ(p)), where T_λ(p) is the proximal mapping prox_λ f(exp_p(-λ * grad f(p))).
"""
mutable struct StopWhenGradientMappingNormLess{TF} <: StoppingCriterion
    threshold::TF
    last_change::TF
    at_iteration::Int
    function StopWhenGradientMappingNormLess(ε::TF) where {TF}
        return new{TF}(ε, zero(ε), -1)
    end
end
function get_reason(c::StopWhenGradientMappingNormLess)
    if (c.last_change < c.threshold) && (c.at_iteration >= 0)
        return "The algorithm reached approximately critical point after $(c.at_iteration) iterations; the gradient mapping norm ($(c.last_change)) is less than $(c.threshold).\n"
    end
    return ""
end
indicates_convergence(c::StopWhenGradientMappingNormLess) = true
function Base.show(io::IO, c::StopWhenGradientMappingNormLess)
    return print(io, "StopWhenGradientMappingNormLess($(c.threshold))")
end
function status_summary(c::StopWhenGradientMappingNormLess; context::Symbol = :default)
    has_stopped = (c.at_iteration >= 0)
    s = has_stopped ? "reached" : "not reached"
    return (_is_inline(context) ? "|G| < $(c.threshold):$(_MANOPT_INDENT)" : "A stopping criterion to stop when the gradient mapping norm is less then a tolerance.\n$(_MANOPT_INDENT)") * s
end

#
#
# ---

@doc """
    StopWhenLagrangeMultiplierLess <: StoppingCriterion

Stopping Criteria for Lagrange multipliers.

Currently these are meant for the [`convex_bundle_method`](@ref) and [`proximal_bundle_method`](@ref),
where based on the Lagrange multipliers an approximate (sub)gradient ``g`` and an error estimate ``ε``
is computed.

The `mode=:both` requires that both
``ε`` and ``$(_tex(:abs, "g"))`` are smaller than their `tolerance`s for the [`convex_bundle_method`](@ref),
and that
``c`` and ``$(_tex(:abs, "d"))`` are smaller than their `tolerance`s for the [`proximal_bundle_method`](@ref).

The `mode=:estimate` requires that, for the [`convex_bundle_method`](@ref)
``-ξ = $(_tex(:abs, "g"))^2 + ε`` is less than a given `tolerance`.
For the [`proximal_bundle_method`](@ref), the equation reads ``-ν = μ $(_tex(:abs, "d"))^2 + c``.

# Constructors

    StopWhenLagrangeMultiplierLess(tolerance=1e-6; mode::Symbol=:estimate, names=nothing)

Create the stopping criterion for one of the `mode`s mentioned.
Note that tolerance can be a single number for the `:estimate` case,
but a vector of two values is required for the `:both` mode.
Here the first entry specifies the tolerance for ``ε`` (``c``),
the second the tolerance for ``$(_tex(:abs, "g"))`` (``$(_tex(:abs, "d"))``), respectively.
"""
mutable struct StopWhenLagrangeMultiplierLess{
        T <: Real, A <: AbstractVector{<:T}, B <: Union{Nothing, <:AbstractVector{<:String}},
    } <: StoppingCriterion
    tolerances::A
    values::A
    names::B
    mode::Symbol
    at_iteration::Int
    function StopWhenLagrangeMultiplierLess(
            tol::T; mode::Symbol = :estimate, names::B = nothing
        ) where {T <: Real, B <: Union{Nothing, <:AbstractVector{<:String}}}
        return new{T, Vector{T}, B}([tol], zero([tol]), names, mode, -1)
    end
    function StopWhenLagrangeMultiplierLess(
            tols::A; mode::Symbol = :estimate, names::B = nothing
        ) where {T <: Real, A <: AbstractVector{<:T}, B <: Union{Nothing, <:AbstractVector{<:String}}}
        return new{T, A, B}(tols, zero(tols), names, mode, -1)
    end
end
function get_reason(sc::StopWhenLagrangeMultiplierLess)
    if (sc.at_iteration >= 0)
        if isnothing(sc.names)
            tol_str = join(
                ["$ai < $bi" for (ai, bi) in zip(sc.values, sc.tolerances)], ", "
            )
        else
            tol_str = join(
                [
                    "$si = $ai < $bi" for
                        (si, ai, bi) in zip(sc.names, sc.values, sc.tolerances)
                ],
                ", ",
            )
        end
        return "After $(sc.at_iteration) iterations the algorithm reached an approximate critical point with tolerances $tol_str.\n"
    end
    return ""
end
function status_summary(sc::StopWhenLagrangeMultiplierLess; context::Symbol = :default)
    s = (sc.at_iteration >= 0) ? "reached" : "not reached"
    msg = "Lagrange multipliers"
    isnothing(sc.names) && (msg *= " with tolerances $(sc.tolerances)")
    if !isnothing(sc.names)
        msg *= join(["$si < $bi" for (si, bi) in zip(sc.names, sc.tolerances)], ", ")
    end
    return (_is_inline(context) ? "" : "A stopping criterion to stop when the Lagrange multipliers are less than $(sc.tolerances).\n$(_MANOPT_INDENT)") * "$(msg):$(_MANOPT_INDENT)$(s)"
end
function show(io::IO, sc::StopWhenLagrangeMultiplierLess)
    n = isnothing(sc.names) ? "" : ", $(names)"
    return print(
        io,
        "StopWhenLagrangeMultiplierLess($(sc.tolerances); mode=:$(sc.mode)$n)",
    )
end
