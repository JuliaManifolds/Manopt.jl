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
