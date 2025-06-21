@doc """
    DifferentialFunction{D} <: AbstractFirstOrderFunction

A wrapper for a function representing the differential ``Df: $(_math(:TM)) → ℝ``,
or in other words is is a map ``Df(p)[X] ∈ ℝ``.
Compared to a usual function or functor, this type is meant to distinguish
the differential from the gradient of a function, since both have similar
signatures that would only be distinguishable if we require types points and vectors.

# Fields
* `diff_f!!::D`: a function or functor for the gradient

# Constructor

    DifferentialFunction(diff_f::D)

Create a differential function `diff_f`
"""
struct DifferentialFunction{D} <: AbstractFirstOrderFunction
    diff_f!!::D
end
# TODO: get_gradient for this struct?

#
# Access the differential
# -----------------------------
@doc """
    get_differential(amgo::AbstractManifoldFirstOrderObjective, M, p, X)

return the differential of an [`AbstractManifoldFirstOrderObjective`](@ref) `amgo`
on the [`AbstractManifold`](@extref) `M` at the point `p` and with tangent vector `X`.
"""
get_differential(amgo::AbstractManifoldFirstOrderObjective, M::AbstractManifold, p, X)

function get_differential(
    amfoo::AbstractManifoldFirstOrderObjective{E,F,DifferentialFunction},
    M::AbstractManifold,
    p,
    X;
) where {E,F}
    return amfoo.firsy_order!!(M, p, X)
end
# Default – functions, funtors as well as GradientFunctions
function get_differential(
    amfoo::AbstractManifoldFirstOrderObjective, M::AbstractManifold, p, X;
)
    return inner(M, p, get_gradient(amfoo, M, p, X), X)
end
