@doc """
    AbstractManifoldFirstOrderObjective{F,G} <: AbstractManifoldCostObjective{F}

An abstract type for all objectives that provide

* a cost – reflected by the type `F`
* first order information, so either a (full) gradient or a differential,
  or a subgradient – reflected by the parameter `G`.
"""
abstract type AbstractManifoldFirstOrderObjective{F, G} <:
AbstractManifoldCostObjective{F} end

@doc """
    (c, d) = get_cost_and_differential(M, objective::AbstractManifoldFirstOrderObjective, p, X; Y=nothing)

Evaluate the cost and the differential of a [`AbstractManifoldFirstOrderObjective`] `objective` at a point `p` in direction `X`.

The keyword argument `Y` can be used to provide memory for the evaluation of the gradient,
in case no separate differential is provided.
"""
function get_cost_and_differential end

function get_cost_and_gradient! end
@doc """
    (c, X) = get_cost_and_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    (c, X) = get_cost_and_gradient!(M, X, objective::AbstractManifoldFirstOrderObjective, p)

Evaluate the cost and the gradient of a [`AbstractManifoldFirstOrderObjective`] `objective` at a point `p`
simultaneously. The gradient part can be evaluated in-place of `X`
"""
function get_cost_and_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    X = zero_vector(M, p)
    return get_cost_and_gradient!(M, objective, objective, p)
end

@doc """
    d = get_differential(M, objective::AbstractManifoldFirstOrderObjective, p, X; gradient=nothing)

Evaluate the differential of a [`AbstractManifoldFirstOrderObjective`] `objective` at a point `p` in direction `X`.

The keyword argument `Y` can be used to provide memory for the evaluation of the gradient,
in case no separate differential is provided.
"""
function get_differential end


@doc """
    get_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    get_gradient!(M, X, objective::AbstractManifoldFirstOrderObjective, p)

Evaluate the gradient of a [`AbstractManifoldFirstOrderObjective`] `objective` at a point `p`.
This can be evaluated in-place of `X`.
"""
function get_gradient(M, objective::AbstractManifoldFirstOrderObjective, p)
    X = zero_vector(M, p)
    return get_gradient!(M, X, objective, p)
end

# Decorator case
function get_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient(M, get_objective(admo, false), p)
end
function get_gradient!(M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradient!(M, X, get_objective(admo, false), p)
end
