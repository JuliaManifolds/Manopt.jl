@doc raw"""
    SubGradientProblem <:AbstractManoptProblem

A structure to store information about a subgradient based optimization problem

# Fields
* `manifold` – a [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)
* `cost` – the function $F$ to be minimized
* `subgradient` – a function returning a subgradient $\partial F$ of $F$

# Constructor

    SubGradientProblem(M, f, ∂f)

Generate the [`Problem`] for a subgradient problem, i.e. a function `f` on the
manifold `M` and a function `∂f` that returns an element from the subdifferential
at a point.
"""
struct ManifoldSubgradientObjective{T<:AbstractEvaluationType,C,S} <:
       AbstractManifoldCostObjective{T}
    cost::C
    subgradient!!::S
    function ManifoldSubgradientObjective(
        cost::C, subgrad::S; evaluation::AbstractEvaluationType=AllocatingEvaluation()
    ) where {C,S}
        return new{typeof(evaluation),C,S}(cost, subgrad)
    end
end

@doc raw"""
    get_subgradient(mp::AbstractManoptProblem, p)
    get_subgradient!(mp::AbstractManoptProblem, X, p)

evaluate the subgradient of an [`AbstractManoptProblem`](@ref) `mp` at `p`.

The evaluation is done in place of `X` for the `!`-variant.
The result might not be deterministic, _one_ element of the subdifferential is returned.
"""
function get_subgradient(mp::AbstractManoptProblem, p)
    return get_subgradient(get_manifold(mp), get_objective(mp), p)
end
function get_subgradient!(mp::AbstractManoptProblem, X, p)
    return get_subgradient!(get_manifold(mp), X, get_objective(mp), p)
end

"""
    X = get_subgradient(M;;AbstractManifold, sgo::ManifoldSubgradientObjective, p)
    get_subgradient!(M;;AbstractManifold, X, sgo::ManifoldSubgradientObjective, p)

Evaluate the (sub)gradient of a [`ManifoldSubgradientObjective`](@ref) `sgo`
at the point `p`.

The evaluation is done in place of `X` for the `!`-variant.
The result might not be deterministic, _one_ element of the subdifferential is returned.
"""
function get_subgradient(
    M::AbstractManifold, sgo::ManifoldSubgradientObjective{AllocatingEvaluation}, p
)
    return sgo.subgradient!!(M, p)
end
function get_subgradient(
    M::AbstractManifold, sgo::ManifoldSubgradientObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    return sgo.subgradient!!(M, X, p)
end
function get_subgradient!(
    M::AbstractManifold, X, sgo::ManifoldSubgradientObjective{AllocatingEvaluation}, p
)
    copyto!(M, X, sgo.subgradient!!(M, p))
    return X
end
function get_subgradient!(
    M::AbstractManifold, X, sgo::ManifoldSubgradientObjective{InplaceEvaluation}, p
)
    sgo.subgradient!!(M, X, p)
    return X
end
