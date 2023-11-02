@doc raw"""
    ManifoldSubgradientObjective{T<:AbstractEvaluationType,C,S} <:AbstractManifoldCostObjective{T, C}

A structure to store information about a objective for a subgradient based optimization problem

# Fields
* `cost` – the function ``f`` to be minimized
* `subgradient` – a function returning a subgradient ``\partial f`` of ``f``

# Constructor

    ManifoldSubgradientObjective(f, ∂f)

Generate the [`ManifoldSubgradientObjective`](@ref) for a subgradient objective, i.e.
a (cost) function `f(M, p)` and a function `∂f(M, p)` that returns a not necessarily deterministic
element from the subdifferential at `p` on a manifold `M`.
"""
struct ManifoldSubgradientObjective{T<:AbstractEvaluationType,C,S} <:
       AbstractManifoldCostObjective{T,C}
    cost::C
    subgradient!!::S
    function ManifoldSubgradientObjective(
        cost::C, subgrad::S; evaluation::AbstractEvaluationType=AllocatingEvaluation()
    ) where {C,S}
        return new{typeof(evaluation),C,S}(cost, subgrad)
    end
end

@doc raw"""
    get_subgradient(amp::AbstractManoptProblem, p)
    get_subgradient!(amp::AbstractManoptProblem, X, p)

evaluate the subgradient of an [`AbstractManoptProblem`](@ref) `amp` at point `p`.

The evaluation is done in place of `X` for the `!`-variant.
The result might not be deterministic, _one_ element of the subdifferential is returned.
"""
function get_subgradient(amp::AbstractManoptProblem, p)
    return get_subgradient(get_manifold(amp), get_objective(amp), p)
end
function get_subgradient!(amp::AbstractManoptProblem, X, p)
    return get_subgradient!(get_manifold(amp), X, get_objective(amp), p)
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
function get_subgradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_subgradient(M, get_objective(admo, false), p)
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
function get_subgradient!(
    M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p
)
    return get_subgradient!(M, X, get_objective(admo, false), p)
end
