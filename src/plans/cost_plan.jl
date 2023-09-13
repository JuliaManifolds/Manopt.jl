@doc raw"""
    AbstractManifoldCostObjective{T<:AbstractEvaluationType} <: AbstractManifoldObjective{T}

Representing objectives on manifolds with a cost function implemented.
"""
abstract type AbstractManifoldCostObjective{T<:AbstractEvaluationType,TC} <:
              AbstractManifoldObjective{T} end

@doc raw"""
    ManifoldCostObjective{T, TC} <: AbstractManifoldCostObjective{T, TC}

speficy an [`AbstractManifoldObjective`](@ref) that does only have information about
the cost function ``f\colon \mathbb M → ℝ`` implemented as a function `(M, p) -> c`
to compute the cost value `c` at `p` on the manifold `M`.

* `cost` – a function ``f: \mathcal M → ℝ`` to minimize

# Constructors

    ManifoldCostObjective(f)

Generate a problem. While this Problem does not have any allocating functions,
the type `T` can be set for consistency reasons with other problems.

# Used with
[`NelderMead`](@ref), [`particle_swarm`](@ref)
"""
struct ManifoldCostObjective{T<:AbstractEvaluationType,TC} <:
       AbstractManifoldCostObjective{T,TC}
    cost::TC
end
function ManifoldCostObjective(cost::Tcost) where {Tcost}
    return ManifoldCostObjective{AllocatingEvaluation,Tcost}(cost)
end
@doc raw"""
    get_cost(M::AbstractManifold, mco::AbstractManifoldCostObjective, p)

Evaluate the cost function from within the [`AbstractManifoldCostObjective`](@ref) on `M`
at `p`.

By default this implementation assumed that the cost is stored within `mco.cost`.
"""
function get_cost(M::AbstractManifold, mco::AbstractManifoldCostObjective, p)
    return get_cost_function(mco)(M, p)
end
function get_cost(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_cost(M, get_objective(admo, false), p)
end

@doc raw"""
    get_cost_function(amco::AbstractManifoldCostObjective)

return the function to evaluate (just) the cost ``f(p)=c`` as a function `(M,p) -> c`.
"""
get_cost_function(mco::AbstractManifoldCostObjective, recursive=false) = mco.cost
function get_cost_function(admo::AbstractDecoratedManifoldObjective, recursive=false)
    return get_cost_function(get_objective(admo, recursive))
end
