@doc raw"""
    ManifoldCostObjective{T,Tcost} <: AbstractManifoldObjective

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
struct ManifoldCostObjective{T,Tcost} <: AbstractManifoldObjective{T}
    cost::Tcost
end
function ManifoldCostObjective(cost::Tcost) where {Tcost}
    return ManifoldCostObjective{AllocatingEvaluation,Tcost}(cost)
end

function get_cost(M::AbstractManifold, mco::ManifoldCostObjective, p)
    return mco.cost(M, p)
end
