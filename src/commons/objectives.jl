@doc """
    ManifoldCostObjective{F} <: AbstractManifoldCostObjective{F}

specify an [`AbstractManifoldObjective`](@ref) that does only have information about
the cost function ``f:  $(_math(:Manifold)) → ℝ`` implemented as a function `(M, p) -> c`
to compute the cost value `c` at `p` on the manifold `M`.

* `cost`: a function ``f: $(_math(:Manifold)) → ℝ`` to minimize

# Constructors

    ManifoldCostObjective(f::F)

Generate a problem. While this Problem does not have any allocating functions,

## See also
[`NelderMead`](@ref), [`particle_swarm`](@ref)
"""
struct ManifoldCostObjective{F} <: AbstractManifoldCostObjective{F}
    cost::F
end
function show(io::IO, mco::ManifoldCostObjective{F}) where {F}
    return print(io, "ManifoldCostObjective(mco.cost)")
end
function status_summary(::ManifoldCostObjective{F}; context::Symbol = :default) where {F}
    return "A cost function on a Riemannian manifold `f = (M,p) -> ℝ`."
end
