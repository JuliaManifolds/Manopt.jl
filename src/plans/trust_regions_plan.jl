
@doc raw"""
    TrustRegionModelObjective{TH<:Union{Function,Nothing},O<:AbstractManifoldHessianObjective,T} <: AbstractManifoldSubObjective{O}

A trust region model of the form

```math
    m(X) = f(p) + ⟨\operatorname{grad} f(p), X⟩_p + \frac{1}(2} ⟨\operatorname{Hess} f(p)[X], X⟩_p
```

where

and we further dtore the current trust region radius ``Δ`` is the current trust region radius

# Fields

* `objective` – an [`AbstractManifoldHessianObjective`](@ref) proving ``f``, its gradient and Hessian

If `H` is set to nothing, the hessian from the `objective` is used.
"""
struct TrustRegionModelObjective{
    E<:AbstractEvaluationType,
    O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective},
} <: AbstractManifoldSubObjective{E,O}
    objective::O
end
function TrustRegionModelObjective(mho::O) where {E,O<:AbstractManifoldHessianObjective{E}}
    return TrustRegionModelObjective{E,O}(mho)
end
function TrustRegionModelObjective(
    mho::O
) where {E,O<:AbstractDecoratedManifoldObjective{E}}
    return TrustRegionModelObjective{E,O}(mho)
end

# TODO: Document

get_objective(trmo::TrustRegionModelObjective) = trmo.objective

function get_cost(TpM::TangentSpace, trmo::TrustRegionModelObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    c = get_objective_cocst(M, trmo, p)
    G = get_objective_gradient(M, trmo, p)
    Y = get_objective_hessian(M, trmo, p, X)
    return c + inner(M, p, G, X) + 1 / 2 * inner(M, p, Y, X)
end
function get_gradient(TpM::TangentSpace, trmo::TrustRegionModelObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    return get_objective_gradient(M, trmo, p) + get_objective_hessian(M, trmo, p, X)
end
function get_gradient!(
    TpM::TangentSpace, Y, trmo::TrustRegionModelObjective{InplaceEvaluation}, X
)
    M = base_manifold(TpM)
    p = TpM.point
    get_objective_hessian!(M, Y, trmo, p, X)
    Y .+= get_objective_gradient(M, trmo, p)
    return Y
end
function get_hessian(
    TpM::TangentSpace, trmo::TrustRegionModelObjective{InplaceEvaluation}, X, V
)
    M = base_manifold(TpM)
    p = TpM.point
    return get_objective_hessian(M, trmo, p, V)
end
function get_hessian!(
    TpM::TangentSpace, W, trmo::TrustRegionModelObjective{InplaceEvaluation}, V
)
    M = base_manifold(TpM)
    p = TpM.point
    return get_objective_hessian!(M, W, trmo, p, V)
end
