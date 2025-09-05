
@doc """
    TrustRegionModelObjective{O<:AbstractManifoldHessianObjective} <: AbstractManifoldSubObjective{O}

A trust region model of the form

```math
    m(X) = f(p) + ⟨$(_tex(:grad)) f(p), X⟩_p + $(_tex(:frac, "1", "2")) ⟨$(_tex(:Hess)) f(p)[X], X⟩_p
```

# Fields

* `objective`: an [`AbstractManifoldHessianObjective`](@ref) proving ``f``, its gradient and Hessian

# Constructors

    TrustRegionModelObjective(objective)

with either an [`AbstractManifoldHessianObjective`](@ref) `objective` or an decorator containing such an objective
"""
struct TrustRegionModelObjective{
        E <: AbstractEvaluationType,
        O <: Union{ManifoldHessianObjective, AbstractDecoratedManifoldObjective},
    } <: AbstractManifoldSubObjective{E, O}
    objective::O
end
function TrustRegionModelObjective(
        mho::O
    ) where {
        E, O <: Union{AbstractManifoldHessianObjective{E}, AbstractDecoratedManifoldObjective{E}},
    }
    return TrustRegionModelObjective{E, O}(mho)
end

get_objective(trmo::TrustRegionModelObjective) = trmo.objective

@doc """
    get_cost(TpM, trmo::TrustRegionModelObjective, X)

Evaluate the tangent space [`TrustRegionModelObjective`](@ref)

```math
m(X) = f(p) + ⟨$(_tex(:grad)) f(p), X ⟩_p + $(_tex(:frac, "1", "2")) ⟨$(_tex(:Hess)) f(p)[X], X⟩_p.
```
"""
function get_cost(TpM::TangentSpace, trmo::TrustRegionModelObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    c = get_objective_cost(M, trmo, p)
    G = get_objective_gradient(M, trmo, p)
    Y = get_objective_hessian(M, trmo, p, X)
    return c + inner(M, p, G, X) + 1 / 2 * inner(M, p, Y, X)
end
@doc """
    get_gradient(TpM, trmo::TrustRegionModelObjective, X)

Evaluate the gradient of the [`TrustRegionModelObjective`](@ref)

```math
$(_tex(:grad)) m(X) = $(_tex(:grad)) f(p) + $(_tex(:Hess)) f(p)[X].
```
"""
function get_gradient(TpM::TangentSpace, trmo::TrustRegionModelObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    return get_objective_gradient(M, trmo, p) + get_objective_hessian(M, trmo, p, X)
end
function get_gradient!(TpM::TangentSpace, Y, trmo::TrustRegionModelObjective, X)
    M = base_manifold(TpM)
    p = TpM.point
    get_objective_hessian!(M, Y, trmo, p, X)
    Y .+= get_objective_gradient(M, trmo, p)
    return Y
end
@doc """
    get_hessian(TpM, trmo::TrustRegionModelObjective, X)

Evaluate the Hessian of the [`TrustRegionModelObjective`](@ref)

```math
$(_tex(:Hess)) m(X)[Y] = $(_tex(:Hess)) f(p)[Y].
```
"""
function get_hessian(TpM::TangentSpace, trmo::TrustRegionModelObjective, X, V)
    M = base_manifold(TpM)
    p = TpM.point
    return get_objective_hessian(M, trmo, p, V)
end
function get_hessian!(TpM::TangentSpace, W, trmo::TrustRegionModelObjective, X, V)
    M = base_manifold(TpM)
    p = TpM.point
    return get_objective_hessian!(M, W, trmo, p, V)
end
