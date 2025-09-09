@doc """
    AdaptiveRegularizationWithCubicsModelObjective

A model for the adaptive regularization with Cubics

```math
m(X) = f(p) + ⟨$(_tex(:grad)) f(p), X ⟩_p + $(_tex(:frac, "1", "2")) ⟨$(_tex(:Hess)) f(p)[X], X⟩_p
       +  $(_tex(:frac, "σ", "3")) $(_tex(:norm, "X"))^3,
```

cf. Eq. (33) in [AgarwalBoumalBullinsCartis:2020](@cite)

# Fields

* `objective`: an [`AbstractManifoldHessianObjective`](@ref) proving ``f``, its gradient and Hessian
* `σ`:         the current (cubic) regularization parameter

# Constructors

    AdaptiveRegularizationWithCubicsModelObjective(mho, σ=1.0)

with either an [`AbstractManifoldHessianObjective`](@ref) `objective` or an decorator containing such an objective.
"""
mutable struct AdaptiveRegularizationWithCubicsModelObjective{
        E <: AbstractEvaluationType,
        O <: Union{ManifoldHessianObjective, AbstractDecoratedManifoldObjective},
        R,
    } <: AbstractManifoldSubObjective{E, O}
    objective::O
    σ::R
end
function AdaptiveRegularizationWithCubicsModelObjective(
        mho::O, σ::R = 1.0
    ) where {
        E, O <: Union{AbstractManifoldHessianObjective{E}, AbstractDecoratedManifoldObjective{E}}, R,
    }
    return AdaptiveRegularizationWithCubicsModelObjective{E, O, R}(mho, σ)
end
function set_parameter!(
        f::AdaptiveRegularizationWithCubicsModelObjective,
        ::Union{Val{:σ}, Val{:RegularizationParameter}},
        σ,
    )
    f.σ = σ
    return f
end

get_objective(arcmo::AdaptiveRegularizationWithCubicsModelObjective) = arcmo.objective

@doc """
    get_cost(TpM, trmo::AdaptiveRegularizationWithCubicsModelObjective, X)

Evaluate the tangent space [`AdaptiveRegularizationWithCubicsModelObjective`](@ref)

```math
m(X) = f(p) + ⟨$(_tex(:grad)) f(p), X ⟩_p + $(_tex(:frac, "1", "2")) ⟨$(_tex(:Hess)) f(p)[X], X⟩_p
       + $(_tex(:frac, "σ", "3")) $(_tex(:norm, "X"))^3,
```

at `X`, cf. Eq. (33) in [AgarwalBoumalBullinsCartis:2020](@cite).
"""
function get_cost(
        TpM::TangentSpace, arcmo::AdaptiveRegularizationWithCubicsModelObjective, X
    )
    M = base_manifold(TpM)
    p = TpM.point
    c = get_objective_cost(M, arcmo, p)
    G = get_objective_gradient(M, arcmo, p)
    Y = get_objective_hessian(M, arcmo, p, X)
    return c + inner(M, p, G, X) + 1 / 2 * inner(M, p, Y, X) + arcmo.σ / 3 * norm(M, p, X)^3
end
function get_cost_function(arcmo::AdaptiveRegularizationWithCubicsModelObjective)
    return (TpM, X) -> get_cost(TpM, arcmo, X)
end
@doc """
    get_gradient(TpM, trmo::AdaptiveRegularizationWithCubicsModelObjective, X)

Evaluate the gradient of the [`AdaptiveRegularizationWithCubicsModelObjective`](@ref)

```math
$(_tex(:grad)) m(X) = $(_tex(:grad)) f(p) + $(_tex(:Hess)) f(p)[X]
       + σ$(_tex(:norm, "X")) X,
```

at `X`, cf. Eq. (37) in [AgarwalBoumalBullinsCartis:2020](@cite).
"""
function get_gradient(
        TpM::TangentSpace, arcmo::AdaptiveRegularizationWithCubicsModelObjective, X
    )
    M = base_manifold(TpM)
    p = TpM.point
    G = get_objective_gradient(M, arcmo, p)
    return G + get_objective_hessian(M, arcmo, p, X) + arcmo.σ * norm(M, p, X) * X
end
function get_gradient!(
        TpM::TangentSpace, Y, arcmo::AdaptiveRegularizationWithCubicsModelObjective, X
    )
    M = base_manifold(TpM)
    p = TpM.point
    get_objective_hessian!(M, Y, arcmo, p, X)
    Y .= Y + get_objective_gradient(M, arcmo, p) + arcmo.σ * norm(M, p, X) * X
    return Y
end
function get_gradient_function(arcmo::AdaptiveRegularizationWithCubicsModelObjective)
    return (TpM, X) -> get_gradient(TpM, arcmo, X)
end
