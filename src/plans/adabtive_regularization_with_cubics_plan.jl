@doc raw"""
    AdaptiveRagularizationWithCubicsModelObjective

A model for the adaptive regularization with Cubics

```math
m(X) = f(p) + ⟨\operatorname{grad} f(p), X ⟩_p + \frac{1}{2} ⟨\operatorname{Hess} f(p)[X], X⟩_p
       +  \frac{σ}{3} \lVert X \rVert^3,
```

cf. Eq. (33) in [AgarwalBoumalBullinsCartis:2020](@cite)

# Fields

* `objective` – an [`AbstractManifoldHessianObjective`](@ref) proving ``f``, its gradient and Hessian
* `σ` – the current (cubic) regularization parameter

# Constructors

    AdaptiveRagularizationWithCubicsModelObjective(mho, σ=1.0)

with either an [`AbstractManifoldHessianObjective`](@ref) `objective` or an decorator containing such an objective.
"""
mutable struct AdaptiveRagularizationWithCubicsModelObjective{
    E<:AbstractEvaluationType,
    O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective},
    R,
} <: AbstractManifoldSubObjective{E,O}
    objective::O
    σ::R
end
function AdaptiveRagularizationWithCubicsModelObjective(
    mho::O, σ::R=1.0
) where {E,O<:AbstractManifoldHessianObjective{E},R}
    return AdaptiveRagularizationWithCubicsModelObjective{E,O,R}(mho, σ)
end
function AdaptiveRagularizationWithCubicsModelObjective(
    mho::O, σ::R=1.0
) where {E,O<:AbstractDecoratedManifoldObjective{E},R}
    return AdaptiveRagularizationWithCubicsModelObjective{E,O,R}(mho, σ)
end
function set_manopt_parameter!(
    f::AdaptiveRagularizationWithCubicsModelObjective, ::Val{:σ}, σ
)
    f.σ = σ
    return f
end
function set_manopt_parameter!(
    f::AdaptiveRagularizationWithCubicsModelObjective, ::Val{:RegularizationParameter}, σ
)
    f.σ = σ
    return f
end

get_objective(arcmo::AdaptiveRagularizationWithCubicsModelObjective) = arcmo.objective

@doc raw"""
    get_cost(TpM, trmo::AdaptiveRagularizationWithCubicsModelObjective, X)

Evaluate the tangent space [`AdaptiveRagularizationWithCubicsModelObjective`](@ref)

```math
m(X) = f(p) + ⟨\operatorname{grad} f(p), X ⟩_p + \frac{1}{2} ⟨\operatorname{Hess} f(p)[X], X⟩_p
       +  \frac{σ}{3} \lVert X \rVert^3,
```

at `X`, cf. Eq. (33) in [AgarwalBoumalBullinsCartis:2020](@cite).
"""
function get_cost(
    TpM::TangentSpace, trmo::AdaptiveRagularizationWithCubicsModelObjective, X
)
    M = base_manifold(TpM)
    p = TpM.point
    c = get_objective_cocst(M, trmo, p)
    G = get_objective_gradient(M, trmo, p)
    Y = get_objective_hessian(M, trmo, p, X)
    return c + inner(M, p, G, X) + 1 / 2 * inner(M, p, Y, X) + σ * norm(M, p, X)^3
end
@doc raw"""
    get_gradient(TpM, trmo::AdaptiveRagularizationWithCubicsModelObjective, X)

Evaluate the gradient of the [`AdaptiveRagularizationWithCubicsModelObjective`](@ref)

```math
\operatorname{grad} m(X) = \operatorname{grad} f(p) + \operatorname{Hess} f(p)[X]
       + σ\lVert X \rVert X,
```

at `X`, cf. Eq. (37) in [AgarwalBoumalBullinsCartis:2020](@cite).
"""
function get_gradient(
    TpM::TangentSpace, arcmo::AdaptiveRagularizationWithCubicsModelObjective, X
)
    M = base_manifold(TpM)
    p = TpM.point
    return get_objective_gradient(M, arcmo, p) +
           get_objective_hessian(M, arcmo, p, X) +
           arcmo.σ * norm(M, p, X) * X
end
function get_gradient!(
    TpM::TangentSpace,
    Y,
    arcmo::AdaptiveRagularizationWithCubicsModelObjective,
    X,
)
    M = base_manifold(TpM)
    p = TpM.point
    get_objective_hessian!(M, Y, arcmo, p, X)
    Y .+= get_objective_gradient(M, arcmo, p) + arcmo.σ * norm(M, p, X) * X
    return Y
end
# Also Implement the Hessian for Newton subsubsolver?
