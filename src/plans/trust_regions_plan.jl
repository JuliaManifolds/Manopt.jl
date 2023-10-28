
@doc raw"""
    TrustRegionModelObjective{TH<:Union{Function,Nothing},O<:AbstractManifoldHessianObjective,T} <: AbstractManifoldSubObjective{O}

A trust region model of the form

```math
    f(X) = c + ⟨G, X⟩_p + \frac{1}(2} ⟨B(X), X⟩_p
```

where

* ``G`` is (a tangent vector that is an approximation of) the gradient ``\operatorname{grad} f(p)``
* ``B`` is (a bilinear form that is an approximantion of) the Hessian ``\operatorname{Hess} f(p)``
* ``c`` is the current cost ``f(p)``, but might be set to zero for simplicity, since we are only interested in the minimizer

and we further dtore the current trust region radius ``Δ`` is the current trust region radius

# Fields

* `objective` – an [`AbstractManifoldHessianObjective`](@ref) proving ``f``, its gradient and Hessian
* `c` the current cost at `p`
* `G` the current Gradient at `p``
* `H` the current bilinear form (Approximation of the Hessian)
* `Δ` the current trust region radius

If `H` is set to nothing, the hessian from the `objective` is used.
"""
struct TrustRegionModelObjective{
    E<:AbstractEvaluationType,
    TH<:Union{Function,Nothing},
    O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective},
    T,
    R,
} <: AbstractManifoldSubObjective{E,O}
    objective::O
    c::R
    G::T
    H::TH
    Δ::R
end
function TrustRegionModelObjective(TpM::TangentSpace, mho, p=rand(M); kwargs...)
    return TrustRegionModelObjective(base_manifold(TpM), mho, p; kwargs...)
end
function TrustRegionModelObjective(
    M::AbstractManifold,
    mho::O,
    p=rand(M);
    cost::R=get_cost(M, mho, p),
    trust_region_radius::R=injectivity_radius(M) / 8,
    gradient::T=get_gradient(M, mho, p),
    bilinear_form::TH=nothing,
) where {
    TH<:Union{Function,Nothing},
    E<:AbstractEvaluationType,
    O<:AbstractManifoldHessianObjective{E},
    T,
    R,
}
    return TrustRegionModelObjective{E,TH,O,T,R}(
        mho, cost, gradient, bilinear_form, trust_region_radius
    )
end

get_objective(trm::TrustRegionModelObjective) = trom.objective
