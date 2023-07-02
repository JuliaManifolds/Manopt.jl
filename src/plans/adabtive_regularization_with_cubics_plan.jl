@doc raw"""
    AdaptiveRegularizationCubicCost

We define the model ``m(X)`` in the tangent space of the current iterate ``p=p_k`` as

```math
    m(X) = f(p) + <X, \operatorname{grad}f(p)>
      + \frac{1}{2} <X, \operatorname{Hess} f(p)[X]> +  \frac{σ}{3} \lVert X \rVert^3
```

# Fields
* `mho` – an [`AbstractManifoldObjective`](@ref) that should provide at least [`get_cost`](@ref), [`get_gradient`](@ref) and [`get_Hessian`](@ref).
* `σ` – the current regularization parameter

# Constructor
    AdaptiveRegularizationCubicCost(mho, σ)

"""
mutable struct AdaptiveRegularizationCubicCost{R,O<:AbstractManifoldObjective}
    mho::O
    σ::R
end
function (f::AdaptiveRegularizationCubicCost)(M, p, X)
    return get_cost(M, f.mho, p) +
           inner(M, p, X, get_gradient(M, f.mho, p)) +
           1 / 2 * inner(M, p, X, get_hessian(M, f.mho, p, X)) +
           f.σ / 3 * norm(M, p, X)^3
end

@doc raw"""
    AdaptiveRegularizationCubicGrad

We define the model ``m(X)`` in the tangent space of the current iterate ``p=p_k`` as

```math
    m(X) = f(p) + <X, \operatorname{grad}f(p)>
      + \frac{1}{2} <X, \operatorname{Hess} f(p)[X]> +  \frac{σ}{3} \lVert X \rVert^3
```

This struct represents its gradient, given by

```math
    \operatorname{grad} m(X) = \operatorname{grad}f(p) + \operatorname{Hess} f(p)[X] + σ \lVert X \rVert^2X
```

# Fields
* `mho` – an [`AbstractManifoldObjective`](@ref) that should provide at least [`get_cost`](@ref), [`get_gradient`](@ref) and [`get_Hessian`](@ref).
* `σ` – the current regularization parameter

# Constructor
    AdaptiveRegularizationCubicGrad(mho, σ)
"""
struct AdaptiveRegularizationCubicGrad{R,O<:AbstractManifoldObjective}
    mho::O
    σ::R
end
function (grad_f::AdaptiveRegularizationCubicGrad)(M, p, X)
    return get_gradient(M, f.mho, p)get_hessian(M, f.mho, p, X) + f.σ * norm(M, p, X)^2 * X
end
