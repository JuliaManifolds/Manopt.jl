@doc raw"""
    AdaptiveRegularizationCubicCost

We define the model ``m(X)`` in the tangent space of the current iterate ``p=p_k`` as

```math
    m(X) = f(p) + <X, \operatorname{grad}f(p)>
      + \frac{1}{2} <X, \operatorname{Hess} f(p)[X]> +  \frac{σ}{3} \lVert X \rVert^3
```

# Fields
* `mho` – an [`AbstractManifoldObjective`](@ref) that should provide at least [`get_cost`](@ref), [`get_gradient`](@ref) and [`get_hessian`](@ref).
* `σ` – the current regularization parameter
* `X` – a storage for the gradient at `p` of the original cost

# Constructors

    AdaptiveRegularizationCubicCost(mho, σ, X)
    AdaptiveRegularizationCubicCost(M, mho, σ; p=rand(M), X=get_gradient(M, mho, p))

Initialize the cubic cost to the objective `mho`, regularization parameter `σ`, and
(temporary) gradient `X`.

!!! note
    For this gradient function to work, we require the [`TangentSpaceAtPoint`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/vector_bundle.html#Manifolds.TangentSpaceAtPoint)
    from `Manifolds.jl`
"""
mutable struct AdaptiveRegularizationCubicCost{T,R,O<:AbstractManifoldObjective}
    mho::O
    σ::R
    X::T
end
function AdaptiveRegularizationCubicCost(
    M, mho::O, σ::R; p::P=rand(M), X::T=get_gradient(M, mho, p)
) where {P,T,R,O}
    return AdaptiveRegularizationCubicCost{T,R,O}(mho, σ, X)
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
    \operatorname{grad} m(X) = \operatorname{grad}f(p) + \operatorname{Hess} f(p)[X] + σ \lVert X \rVert^2 X
```

# Fields

* `mho` – an [`AbstractManifoldObjective`](@ref) that should provide at least [`get_cost`](@ref), [`get_gradient`](@ref) and [`get_hessian`](@ref).
* `σ` – the current regularization parameter
* `X` – a storage for the gradient at `p` of the original cost

# Constructors

    AdaptiveRegularizationCubicGrad(mho, σ, X)
    AdaptiveRegularizationCubicGrad(M, mho, σ; p=rand(M), X=get_gradient(M, mho, p))

Initialize the cubic cost to the original objective `mho`, regularization parameter `σ`, and
(temporary) gradient `X`.

!!! note
    * For this gradient function to work, we require the [`TangentSpaceAtPoint`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/vector_bundle.html#Manifolds.TangentSpaceAtPoint)
    from `Manifolds.jl`
    * The gradient functor provides both an allocating as well as an in-place variant.
"""
struct AdaptiveRegularizationCubicGrad{T,R,O<:AbstractManifoldObjective}
    mho::O
    σ::R
    X::T
end
function AdaptiveRegularizationCubicGrad(
    M, mho::O, σ::R; p::P=rand(M), X::T=get_gradient(M, mho, p)
) where {P,T,R,O}
    return AdaptiveRegularizationCubicGrad{T,R,O}(mho, σ, p, X)
end
