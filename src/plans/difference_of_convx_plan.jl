@doc raw"""
    ManifoldDifferenceOfConvexObjective{E} <: AbstractManifoldCostObjective{E}

Specify an objetive for a [`difference_of_convex`](@ref) algorithm.

The objective ``f: \mathcal M \to ℝ`` is given as

```math
    f(p) = g(p) - h(p)
```

where both ``g`` and ``h`` are convex, lsc. and proper.
Furthermore we assume that the subdifferential ``∂h`` of ``h`` is given.

# Fields

* `cost` – an implementation of ``f(p) = g(p)-h(p)``
* `∂h!!` – a deterministic version of ``∂h: \mathcal M → T\mathcal M``,
  i.e. calling `∂h(M, p)` returns a subgradient of ``h`` at `p` and if there is more than
  one, it returns a deterministic choice.

Note that the subdifferential might be given in two possible signatures
* `∂h(M,p)` which does an [`AllocatingEvaluation`](@ref)
* `∂h!(M, X, p)` which does an [`InplaceEvaluation`](@ref) in place of `X`.
"""
struct ManifoldDifferenceOfConvexObjective{E,TCost,TSubGrad} <:
       AbstractManifoldCostObjective{E,TCost}
    cost::TCost
    ∂h!!::TSubGrad
    function ManifoldDifferenceOfConvexObjective(
        cost::TC, ∂h::TSH; evaluation::AbstractEvaluationType=AllocatingEvaluation()
    ) where {TC,TSH}
        return new{typeof(evaluation),TC,TSH}(cost, ∂h)
    end
end

"""
    get_subgradient(p, q)
    get_subgradient!(p, X, q)

Evaluate the (sub)gradient of a [`ManifoldDifferenceOfConvexObjective`](@ref)` p` at the point `q`.

The evaluation is done in place of `X` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`InplaceEvaluation`](@ref)
memory for the result is allocated.
"""
function get_subgradient(
    M::AbstractManifold, doco::ManifoldDifferenceOfConvexObjective{AllocatingEvaluation}, p
)
    return doco.∂h!!(M, p)
end
function get_subgradient(
    M::AbstractManifold, doco::ManifoldDifferenceOfConvexObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    return doco.∂h!!(M, X, p)
end
function get_subgradient!(
    M::AbstractManifold,
    doco::ManifoldDifferenceOfConvexObjective{AllocatingEvaluation},
    X,
    p,
)
    return copyto!(M, p, X, doco.∂h!!(M, p))
end
function get_subgradient!(
    M::AbstractManifold, doco::ManifoldDifferenceOfConvexObjective{InplaceEvaluation}, X, p
)
    return doco.∂h!!(M, X, p)
end

@doc raw"""
    LinearizedSubProblem

A functor `(M,q) → ℝ` to represent the inner problem of a [`ManifoldDifferenceOfConvexObjective`](@ref),
i.e. a cost function of the form

```math
    F_{p_k,X_k}(p) = g(p) - ⟨X_k, \log_p_kp⟩
```
for a point `p_k` and a tangent vector `X_k` at `p_k` (e.g. outer iterates)
that are stored within this functor as well.

# Fields

* `g` a function
* `p` a point on a manifold
* `X` a tangent vector at `p`
"""
mutable struct LinearizedDCCost{P,T,TG}
    g::TG
    pk::P
    Xk::T
end
(F::LinearizedDCCost)(M, p) = F.f(p) - inner(M, F.pk, F.Xk, log(M, F.pk, p))

function set_manopt_parameter!(ldc::LinearizedDCCost, ::Val{:p}, ρ)
    return ldc.pk .= p
    return ldc
end
function set_manopt_parameter!(ldc::LinearizedDCCost, ::Val{:X}, X)
    ldc.Xk = X
    return ldc
end

@doc raw"""
    LinearizedDCGrad

A functor `(M,X,p) → ℝ` to represent the gradient of the inner problem of a [`ManifoldDifferenceOfConvexObjective`](@ref),
i.e. for a cost function of the form

```math
    F_{p_k,X_k}(p) = f(p) - ⟨X_k, \log_p_kp⟩
```

its gradient is given by using ``F=F_1(F_2(p))``, where ``F_1(X) = ⟨X_k,X⟩`` and ``F_2(p) = \log_p_kp``
and the chain rule as well as the [`adjoint_differential_log_argument`](@ref) for ``D^*F_2(p)``

```math
    \operatorname{grad} F(q) = \operatorname{grad} f(q) - DF_2^*(q)[X]
```

for a point `pk` and a tangent vector `Xk` at `pk` (the outer iterates) that are stored within this functor as well

# Fields

* `grad_g` the gradient of ``g`` (see [`LinearizedSubProblem`](@ref)) as
* `pk` a point on a manifold
* `Xk` a tangent vector at `pk`

# Constructor
    LinearizedDCGrad(grad_f, p, X; evaluation=AllocatingEvaluation())

Where you specify whether `grad_g` is [`AllocatingEvaluation`](@ref) or [`InplaceEvaluation`](@ref),
while this function still provides _both_ signatures.
"""
mutable struct LinearizedDCGrad{E<:AbstractEvaluationType,P,T,TG}
    grad_g::TG
    pk::P
    Xk::T
    function LinearizedDCGrad(
        grad_g::TG, pk::P, Xk::T; evaluation::E=AllocatingEvaluation()
    ) where {TG,P,T,E<:AbstractEvaluationType}
        return new{E,P,T,TG}(grad_g, pk, Xk)
    end
end
function (grad_F::LinearizedDCGrad{AllocatingEvaluation})(M, p)
    return grad_F.grad_g(M, p) .-
           adjoint_differential_log_argument(M, grad_F.pk, p, grad_F.Xk)
end
function (grad_F::LinearizedDCGrad{AllocatingEvaluation})(M, X, p)
    copyto!(
        M,
        X,
        p,
        grad_F.grad_f(M, p) .-
        adjoint_differential_log_argument(M, grad_F.pk, p, grad_F.Xk),
    )
    return Y
end
function (grad_F!::LinearizedDCGrad{InplaceEvaluation})(M, X, p)
    grad_F!.grad_f(M, X, p)
    X .-= adjoint_differential_log_argument(M, grad_F!.pk, p, grad_F!.Xk)
    return X
end
function (grad_F!::LinearizedDCGrad{InplaceEvaluation})(M, p)
    X = zero_vector(M, p)
    grad_F!.grad_g(M, X, p)
    X .-= adjoint_differential_log_argument(M, grad_F!.pk, p, grad_F!.Xk)
    return Y
end

function set_manopt_parameter!(ldcg::LinearizedDCGrad, ::Val{:p}, ρ)
    return ldcg.pk .= p
    return ldcg
end
function set_manopt_parameter!(ldcg::LinearizedDCGrad, ::Val{:X}, X)
    ldcg.Xk = X
    return ldcg
end
