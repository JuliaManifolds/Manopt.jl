@doc raw"""
    ManifoldDifferenceOfConvexObjective{E} <: AbstractManifoldCostObjective{E}

Specify an objective for a [`difference_of_convex_algorithm`](@ref).

The objective ``f: \mathcal M → ℝ`` is given as

```math
    f(p) = g(p) - h(p)
```

where both ``g`` and ``h`` are convex, lower semicontinuous and proper.
Furthermore the subdifferential ``∂h`` of ``h`` is required.

# Fields

* `cost`: an implementation of ``f(p) = g(p)-h(p)`` as a function `f(M,p)`.
* `∂h!!`: a deterministic version of ``∂h: \mathcal M → T\mathcal M``,
  in the sense that calling `∂h(M, p)` returns a subgradient of ``h`` at `p` and
  if there is more than one, it returns a deterministic choice.

Note that the subdifferential might be given in two possible signatures

* `∂h(M,p)` which does an [`AllocatingEvaluation`](@ref)
* `∂h!(M, X, p)` which does an [`InplaceEvaluation`](@ref) in place of `X`.
"""
struct ManifoldDifferenceOfConvexObjective{E,F,G,S} <:
       AbstractManifoldFirstOrderObjective{E,Tuple{F,G}}
    cost::F
    gradient!!::G
    ∂h!!::S
    function ManifoldDifferenceOfConvexObjective(
        cost::TC, ∂h::TSH; gradient::TG=nothing, evaluation::ET=AllocatingEvaluation()
    ) where {ET<:AbstractEvaluationType,TC,TG,TSH}
        return new{ET,TC,TG,TSH}(cost, gradient, ∂h)
    end
end

function get_gradient_function(doco::ManifoldDifferenceOfConvexObjective, recursive=false)
    return doco.gradient!!
end

function get_gradient(M::AbstractManifold, doco::ManifoldDifferenceOfConvexObjective{AllocatingEvaluation}, p)
    return doco.gradient!!(M,p)
end
function get_gradient(
    M::AbstractManifold, doco::ManifoldDifferenceOfConvexObjective{InplaceEvaluation}, p
)
    X = zero_vector(M,p)
    return doco.gradient!!(M, X, p)
end
function get_gradient!(
    M::AbstractManifold, X, doco::ManifoldDifferenceOfConvexObjective{AllocatingEvaluation}, p
)
    return copyto!(M, X, p, doco.gradient!!(M, p))
end
function get_gradient!(
    M::AbstractManifold, X, doco::ManifoldDifferenceOfConvexObjective{InplaceEvaluation}, p
)
    return doco.gradient!!(M, X, p)
end

"""
    X = get_subtrahend_gradient(amp, q)
    get_subtrahend_gradient!(amp, X, q)

Evaluate the (sub)gradient of the subtrahend `h` from within
a [`ManifoldDifferenceOfConvexObjective`](@ref) `amp` at the point `q` (in place of `X`).

The evaluation is done in place of `X` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`InplaceEvaluation`](@ref)
memory for the result is allocated.
"""
function get_subtrahend_gradient(amp::AbstractManoptProblem, p)
    return get_subtrahend_gradient(get_manifold(amp), get_objective(amp), p)
end
function get_subtrahend_gradient!(amp::AbstractManoptProblem, X, p)
    get_subtrahend_gradient!(get_manifold(amp), X, get_objective(amp), p)
    return X
end

function get_subtrahend_gradient(
    M::AbstractManifold, doco::ManifoldDifferenceOfConvexObjective{AllocatingEvaluation}, p
)
    return doco.∂h!!(M, p)
end

function get_subtrahend_gradient(
    M::AbstractManifold, doco::ManifoldDifferenceOfConvexObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    return doco.∂h!!(M, X, p)
end
function get_subtrahend_gradient(
    M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p
)
    return get_subtrahend_gradient(M, get_objective(admo, false), p)
end

function get_subtrahend_gradient!(
    M::AbstractManifold,
    X,
    doco::ManifoldDifferenceOfConvexObjective{AllocatingEvaluation},
    p,
)
    return copyto!(M, X, p, doco.∂h!!(M, p))
end
function get_subtrahend_gradient!(
    M::AbstractManifold, X, doco::ManifoldDifferenceOfConvexObjective{InplaceEvaluation}, p
)
    return doco.∂h!!(M, X, p)
end
function get_subtrahend_gradient!(
    M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p
)
    return get_subtrahend_gradient!(M, X, get_objective(admo, false), p)
end

@doc raw"""
    LinearizedDCCost

A functor `(M,q) → ℝ` to represent the inner problem of a [`ManifoldDifferenceOfConvexObjective`](@ref).
This is a cost function of the form

```math
    F_{p_k,X_k}(p) = g(p) - ⟨X_k, \log_{p_k}p⟩
```
for a point `p_k` and a tangent vector `X_k` at `p_k` (for example outer iterates)
that are stored within this functor as well.

# Fields

* `g` a function
* `pk` a point on a manifold
* `Xk` a tangent vector at `pk`

Both interim values can be set using
`set_parameter!(::LinearizedDCCost, ::Val{:p}, p)`
and `set_parameter!(::LinearizedDCCost, ::Val{:X}, X)`, respectively.

# Constructor
    LinearizedDCCost(g, p, X)
"""
mutable struct LinearizedDCCost{P,T,TG}
    g::TG
    pk::P
    Xk::T
end
(F::LinearizedDCCost)(M, p) = F.g(M, p) - inner(M, F.pk, F.Xk, log(M, F.pk, p))

function set_parameter!(ldc::LinearizedDCCost, ::Val{:p}, p)
    ldc.pk .= p
    return ldc
end
function set_parameter!(ldc::LinearizedDCCost, ::Val{:X}, X)
    ldc.Xk .= X
    return ldc
end

@doc raw"""
    LinearizedDCGrad

A functor `(M,X,p) → ℝ` to represent the gradient of the inner problem of a [`ManifoldDifferenceOfConvexObjective`](@ref).
This is a gradient function of the form

```math
    F_{p_k,X_k}(p) = g(p) - ⟨X_k, \log_{p_k}p⟩
```

its gradient is given by using ``F=F_1(F_2(p))``, where ``F_1(X) = ⟨X_k,X⟩`` and ``F_2(p) = \log_{p_k}p``
and the chain rule as well as the adjoint differential of the logarithmic map with respect to its argument for ``D^*F_2(p)``

```math
    \operatorname{grad} F(q) = \operatorname{grad} f(q) - DF_2^*(q)[X]
```

for a point `pk` and a tangent vector `Xk` at `pk` (the outer iterates) that are stored within this functor as well

# Fields

* `grad_g!!` the gradient of ``g`` (see also [`LinearizedDCCost`](@ref))
* `pk` a point on a manifold
* `Xk` a tangent vector at `pk`

Both interim values can be set using
`set_parameter!(::LinearizedDCGrad, ::Val{:p}, p)`
and `set_parameter!(::LinearizedDCGrad, ::Val{:X}, X)`, respectively.

# Constructor
    LinearizedDCGrad(grad_g, p, X; evaluation=AllocatingEvaluation())

Where you specify whether `grad_g` is [`AllocatingEvaluation`](@ref) or [`InplaceEvaluation`](@ref),
while this function still provides _both_ signatures.
"""
mutable struct LinearizedDCGrad{E<:AbstractEvaluationType,P,T,TG}
    grad_g!!::TG
    pk::P
    Xk::T
    function LinearizedDCGrad(
        grad_g::TG, pk::P, Xk::T; evaluation::E=AllocatingEvaluation()
    ) where {TG,P,T,E<:AbstractEvaluationType}
        return new{E,P,T,TG}(grad_g, pk, Xk)
    end
end
function (grad_f::LinearizedDCGrad{AllocatingEvaluation})(M, p)
    return grad_f.grad_g!!(M, p) .-
           adjoint_differential_log_argument(M, grad_f.pk, p, grad_f.Xk)
end
function (grad_f::LinearizedDCGrad{AllocatingEvaluation})(M, X, p)
    copyto!(M, X, p, grad_f(M, p))
    return X
end
function (grad_f!::LinearizedDCGrad{InplaceEvaluation})(M, X, p)
    grad_f!.grad_g!!(M, X, p)
    X .-= adjoint_differential_log_argument(M, grad_f!.pk, p, grad_f!.Xk)
    return X
end
function (grad_f!::LinearizedDCGrad{InplaceEvaluation})(M, p)
    X = zero_vector(M, p)
    grad_f!.grad_g!!(M, X, p)
    X .-= adjoint_differential_log_argument(M, grad_f!.pk, p, grad_f!.Xk)
    return X
end

function set_parameter!(ldcg::LinearizedDCGrad, ::Val{:p}, p)
    ldcg.pk .= p
    return ldcg
end
function set_parameter!(ldcg::LinearizedDCGrad, ::Val{:X}, X)
    ldcg.Xk .= X
    return ldcg
end

#
# Difference of Convex Proximal Algorithm plan
#
@doc raw"""
    ManifoldDifferenceOfConvexProximalObjective{E} <: Problem

Specify an objective [`difference_of_convex_proximal_point`](@ref) algorithm.
The problem is of the form

```math
    \operatorname*{argmin}_{p∈\mathcal M} g(p) - h(p)
```

where both ``g`` and ``h`` are convex, lower semicontinuous and proper.

# Fields

* `cost`:     implementation of ``f(p) = g(p)-h(p)``
* `gradient`: the gradient of the cost
* `grad_h!!`: a function ``\operatorname{grad}h: \mathcal M → T\mathcal M``,

Note that both the gradients might be given in two possible signatures
as allocating or in-place.

 # Constructor

    ManifoldDifferenceOfConvexProximalObjective(gradh; cost=nothing, gradient=nothing)

an note that neither cost nor gradient are required for the algorithm,
just for eventual debug or stopping criteria.
"""
struct ManifoldDifferenceOfConvexProximalObjective{E<:AbstractEvaluationType,GH,F,G} <:
       AbstractManifoldFirstOrderObjective{E,Tuple{F,G}}
    cost::F
    gradient!!::G
    grad_h!!::GH
    function ManifoldDifferenceOfConvexProximalObjective(
        grad_h::THG;
        cost::TC=nothing,
        gradient::TG=nothing,
        evaluation::ET=AllocatingEvaluation(),
    ) where {ET<:AbstractEvaluationType,TC,TG,THG}
        return new{ET,THG,TC,TG}(cost, gradient, grad_h)
    end
end

function get_gradient(
    M::AbstractManifold,
    dcpo::ManifoldDifferenceOfConvexProximalObjective{AllocatingEvaluation},
    p,
)
    return dcpo.gradient!!(M, p)
end
function get_gradient(
    M::AbstractManifold,
    dcpo::ManifoldDifferenceOfConvexProximalObjective{InplaceEvaluation},
    p,
)
    X = zero_vector(M, p)
    return dcpo.gradient!!(M, X, p)
end
function get_gradient!(
    M::AbstractManifold,
    X,
    dcpo::ManifoldDifferenceOfConvexProximalObjective{AllocatingEvaluation},
    p,
)
    return copyto!(M, X, p, dcpo.gradient!!(M, p))
end
function get_gradient!(
    M::AbstractManifold,
    X,
    dcpo::ManifoldDifferenceOfConvexProximalObjective{InplaceEvaluation},
    p,
)
    return dcpo.gradient!!(M, X, p)
end

function get_gradient_function(
    dcpo::ManifoldDifferenceOfConvexProximalObjective, recursive=false
)
    return dcpo.gradient!!
end

@doc raw"""
    X = get_subtrahend_gradient(M::AbstractManifold, dcpo::ManifoldDifferenceOfConvexProximalObjective, p)
    get_subtrahend_gradient!(M::AbstractManifold, X, dcpo::ManifoldDifferenceOfConvexProximalObjective, p)

Evaluate the gradient of the subtrahend ``h`` from within
a [`ManifoldDifferenceOfConvexProximalObjective`](@ref)` `P` at the point `p` (in place of X).
"""
get_subtrahend_gradient(
    M::AbstractManifold, dcpo::ManifoldDifferenceOfConvexProximalObjective, p
)

function get_subtrahend_gradient(
    M::AbstractManifold,
    dcpo::ManifoldDifferenceOfConvexProximalObjective{AllocatingEvaluation},
    p,
)
    return dcpo.grad_h!!(M, p)
end
function get_subtrahend_gradient(
    M::AbstractManifold,
    dcpo::ManifoldDifferenceOfConvexProximalObjective{InplaceEvaluation},
    p,
)
    X = zero_vector(M, p)
    dcpo.grad_h!!(M, X, p)
    return X
end
function get_subtrahend_gradient!(
    M::AbstractManifold,
    X,
    dcpo::ManifoldDifferenceOfConvexProximalObjective{AllocatingEvaluation},
    p,
)
    return copyto!(M, X, p, dcpo.grad_h!!(M, p))
end
function get_subtrahend_gradient!(
    M::AbstractManifold,
    X,
    dcpo::ManifoldDifferenceOfConvexProximalObjective{InplaceEvaluation},
    p,
)
    dcpo.grad_h!!(M, X, p)
    return X
end

@doc raw"""
    ProximalDCCost

A functor `(M, p) → ℝ` to represent the inner cost function of a [`ManifoldDifferenceOfConvexProximalObjective`](@ref).
This is the cost function of the proximal map of `g`.

```math
    F_{p_k}(p) = \frac{1}{2λ}d_{\mathcal M}(p_k,p)^2 + g(p)
```

for a point `pk` and a proximal parameter ``λ``.

# Fields

* `g`  - a function
* `pk` - a point on a manifold
* `λ`  - the prox parameter

Both interim values can be set using
`set_parameter!(::ProximalDCCost, ::Val{:p}, p)`
and `set_parameter!(::ProximalDCCost, ::Val{:λ}, λ)`, respectively.

# Constructor

    ProximalDCCost(g, p, λ)
"""
mutable struct ProximalDCCost{P,TG,R}
    g::TG
    pk::P
    λ::R
end
(F::ProximalDCCost)(M, p) = 1 / (2 * F.λ) * distance(M, p, F.pk)^2 + F.g(M, p)

function set_parameter!(pdcc::ProximalDCCost, ::Val{:p}, p)
    pdcc.pk .= p
    return pdcc
end
function set_parameter!(pdcc::ProximalDCCost, ::Val{:λ}, λ)
    pdcc.λ = λ
    return pdcc
end

@doc raw"""
    ProximalDCGrad

A functor `(M,X,p) → ℝ` to represent the gradient of the inner cost function of a [`ManifoldDifferenceOfConvexProximalObjective`](@ref).
This is the gradient function of the proximal map cost function of `g`. Based on

```math
    F_{p_k}(p) = \frac{1}{2λ}d_{\mathcal M}(p_k,p)^2 + g(p)
```

it reads

```math
    \operatorname{grad} F_{p_k}(p) = \operatorname{grad} g(p) - \frac{1}{λ}\log_p p_k
```

for a point `pk` and a proximal parameter `λ`.

# Fields

* `grad_g`  - a gradient function
* `pk` - a point on a manifold
* `λ`  - the prox parameter

Both interim values can be set using
`set_parameter!(::ProximalDCGrad, ::Val{:p}, p)`
and `set_parameter!(::ProximalDCGrad, ::Val{:λ}, λ)`, respectively.


# Constructor
    ProximalDCGrad(grad_g, pk, λ; evaluation=AllocatingEvaluation())

Where you specify whether `grad_g` is [`AllocatingEvaluation`](@ref) or [`InplaceEvaluation`](@ref),
while this function still always provides _both_ signatures.
"""
mutable struct ProximalDCGrad{E<:AbstractEvaluationType,P,TG,R}
    grad_g!!::TG
    pk::P
    λ::R
    function ProximalDCGrad(
        grad_g::TG, pk::P, λ::R; evaluation::E=AllocatingEvaluation()
    ) where {TG,P,R,E<:AbstractEvaluationType}
        return new{E,P,TG,R}(grad_g, pk, λ)
    end
end
function (grad_f::ProximalDCGrad{AllocatingEvaluation})(M, p)
    return grad_f.grad_g!!(M, p) - 1 / grad_f.λ * log(M, p, grad_f.pk)
end
function (grad_f::ProximalDCGrad{AllocatingEvaluation})(M, X, p)
    copyto!(M, X, p, grad_f.grad_g!!(M, p) - 1 / grad_f.λ * log(M, p, grad_f.pk))
    return X
end
function (grad_f!::ProximalDCGrad{InplaceEvaluation})(M, X, p)
    grad_f!.grad_g!!(M, X, p)
    X .-= 1 / grad_f!.λ * log(M, p, grad_f!.pk)
    return X
end
function (grad_f!::ProximalDCGrad{InplaceEvaluation})(M, p)
    X = zero_vector(M, p)
    grad_f!.grad_g!!(M, X, p)
    X .-= 1 / grad_f!.λ * log(M, p, grad_f!.pk)
    return X
end
function set_parameter!(pdcg::ProximalDCGrad, ::Val{:p}, p)
    pdcg.pk .= p
    return pdcg
end
function set_parameter!(pdcg::ProximalDCGrad, ::Val{:λ}, λ)
    pdcg.λ = λ
    return pdcg
end
