@doc raw"""
    ManifoldProximalGradientObjective{E,<:AbstractEvaluationType, TC, TG, TP} <: AbstractManifoldObjective{E,TC,TG}

Model an objective of the form
```math
    f(p) = g(p) + h(p),\qquad p \in \mathcal M,
```
where ``g: \mathcal M → \bar{ℝ}`` is differentiable
and ``h: → \bar{ℝ}`` is convex, lower semicontinous, and proper.

This objective also provides ``\operatorname{grad} g`` and ``\operatorname{prox}_{λ} h``.

# Fields

* `cost`: the overall cost ``f```
* `gradient_g`: the ``\operatorname{grad} g``
* `proximal_map_h` and ``\operatorname{prox}_{λ} h``

# Constructor
    ManifoldProximalGradientObjective(f, prox_g, prox_h;
        evalauation=[`AllocatingEvaluation`](@ref)
    )

Generate the proximal gradient objective given the cost `f`, the gradient of the smooth
component `grad_g`, and the proximal map of the nonsmooth component `prox_h`.

## Keyword arguments

* `evaluation=`[`AllocatingEvaluation`](@ref): whether the gradient and proximal map
  is given as an allocation function or an in-place ([`InplaceEvaluation`](@ref)).
"""
struct ManifoldProximalGradientObjective{E<:AbstractEvaluationType,TC,TG,TP} <:
       AbstractManifoldCostObjective{E,TC}
    cost::TC
    gradient_g!!::TG
    proximal_map_h!!::TP
    function ManifoldProximalGradientObjective(
        f::TF, grad_g::TG, prox_h::TP; evaluation::E=AllocatingEvaluation()
    ) where {TF,TG,TP,E<:AbstractEvaluationType}
        return new{E,TF,TG,TP}(f, grad_g, prox_h)
    end
end

"""
    get_gradient(M::AbstractManifold, mgo::ManifoldProximalGradientObjective, p)
    get_gradient!(M::AbstractManifold, X, mgo::ManifoldProximalGradientObjective, p)

evaluate the gradient of the smooth part of a [`ManifoldProximalGradientObjective`](@ref) `mgo` at `p`.
"""
get_gradient(M::AbstractManifold, mgo::ManifoldProximalGradientObjective, p)

function get_gradient(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation}, p
)
    return mpgo.gradient_g!!(M, p)
end
function get_gradient(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, p
)
    X = zero_vector(M, p)
    mpgo.gradient_g!!(M, X, p)
    return X
end

function get_gradient!(
    M::AbstractManifold, X, mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation}, p
)
    copyto!(M, X, p, mpgo.gradient_g!!(M, p))
    return X
end
function get_gradient!(
    M::AbstractManifold, X, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, p
)
    mpgo.gradient_g!!(M, X, p)
    return X
end

@doc raw"""
    q = get_proximal_map(M::AbstractManifold, mpo::ManifoldProximalGradientObjective, λ, p)
    get_proximal_map!(M::AbstractManifold, q, mpo::ManifoldProximalGradientObjective, λ, p)

evaluate proximal map of the nonsmooth component ``h`` of the [`ManifoldProximalGradientObjective`](@ref)` mpo`
at the point `p` on `M` with parameter ``λ>0``.
"""
get_proximal_map(::AbstractManifold, ::ManifoldProximalGradientObjective, ::Any...)

function get_proximal_map(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation}, λ, p
)
    return mpgo.proximal_map_h!!(M, λ, p)
end
function get_proximal_map!(
    M::AbstractManifold,
    q,
    mpgo::ManifoldProximalGradientObjective{AllocatingEvaluation},
    λ,
    p,
)
    copyto!(M, q, mpgo.proximal_map_h!!(M, λ, p))
    return q
end

function get_proximal_map(
    M::AbstractManifold, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, λ, p
)
    q = allocate_result(M, get_proximal_map, p)
    mpgo.proximal_map_h!!(M, q, λ, p)
    return q
end
function get_proximal_map!(
    M::AbstractManifold, q, mpgo::ManifoldProximalGradientObjective{InplaceEvaluation}, λ, p
)
    mpgo.proximal_map_h!!(M, q, λ, p)
    return q
end

