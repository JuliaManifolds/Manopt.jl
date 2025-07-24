@doc raw"""
    ManifoldAlternatingGradientObjective{E<:AbstractEvaluationType,F,G} <: AbstractManifoldFirstOrderObjective{E, Tuple{F,G}}

An alternating gradient objective consists of

* a cost function ``F(x)``
* a gradient ``\operatorname{grad}F`` that is either
  * given as one function ``\operatorname{grad}F`` returning a tangent vector `X` on `M` or
  * an array of gradient functions ``\operatorname{grad}F_i``, `ì=1,…,n` s each returning a component of the gradient
  which might be allocating or mutating variants, but not a mix of both.

!!! note

    This Objective is usually defined using the `ProductManifold` from `Manifolds.jl`, so `Manifolds.jl` to be loaded.

# Constructors

    ManifoldAlternatingGradientObjective(F, gradF::Function;
        evaluation=AllocatingEvaluation()
    )
    ManifoldAlternatingGradientObjective(F, gradF::AbstractVector{<:Function};
        evaluation=AllocatingEvaluation()
    )

Create a alternating gradient problem with an optional `cost` and the gradient either as one
function (returning an array) or a vector of functions.
"""
struct ManifoldAlternatingGradientObjective{E <: AbstractEvaluationType, F, G} <:
    AbstractManifoldFirstOrderObjective{E, Tuple{F, G}}
    cost::F
    gradient!!::G
end
function ManifoldAlternatingGradientObjective(
        f::TCost, grad_f::G; evaluation::AbstractEvaluationType = AllocatingEvaluation()
    ) where {G, TCost}
    return ManifoldAlternatingGradientObjective{typeof(evaluation), TCost, G}(f, grad_f)
end
function ManifoldAlternatingGradientObjective(
        f::TCost,
        grad_f::AbstractVector{<:TG};
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
    ) where {TCost, TG}
    return ManifoldAlternatingGradientObjective{typeof(evaluation), TCost, typeof(grad_f)}(
        f, grad_f
    )
end

function get_gradient(
        M::AbstractManifold,
        mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation, TC, <:Function},
        p,
    ) where {TC}
    return mago.gradient!!(M, p)
end
function get_gradient!(
        M::AbstractManifold,
        X,
        mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation, TC, <:Function},
        p,
    ) where {TC}
    copyto!(M, X, p, get_gradient(M, mago, p))
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    copyto!(M, X, p, get_gradient(M, mago, p))
    return X
end
function get_gradient(
        M::AbstractManifold,
        mago::ManifoldAlternatingGradientObjective{InplaceEvaluation, TC, <:Function},
        p,
    ) where {TC}
    X = zero_vector(M, p)
    mago.gradient!!(M, X, p)
    return X
end
function get_gradient(
        M::AbstractManifold,
        mago::ManifoldAlternatingGradientObjective{InplaceEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    X = zero_vector(M, p)
    get_gradient!(M, X, mago, p)
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        mago::ManifoldAlternatingGradientObjective{InplaceEvaluation, TC, <:Function},
        p,
    ) where {TC}
    mago.gradient!!(M, X, p)
    return X
end

function get_gradient(
        M::AbstractManifold,
        mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation, TC, <:AbstractVector},
        p,
        k,
    ) where {TC}
    return mago.gradient!![k](M, p)
end
function get_gradient!(
        M::AbstractManifold,
        X,
        mago::ManifoldAlternatingGradientObjective{AllocatingEvaluation, TC, <:AbstractVector},
        p,
        k,
    ) where {TC}
    copyto!(M[k], X, mago.gradient!![k](M, p))
    return X
end
function get_gradient(
        M::AbstractManifold,
        mago::ManifoldAlternatingGradientObjective{InplaceEvaluation, TC},
        p,
        i,
    ) where {TC}
    X = zero_vector(M[i], p[M, i])
    get_gradient!(M, X, mago, p, i)
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        mago::ManifoldAlternatingGradientObjective{InplaceEvaluation, TC, <:Function},
        p,
        k,
    ) where {TC}
    # this takes a lot more allocations than other methods, but the gradient can only be evaluated in full
    Xf = zero_vector(M, p)
    get_gradient!(M, Xf, mago, p)
    copyto!(M[k], X, p[M, k], Xf[M, k])
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        mago::ManifoldAlternatingGradientObjective{InplaceEvaluation, TC, <:AbstractVector},
        p,
        k,
    ) where {TC}
    mago.gradient!![k](M, X, p)
    return X
end
