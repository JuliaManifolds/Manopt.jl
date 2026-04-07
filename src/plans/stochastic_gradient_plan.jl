@doc """
    ManifoldStochasticGradientObjective{E<:AbstractEvaluationType, F, G} <: AbstractManifoldFirstOrderObjective{E, Tuple{F,G}}

A stochastic gradient objective consists of

* a(n optional) cost function ``f(p) = $(_tex(:displaystyle))$(_tex(:sum, "i=1", "n")) f_i(p)``
* an array of gradients, ``$(_tex(:grad)) f_i(p), i=1,…,n`` which can be given in two forms
  * as one single function ``($(_math(:Manifold))nifold))nifold))), p) ↦ (X_1,…,X_n) ∈ ($(_math(:TangentSpace))n``
  * as a vector of functions ``$(_tex(:bigl))( ($(_math(:Manifold))), p) ↦ X_1, …, ($(_math(:Manifold))), p) ↦ X_n$(_tex(:bigr)))``.

Where both variants can also be provided as [`InplaceEvaluation`](@ref) functions
`(M, X, p) -> X`, where `X` is the vector of `X1,...,Xn` and `(M, X1, p) -> X1, ..., (M, Xn, p) -> Xn`,
respectively.

# Constructors

    ManifoldStochasticGradientObjective(
        grad_f::Function;
        cost=Missing(),
        evaluation=AllocatingEvaluation()
    )
    ManifoldStochasticGradientObjective(
        grad_f::AbstractVector{<:Function};
        cost=Missing(), evaluation=AllocatingEvaluation()
    )

Create a Stochastic gradient problem with the gradient either as one
function (returning an array of tangent vectors) or a vector of functions (each returning one tangent vector).

The optional cost can also be given as either a single function (returning a number)
pr a vector of functions, each returning a value.

# Used with
[`stochastic_gradient_descent`](@ref)

Note that this can also be used with a [`gradient_descent`](@ref), since the (complete) gradient
is just the sums of the single gradients.
"""
struct ManifoldStochasticGradientObjective{T <: AbstractEvaluationType, TCost, TGradient} <:
    AbstractManifoldFirstOrderObjective{T, Tuple{TCost, TGradient}}
    cost::TCost
    gradient!!::TGradient
end
function ManifoldStochasticGradientObjective(
        grad_f!!::G; cost::C = Missing(), evaluation::E = AllocatingEvaluation()
    ) where {
        E <: AbstractEvaluationType,
        G <: Union{Function, AbstractVector{<:Function}},
        C <: Union{Function, AbstractVector{<:Function}, Missing},
    }
    return ManifoldStochasticGradientObjective{E, C, G}(cost, grad_f!!)
end

function get_cost(
        M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{E, C}, p
    ) where {E <: AbstractEvaluationType, C <: AbstractVector{<:Function}}
    return sum(f(M, p) for f in sgo.cost)
end

@doc """
    get_cost(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p, i)

Evaluate the `i`th summand of the cost.

If you use a single function for the stochastic cost, then only the index `ì=1`` is available
to evaluate the whole cost.
"""
function get_cost(
        M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{E, C}, p, i
    ) where {E <: AbstractEvaluationType, C <: AbstractVector{<:Function}}
    return sgo.cost[i](M, p)
end
function get_cost(
        M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{E, C}, p, i
    ) where {E <: AbstractEvaluationType, C <: Function}
    (i == 1) && return sgo.cost(M, p)
    return error(
        "The cost is implemented as a single function and can not be accessed element wise at $i since the index is larger than 1.",
    )
end

@doc """
    get_gradients(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p)
    get_gradients!(M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective, p)

Evaluate all summands gradients ``$(_math(:Sequence, "$(_tex(:grad))f", "i", "1", "n")) at `p` (in place of `X`).

If you use a single function for the stochastic gradient, that works in-place, then [`get_gradient`](@ref) is not available,
since the length (or number of elements of the gradient) can not be determined.
"""
function get_gradients(
        M::AbstractManifold,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:Function},
        p,
    ) where {TC}
    return sgo.gradient!!(M, p)
end
function get_gradients(
        M::AbstractManifold,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    return [grad_i(M, p) for grad_i in sgo.gradient!!]
end
function get_gradients(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradients(M, get_objective(admo, false), p)
end

function get_gradients!(
        M::AbstractManifold,
        X,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:Function},
        p,
    ) where {TC}
    copyto!(M, X, sgo.gradient!!(M, p))
    return X
end
function get_gradients!(
        M::AbstractManifold,
        X,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    copyto!.(Ref(M), X, [grad_i(M, p) for grad_i in sgo.gradient!!])
    return X
end
function get_gradients!(M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p)
    return get_gradients!(M, X, get_objective(admo, false), p)
end

function get_gradients(
        ::AbstractManifold,
        ::ManifoldStochasticGradientObjective{InplaceEvaluation, TC, <:Function},
        ::Any,
    ) where {TC}
    return error(
        "For a mutating function type stochastic gradient, the allocating variant is not possible.",
    )
end
function get_gradients(
        M::AbstractManifold,
        sgo::ManifoldStochasticGradientObjective{InplaceEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    X = [zero_vector(M, p) for _ in sgo.gradient!!]
    get_gradients!(M, X, sgo, p)
    return X
end
function get_gradients!(
        M::AbstractManifold,
        X,
        sgo::ManifoldStochasticGradientObjective{InplaceEvaluation, TC, <:Function},
        p,
    ) where {TC}
    sgo.gradient!!(M, X, p)
    return X
end
function get_gradients!(
        M::AbstractManifold,
        X,
        sgo::ManifoldStochasticGradientObjective{InplaceEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    for (Xi, grad_i) in zip(X, sgo.gradient!!)
        grad_i(M, Xi, p)
    end
    return X
end
# Pass down from problem
function get_gradients(mp::AbstractManoptProblem, p)
    return get_gradients(get_manifold(mp), get_objective(mp), p)
end
function get_gradients!(mp::AbstractManoptProblem, X, p)
    return get_gradients!(get_manifold(mp), X, get_objective(mp), p)
end

@doc """
    get_gradient(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p, k)
    get_gradient!(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, Y, p, k)

Evaluate one of the summands gradients ``$(_tex(:grad))f_k``, ``k ∈ $(_tex(:set, "1,…,n"))``, at `p` (in place of `Y`).

If you use a single function for the stochastic gradient, that works in-place, then [`get_gradient`](@ref) is not available,
since the length (or number of elements of the gradient required for allocation) can not be determined.
"""
function get_gradient(
        M::AbstractManifold,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:Function},
        p,
        k,
    ) where {TC}
    return sgo.gradient!!(M, p)[k]
end
function get_gradient(
        M::AbstractManifold,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:AbstractVector},
        p,
        k,
    ) where {TC}
    return sgo.gradient!![k](M, p)
end
function get_gradient(
        M::AbstractManifold,
        sgo::ManifoldStochasticGradientObjective{InplaceEvaluation, TC},
        p,
        k,
    ) where {TC}
    X = zero_vector(M, p)
    return get_gradient!(M, X, sgo, p, k)
end
function get_gradient(M::AbstractManifold, admo::AbstractDecoratedManifoldObjective, p, k)
    return get_gradient(M, get_objective(admo, false), p, k)
end

function get_gradient!(
        M::AbstractManifold,
        X,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:Function},
        p,
        k,
    ) where {TC}
    copyto!(M, X, sgo.gradient!!(M, p)[k])
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:AbstractVector},
        p,
        k,
    ) where {TC}
    copyto!(M, X, sgo.gradient!![k](M, p))
    return X
end
function get_gradient!(
        ::AbstractManifold,
        ::Any,
        ::ManifoldStochasticGradientObjective{InplaceEvaluation, TC, <:Function},
        ::Any,
        ::Any,
    ) where {TC}
    return error(
        "An in-place variant for single entries of the stochastic gradient as a single function is not implemented, since the size can not be determined.",
    )
end
function get_gradient!(
        M::AbstractManifold,
        X,
        sgo::ManifoldStochasticGradientObjective{InplaceEvaluation, TC, <:AbstractVector},
        p,
        k,
    ) where {TC}
    return sgo.gradient!![k](M, X, p)
end
function get_gradient!(
        M::AbstractManifold, X, admo::AbstractDecoratedManifoldObjective, p, k
    )
    return get_gradient!(M, X, get_objective(admo, false), p, k)
end

# Pass down from problem
function get_gradient(mp::AbstractManoptProblem, p, k)
    return get_gradient(get_manifold(mp), get_objective(mp), p, k)
end
function get_gradient!(mp::AbstractManoptProblem, X, p, k)
    return get_gradient!(get_manifold(mp), X, get_objective(mp), p, k)
end

@doc """
    get_gradient(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p)
    get_gradient!(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, X, p)

Evaluate the complete gradient ``$(_tex(:grad)) f = $(_tex(:displaystyle))$(_tex(:sum, "i=1", "n")) $(_tex(:grad)) f_i(p)`` at `p` (in place of `X`).

If you use a single function for the stochastic gradient, that works in-place, then [`get_gradient`](@ref) is not available,
since the length (or number of elements of the gradient required for allocation) can not be determined.
"""
function get_gradient(
        M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{T, TC, <:Function}, p
    ) where {T <: AbstractEvaluationType, TC}
    # even if the function is in-place, allocation of the full vector of tangent vectors still required
    return sum(get_gradients(M, sgo, p))
end
function get_gradient!(
        M::AbstractManifold,
        X,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:Function},
        p,
    ) where {TC}
    zero_vector!(M, X, p)
    for Xi in sgo.gradient!!(M, p)
        X += Xi
    end
    return X
end
function get_gradient!(
        ::AbstractManifold,
        ::Any,
        ::ManifoldStochasticGradientObjective{InplaceEvaluation, TC, <:Function},
        ::Any,
    ) where {TC}
    return error(
        "An in-place variant for (sum of) the stochastic gradient as a single function is not implemented, since the size can not be determined.",
    )
end
function get_gradient(
        M::AbstractManifold,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    X = zero_vector(M, p)
    get_gradient!(M, X, sgo, p)
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    zero_vector!(M, X, p)
    for k in 1:length(sgo.gradient!!)
        X += get_gradient(M, sgo, p, k)
    end
    return X
end
function get_gradient(
        M::AbstractManifold,
        sgo::ManifoldStochasticGradientObjective{InplaceEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    X = zero_vector(M, p)
    get_gradient!(M, X, sgo, p)
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        sgo::ManifoldStochasticGradientObjective{InplaceEvaluation, TC, <:AbstractVector},
        p,
    ) where {TC}
    zero_vector!(M, X, p)
    Y = copy(M, p, X)
    for grad_i in sgo.gradient!!
        grad_i(M, Y, p)
        X += Y
    end
    return X
end

"""
    AbstractStochasticGradientDescentSolverState <: AbstractManoptSolverState

A generic type for all options related to gradient descent methods working with parts of the total gradient
"""
abstract type AbstractGradientGroupDirectionRule <: DirectionUpdateRule end
