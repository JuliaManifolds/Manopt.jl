@doc raw"""
    ManifoldStochasticGradientObjective{T<:AbstractEvaluationType} <: AbstractManifoldGradientObjective{T}

A stochastic gradient objective consists of

* a(n optional) cost function ``f(p) = \displaystyle\sum_{i=1}^n f_i(p)
* an array of gradients, ``\operatorname{grad}f_i(p), i=1,\ldots,n`` which can be given in two forms
  * as one single function ``(\mathcal M, p) ↦ (X_1,…,X_n) \in (T_p\mathcal M)^n``
  * as a vector of functions ``\bigl( (\mathcal M, p) ↦ X_1, …, (\mathcal M, p) ↦ X_n\bigr)``.

Where both variants can also be provided as [`InplaceEvaluation`](@ref) functions, i.e.
`(M, X, p) -> X`, where `X` is the vector of `X1,...Xn` and `(M, X1, p) -> X1, ..., (M, Xn, p) -> Xn`,
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

Create a Stochastic gradient problem with an optional `cost` and the gradient either as one
function (returning an array of tangent vectors) or a vector of functions (each returning one tangent vector).

# Used with
[`stochastic_gradient_descent`](@ref)

Note that this can also be used with a [`gradient_decent`](@ref), since the (complete) gradient
is just the sums of the single gradients.
"""
struct ManifoldStochasticGradientObjective{T<:AbstractEvaluationType,TCost,TGradient} <:
       AbstractManifoldGradientObjective{T}
    cost::TCost
    gradient!!::TGradient
end
function ManifoldStochasticGradientObjective(
    grad_f!!;
    cost::Union{Function,Missing}=Missing(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
)
    return ManifoldStochasticGradientObjective{
        typeof(evaluation),typeof(cost),typeof(grad_f!!)
    }(
        cost, grad_f!!
    )
end
function ManifoldStochasticGradientObjective(
    grad_f!!::AbstractVector{<:Function};
    cost::Union{Function,Missing}=Missing(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
)
    return ManifoldStochasticGradientObjective{
        typeof(evaluation),typeof(cost),typeof(grad_f!!)
    }(
        cost, grad_f!!
    )
end

@doc raw"""
    get_gradients(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p)
    get_gradients!(M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective, p)

Evaluate all summands gradients ``\{\operatorname{grad}f_i\}_{i=1}^n`` at `p` (in place of `X`).

Note that for the [`InplaceEvaluation`](@ref) based problem and a single function for the
stochastic gradient, the allocating variant is not available, since the number of
tangent vectors can not be determined in this case..
"""
function get_gradients(
    M::AbstractManifold,
    sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation,TC,<:Function},
    p,
) where {TC}
    return sgo.gradient!!(M, p)
end
function get_gradients(
    M::AbstractManifold,
    sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation,TC,<:AbstractVector},
    p,
) where {TC}
    return [grad_i(M, p) for grad_i in sgo.gradient!!]
end
function get_gradients!(
    M::AbstractManifold,
    X,
    sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation,TC,<:Function},
    p,
) where {TC}
    copyto!(M, X, sgo.gradient!!(M, p))
    return X
end
function get_gradients!(
    M::AbstractManifold,
    X,
    sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation,TC,<:AbstractVector},
    p,
) where {TC}
    copyto!.(Ref(M), X, [grad_i(M, p) for grad_i in sgo.gradient!!])
    return X
end
function get_gradients(
    ::AbstractManifold,
    ::ManifoldStochasticGradientObjective{InplaceEvaluation,TC,<:Function},
    ::Any,
) where {TC}
    return error(
        "For a mutating function type stochastic gradient, the allocating variant is not possible.",
    )
end
function get_gradients(
    M::AbstractManifold,
    sgo::ManifoldStochasticGradientObjective{InplaceEvaluation,TC,<:AbstractVector},
    p,
) where {TC}
    X = [zero_vector(M, p) for _ in sgo.gradient!!]
    get_gradients!(M, X, sgo, p)
    return X
end
function get_gradients!(
    M::AbstractManifold,
    X,
    sgo::ManifoldStochasticGradientObjective{InplaceEvaluation,TC,<:Function},
    p,
) where {TC}
    sgo.gradient!!(M, X, p)
    return X
end
function get_gradients!(
    M::AbstractManifold,
    X,
    sgo::ManifoldStochasticGradientObjective{InplaceEvaluation,TC,<:AbstractVector},
    p,
) where {TC}
    for (Xi, grad_i) in zip(X, sgo.gradient!!)
        grad_i(M, Xi, p)
    end
    return X
end
# Passdown from problem
function get_gradients(mp::AbstractManoptProblem, p)
    return get_gradients(get_manifold(mp), get_objective(mp), p)
end
function get_gradients!(mp::AbstractManoptProblem, X, p)
    return get_gradients!(get_manifold(mp), X, get_objective(mp), p)
end

@doc raw"""
    get_gradient(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p, k)
    get_gradient!(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, Y, p, k)

Evaluate one of the summands gradients ``\operatorname{grad}f_k``, ``k∈\{1,…,n\}``, at `x` (in place of `Y`).

Note that for the [`InplaceEvaluation`](@ref) based problem and a single function for the
stochastic gradient, it is not possible to derive the number `n`, and it would also require
`n` allocations`.
"""
function get_gradient(
    M::AbstractManifold,
    sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation,TC,<:Function},
    p,
    k,
) where {TC}
    return sgo.gradient!!(M, p)[k]
end
function get_gradient(
    M::AbstractManifold,
    sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation,TC,<:AbstractVector},
    p,
    k,
) where {TC}
    return sgo.gradient!![k](M, p)
end
function get_gradient(
    M::AbstractManifold,
    sgo::ManifoldStochasticGradientObjective{InplaceEvaluation,TC},
    p,
    k,
) where {TC}
    X = zero_vector(M, p)
    return get_gradient!(M, X, sgo, p, k)
end
function get_gradient!(
    M::AbstractManifold,
    X,
    sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation,TC,<:Function},
    p,
    k,
) where {TC}
    copyto!(M, X, sgo.gradient!!(M, p)[k])
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation,TC,<:AbstractVector},
    p,
    k,
) where {TC}
    copyto!(M, X, sgo.gradient!![k](M, p))
    return X
end
function get_gradient!(
    ::AbstractManifold,
    ::Any,
    ::ManifoldStochasticGradientObjective{InplaceEvaluation,TC,<:Function},
    ::Any,
    ::Any,
) where {TC}
    return error(
        "A mutating variant of the stochastic gradient as a single function is not implemented.",
    )
end
function get_gradient!(
    M::AbstractManifold,
    X,
    sgo::ManifoldStochasticGradientObjective{InplaceEvaluation,TC,<:AbstractVector},
    p,
    k,
) where {TC}
    return sgo.gradient!![k](M, X, p)
end
# Passdown from problem
function get_gradient(mp::AbstractManoptProblem, p, k)
    return get_gradient(get_manifold(mp), get_objective(mp), p, k)
end
function get_gradient!(mp::AbstractManoptProblem, X, p, k)
    return get_gradient!(get_manifold(mp), X, get_objective(mp), p, k)
end

@doc raw"""
    get_gradient(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, p)
    get_gradient!(M::AbstractManifold, sgo::ManifoldStochasticGradientObjective, X, p)

Evaluate the complete gradient ``\operatorname{grad} f = \displaystyle\sum_{i=1}^n \operatorname{grad} f_i(p)`` at `p` (in place of `X`).

Note that for the [`InplaceEvaluation`](@ref) based problem and a single function for the
stochastic gradient, it is not possible to derive the number `n`, and it would also require
`n` allocations`.
"""
function get_gradient(
    M::AbstractManifold, sgo::ManifoldStochasticGradientObjective{T,TC,<:Function}, p
) where {T<:AbstractEvaluationType,TC}
    # even if the function is in-place, we would need to allocate the full vector of tangent vectors
    return sum(get_gradients(M, sgo, p))
end
function get_gradient!(
    M::AbstractManifold, X, sgo::ManifoldStochasticGradientObjective{T,TC,<:Function}, p
) where {T<:AbstractEvaluationType,TC}
    zero_vector!(M, X, p)
    for Xi in sgo.gradient!!(M, p)
        X += Xi
    end
    return X
end
function get_gradient(
    M::AbstractManifold,
    sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation,TC,<:AbstractVector},
    p,
) where {TC}
    X = zero_vector(M, p)
    get_gradient!(M, X, sgo, p)
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    sgo::ManifoldStochasticGradientObjective{AllocatingEvaluation,TC,<:AbstractVector},
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
    sgo::ManifoldStochasticGradientObjective{InplaceEvaluation,TC,<:AbstractVector},
    p,
) where {TC}
    X = zero_vector(M, p)
    get_gradient!(M, X, sgo, p)
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    sgo::ManifoldStochasticGradientObjective{InplaceEvaluation,TC,<:AbstractVector},
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

A generic type for all options related to stochastic gradient descent methods
"""
abstract type AbstractGradientGroupProcessor <: DirectionUpdateRule end

"""
    StochasticGradientDescentState <: AbstractGradientDescentSolverState

Store the following fields for a default stochastic gradient descent algorithm,
see also [`ManifoldStochasticGradientObjective`](@ref) and [`stochastic_gradient_descent`](@ref).

# Fields

* `x` the current iterate
* `direction` ([`StochasticGradient`](@ref)) a direction update to use
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ConstantStepsize`](@ref)`(1.0)`) a [`Stepsize`](@ref)
* `evaluation_order` – (`:Random`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Linear`) or the default `:Random` one.
* `order` the current permutation
* `retraction_method` – (`default_retraction_method(M)`) a `retraction(M,x,ξ)` to use.

# Constructor
    StochasticGradientDescentState(M, x)

Create a `StochasticGradientDescentState` with start point `x`.
all other fields are optional keyword arguments, and the defaults are taken from `M`.
"""
mutable struct StochasticGradientDescentState{
    TX,
    TV,
    D<:DirectionUpdateRule,
    TStop<:StoppingCriterion,
    TStep<:Stepsize,
    RM<:AbstractRetractionMethod,
} <: AbstractGradientSolverState
    p::TX
    X::TV
    direction::D
    stop::TStop
    stepsize::TStep
    order_type::Symbol
    order::Vector{<:Int}
    retraction_method::RM
    k::Int # current iterate
end

function StochasticGradientDescentState(
    M::AbstractManifold,
    p::P,
    X::Q;
    direction::D=StochasticGradient(zero_vector(M, p)),
    order_type::Symbol=:RandomOrder,
    order::Vector{<:Int}=Int[],
    retraction_method::RM=default_retraction_method(M),
    stopping_criterion::SC=StopAfterIteration(1000),
    stepsize::S=ConstantStepsize(M),
) where {
    P,
    Q,
    D<:DirectionUpdateRule,
    RM<:AbstractRetractionMethod,
    SC<:StoppingCriterion,
    S<:Stepsize,
}
    return StochasticGradientDescentState{P,Q,D,SC,S,RM}(
        p,
        X,
        direction,
        stopping_criterion,
        stepsize,
        order_type,
        order,
        retraction_method,
        0,
    )
end

"""
    StochasticGradient <: DirectionUpdateRule

The default gradient processor, which just evaluates the (stochastic) gradient or a subset
thereof.
"""
struct StochasticGradient{T} <: AbstractGradientGroupProcessor
    dir::T
end
function StochasticGradient(M::AbstractManifold; p=random_point(M), X=zero_vector(M, p))
    return StochasticGradient{typeof(X)}(X)
end

function (sg::StochasticGradient)(
    apm::AbstractManoptProblem, sgds::StochasticGradientDescentState, iter
)
    # for each new epoche choose new order if we are at random order
    ((sgds.k == 1) && (sgds.order_type == :Random)) && shuffle!(sgds.order)
    # i is the gradient to choose, either from the order or completely random
    j = sgds.order_type == :Random ? rand(1:length(sgds.order)) : sgds.order[sgds.k]
    return sgds.stepsize(apm, sgds, iter), get_gradient!(apm, sg.dir, sgds.p, j)
end
