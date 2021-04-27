@doc raw"""
    AlternatingGradientProblem <: Problem

An alternating gradient problem consists of
* a `ProductManifold M` ``=\mathcal M = \mathcal M_1 × ⋯ × M_n``
* a cost function ``F(x)``
* a gradient ``\operatorname{grad}F`` that is either
  * given as one function ``\operatorname{grad}F`` returning a tangent vector `X` on `M` or
  * an array of gradient functions ``\operatorname{grad}F_i``, `ì=1,…,n` s each returning a component of the gradient
  which might be allocating or mutating variants, but not a mix of both.

!!! note

    This Problem requires the `ProductManifold` from `Manifolds.jl`, so `Manifolds.jl` to be loaded.

!!! note

    The input of each of the (component) gradients is still the whole vector `x`,
    just that all other then the `i`th input component are assumed to be fixed and just
    the `i`th components gradient is computed / returned.

# Constructors
    AlternatingGradientProblem(M::ProductManifold, F, gradF::Function;
        evaluation=AllocatingEvaluation()
    )
    AlternatingGradientProblem(M::ProductManifold, F, gradF::AbstractVector{<:Function};
        evaluation=AllocatingEvaluation()
    )

Create a alternating gradient problem with an optional `cost` and the gradient either as one
function (returning an array) or a vector of functions.
"""
struct AlternatingGradientProblem{T,MT<:ProductManifold,TCost,TGradient} <:
       AbstractGradientProblem{T}
    M::MT
    cost::TCost
    gradient!!::TGradient
end
function AlternatingGradientProblem(
    M::TM, F::TCost, gradF!!::G; evaluation::AbstractEvaluationType=AllocatingEvaluation()
) where {TM<:ProductManifold,G,TCost}
    return AlternatingGradientProblem{typeof(evaluation),TM,TCost,G}(M, F, gradF!!)
end
function AlternatingGradientProblem(
    M::TM,
    F::TCost,
    gradF!!::AbstractVector{<:TG};
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TM<:ProductManifold,TCost,TG}
    return AlternatingGradientProblem{typeof(evaluation),TM,TCost,typeof(gradF!!)}(
        M, F, gradF!!
    )
end

@doc raw"""
    get_gradient(P::AlternatingGradientProblem, x)
    get_gradient!(P::AlternatingGradientProblem, Y, x)

Evaluate all summands gradients at a point `x` on the `ProductManifold M` (in place of `Y`)
"""
function get_gradient(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:Function}, x
) where {TC}
    return p.gradient!!(p.M, x)
end
function get_gradient(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector}, x
) where {TC}
    Y = ProductRepr([gi(p.M, x) for gi in p.gradient!!]...)
    return Y
end
function get_gradient!(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:Function}, X, x
) where {TC}
    copyto!(p.M, X, x, get_gradient(p, x))
    return X
end
function get_gradient!(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector}, X, x
) where {TC}
    copyto!(p.M, X, x, get_gradient(p, x))
    return X
end
function get_gradient(
    p::AlternatingGradientProblem{MutatingEvaluation,<:Manifold,TC,<:Function}, x
) where {TC}
    Y = zero_tangent_vector(p.M, x)
    return p.gradient!!(p.M, Y, x)
end
function get_gradient(
    p::AlternatingGradientProblem{MutatingEvaluation,<:Manifold,TC,<:AbstractVector}, x
) where {TC}
    Y = zero_tangent_vector(p.M, x)
    get_gradient!(p, Y, x)
    return Y
end
function get_gradient!(
    p::AlternatingGradientProblem{MutatingEvaluation,<:Manifold,TC,<:Function}, X, x
) where {TC}
    return p.gradient!!(p.M, X, x)
end
function get_gradient!(
    p::AlternatingGradientProblem{MutatingEvaluation,<:Manifold,TC,<:AbstractVector}, X, x
) where {TC}
    for (gi, Xi) in zip(p.gradient!!, submanifold_components(p.M, X))
        gi(p.M, Xi, x)
    end
    return X
end

@doc raw"""
    get_gradient(p::AlternatingGradientProblem, k, x)
    get_gradient!(p::AlternatingGradientProblem, Y, k, x)

Evaluate one of the component gradients ``\operatorname{grad}f_k``, ``k∈\{1,…,n\}``, at `x` (in place of `Y`).
"""
function get_gradient(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:Function}, k, x
) where {TC}
    return get_gradient(p, x)[p.M, k]
end
function get_gradient(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector}, k, x
) where {TC}
    return p.gradient!![k](p.M, x)
end
function get_gradient!(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:Function}, X, k, x
) where {TC}
    copyto!(p.M[k], X, p.gradient!!(p.M, x)[p.M, k])
    return X
end
function get_gradient!(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector},
    X,
    k,
    x,
) where {TC}
    copyto!(p.M[k], X, p.gradient!![k](p.M, x))
    return X
end
function get_gradient(
    p::AlternatingGradientProblem{MutatingEvaluation,<:Manifold,TC}, k, x
) where {TC}
    X = zero_tangent_vector(p.M[k], x[p.M, k])
    get_gradient!(p, X, k, x)
    return X
end
function get_gradient!(
    p::AlternatingGradientProblem{MutatingEvaluation,<:Manifold,TC,<:Function}, X, k, x
) where {TC}
    # this takes a lot more allocations than other methods, but the gradient can only be evaluated in full
    Xf = zero_tangent_vector(p.M, x)
    get_gradient!(p, Xf, x)
    copyto!(p.M[k], X, x[p.M, k], Xf[p.M, k])
    return X
end
function get_gradient!(
    p::AlternatingGradientProblem{MutatingEvaluation,<:Manifold,TC,<:AbstractVector},
    X,
    k,
    x,
) where {TC}
    return p.gradient!![k](p.M, X, x)
end

"""
    AlternatingGradientDescentOptions <: AbstractGradientDescentOptions

Store the fields for an alternating gradient descent algorithm,
see also [`AlternatingGradientProblem`](@ref) and [`alternating_gradient_descent`](@ref).

# Fields
* `x` the current iterate
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ConstantStepsize`](@ref)`(1.0)`) a [`Stepsize`](@ref)
* `inner_iterations`– (`5`) how many gradient steps to take in a component before alternating to the next
* `evaluation_order` – (`:Linear`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle newly permuted sequence (`:Random`) or the default `:Linear` evaluation order.
* `order` the current permutation
* `retraction_method` – (`ExponentialRetraction()`) a `retraction(M,x,ξ)` to use.

# Constructor
    AlternatingGradientDescentOptions(x)

Create a [`AlternatingGradientDescentOptions`](@ref) with start point `x`.
all other fields are optional keyword arguments.
"""
mutable struct AlternatingGradientDescentOptions{
    TX,
    TV,
    D<:DirectionUpdateRule,
    TStop<:StoppingCriterion,
    TStep<:Stepsize,
    RM<:AbstractRetractionMethod,
} <: AbstractGradientOptions
    x::TX
    gradient::TV
    direction::D
    stop::TStop
    stepsize::TStep
    order_type::Symbol
    order::Vector{<:Int}
    retraction_method::RM
    k::Int # current iterate
    i::Int # inner iterate
    inner_iterations::Int
end
function AlternatingGradientDescentOptions(
    x,
    X,
    direction::DirectionUpdateRule;
    inner_iterations::Int=5,
    order_type::Symbol=:Linear,
    order::Vector{<:Int}=Int[],
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    stoping_criterion::StoppingCriterion=StopAfterIteration(1000),
    stepsize::Stepsize=ConstantStepsize(1.0),
)
    return AlternatingGradientDescentOptions{
        typeof(x),
        typeof(X),
        typeof(direction),
        typeof(stoping_criterion),
        typeof(stepsize),
        typeof(retraction_method),
    }(
        x,
        X,
        direction,
        stoping_criterion,
        stepsize,
        order_type,
        order,
        retraction_method,
        0,
        0,
        inner_iterations,
    )
end

"""
    AlternatingGradient <: DirectionUpdateRule

The default gradient processor, which just evaluates the (alternating) gradient on one of
the components
"""
struct AlternatingGradient{T} <: AbstractStochasticGradientProcessor
    dir::T
end

function (s::AlternatingGradient)(
    p::AlternatingGradientProblem, o::AlternatingGradientDescentOptions, iter
)
    # at begin of inner iterations reset internal vector to zero
    (o.i == 1) && zero_tangent_vector!(p.M, s.dir, o.x)
    # update order(k)th component inplace
    get_gradient!(p, s.dir[p.M, o.order[o.k]], o.order[o.k], o.x)
    return o.stepsize(p, o, iter), s.dir # return urrent full gradient
end

function (a::ArmijoLinesearch)(
    p::AlternatingGradientProblem, o::AlternatingGradientDescentOptions, ::Int
)
    X = zero_tangent_vector(p.M, o.x)
    X[p.M, o.order[o.k]] .= get_gradient(p, o.order[o.k], o.x)
    a.stepsizeOld = linesearch_backtrack(
        p.M,
        x -> p.cost(p.M, x),
        o.x,
        X,
        a.stepsizeOld,
        a.sufficientDecrease,
        a.contractionFactor,
        a.retraction_method,
    )
    return a.stepsizeOld
end
