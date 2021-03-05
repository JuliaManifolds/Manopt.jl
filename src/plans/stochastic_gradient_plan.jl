@doc raw"""
    StochasticGradientProblem <: Problem

A stochastic gradient problem consists of
* a `Manifold M`
* a(n optional) cost function ``f(x) = \displaystyle\sum_{i=1}^n f_i(x)
* an array of gradients, i.e. a function that returns and array or an array of functions
``\{\operatorname{grad}f_i\}_{i=1}^n``, where both variants can be given in the allocating
variant and the array of function may also be provided as mutating functions `(X,x) -> X`.

# Constructors
    StochasticGradientProblem(M::Manifold, gradF::Function;
        cost=Missing(), evaluation=AllocatingEvaluation()
    )
    StochasticGradientProblem(M::Manifold, gradF::AbstractVector{<:Function};
        cost=Missing(), evaluation=AllocatingEvaluation()
    )

Create a Stochastic gradient problem with an optional `cost` and the gradient either as one
function (returning an array) or a vector of functions.
"""
struct StochasticGradientProblem{T,MT<:Manifold,TCost,TGradient} <:
       AbstractGradientProblem{T}
    M::MT
    cost::TCost
    gradient!!::TGradient
end
function StochasticGradientProblem(
    M::TM,
    gradF!!::G;
    cost::Union{Function,Missing}=Missing(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TM<:Manifold,G}
    return StochasticGradientProblem{typeof(evaluation),TM,typeof(cost),G}(M, cost, gradF!!)
end
function StochasticGradientProblem(
    M::TM,
    gradF!!::AbstractVector{<:Function};
    cost::Union{Function,Missing}=Missing(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
) where {TM<:Manifold}
    return StochasticGradientProblem{typeof(evaluation),TM,typeof(cost),typeof(gradF!!)}(
        M, cost, gradF!!
    )
end

@doc raw"""
    get_gradients(P::StochasticGradientProblem, x)
    get_gradients!(P::StochasticGradientProblem, Y, x)

Evaluate all summands gradients ``\{\operatorname{grad}f_i\}_{i=1}^n`` at `x` (in place of `Y`).

Note that for the [`MutatingEvaluation`](@ref) based problem and a single function for the
stochastic gradient, the allocating variant is not available.
"""
function get_gradients(
    p::StochasticGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:Function}, x
) where {TC}
    return p.gradient!!(p.M, x)
end
function get_gradients(
    p::StochasticGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector}, x
) where {TC}
    return [grad_i(p.M, x) for grad_i in p.gradient!!]
end
function get_gradients!(
    p::StochasticGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:Function}, X, x
) where {TC}
    copyto!(X, p.gradient!!(p.M, x))
    return X
end
function get_gradients!(
    p::StochasticGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector}, X, x
) where {TC}
    copyto!(X, [grad_i(p.M, x) for grad_i in p.gradient!!])
    return X
end
function get_gradients(
    ::StochasticGradientProblem{MutatingEvaluation,<:Manifold,TC,<:Function}, ::Any
) where {TC}
    return error(
        "For a mutating function type stochastic gradient, the allocating variant is not possible.",
    )
end
function get_gradients(
    p::StochasticGradientProblem{MutatingEvaluation,<:Manifold,TC,<:AbstractVector}, x
) where {TC}
    X = [zero_tangent_vector(p.M, x) for _ in 1:length(p.gradient!!)]
    return get_gradients!(p, X, x)
end
function get_gradients!(
    p::StochasticGradientProblem{MutatingEvaluation,<:Manifold,TC,<:Function}, X, x
) where {TC}
    return p.gradient!!(p.M, X, x)
end
function get_gradients!(
    p::StochasticGradientProblem{MutatingEvaluation,<:Manifold,TC,<:AbstractVector}, X, x
) where {TC}
    for i in 1:length(p.gradient!!)
        p.gradient!![i](p.M, X[i], x)
    end
    return X
end

@doc raw"""
    get_gradient(p::StochasticGradientProblem, k, x)
    get_gradient!(p::StochasticGradientProblem, Y, k, x)

Evaluate one of the summands gradients ``\operatorname{grad}f_k``, ``k∈\{1,…,n\}``, at `x` (in place of `Y`).

Note that for the [`MutatingEvaluation`](@ref) based problem and a single function for the
stochastic gradient mutating variant is not available, since it would require too many allocatins.
"""
function get_gradient(
    p::StochasticGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:Function}, k, x
) where {TC}
    return p.gradient!!(p.M, x)[k]
end
function get_gradient(
    p::StochasticGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector}, k, x
) where {TC}
    return p.gradient!![k](p.M, x)
end
function get_gradient!(
    p::StochasticGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:Function}, X, k, x
) where {TC}
    copyto!(X, p.gradient!!(p.M, x)[k])
    return X
end
function get_gradient!(
    p::StochasticGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector},
    X,
    k,
    x,
) where {TC}
    copyto!(X, p.gradient!![k](p.M, x))
    return X
end
function get_gradient(
    p::StochasticGradientProblem{MutatingEvaluation,<:Manifold,TC}, k, x
) where {TC}
    X = zero_tangent_vector(p.M, x)
    return get_gradient!(p, X, k, x)
end
function get_gradient!(
    ::StochasticGradientProblem{MutatingEvaluation,<:Manifold,TC,<:Function},
    ::Any,
    ::Any,
    ::Any,
) where {TC}
    return error(
        "A mutating variant of the stochastic gradient as a single function is not implemented.",
    )
end
function get_gradient!(
    p::StochasticGradientProblem{MutatingEvaluation,<:Manifold,TC,<:AbstractVector}, X, k, x
) where {TC}
    return p.gradient!![k](p.M, X, x)
end

"""
    AbstractStochasticGradientDescentOptions <: Options

A generic type for all options related to stochastic gradient descent methods
"""
abstract type AbstractStochasticGradientProcessor <: DirectionUpdateRule end

"""
    StochasticGradientDescentOptions <: AbstractStochasticGradientDescentOptions

Store the following fields for a default stochastic gradient descent algorithm,
see also [`StochasticGradientProblem`](@ref) and [`stochastic_gradient_descent`](@ref).

# fields

# Fields
* `x` the current iterate
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ConstantStepsize`](@ref)`(1.0)`) a [`Stepsize`](@ref)
* `evaluation_order` – (`:Random`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Linear`) or the default `:Random` one.
* `order` the current permutation
* `retraction_method` – (`ExponentialRetraction()`) a `retraction(M,x,ξ)` to use.

# Constructor
    StochasticGradientDescentOptions(x)

Create a [`StochasticGradientDescentOptions`](@ref) with start point `x`.
all other fields are optional keyword arguments.
"""
mutable struct StochasticGradientDescentOptions{
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
end
function StochasticGradientDescentOptions(
    x,
    X,
    direction::DirectionUpdateRule;
    order_type::Symbol=:RandomOrder,
    order::Vector{<:Int}=Int[],
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    stoping_criterion::StoppingCriterion=StopAfterIteration(1000),
    stepsize::Stepsize=ConstantStepsize(1.0),
)
    return StochasticGradientDescentOptions{
        typeof(x),
        typeof(X),
        typeof(direction),
        typeof(stoping_criterion),
        typeof(stepsize),
        typeof(retraction_method),
    }(
        x, X, direction, stoping_criterion, stepsize, order_type, order, retraction_method, 0
    )
end

"""
    StochasticGradient <: DirectionUpdateRule

The default gradient processor, which just evaluates the (stochastic) gradient or a subset
thereof.
"""
struct StochasticGradient{T} <: AbstractStochasticGradientProcessor
    dir::T
end

function (s::StochasticGradient)(
    p::StochasticGradientProblem, o::StochasticGradientDescentOptions, iter
)
    # for each new epoche choose new order if we are at random order
    ((o.k == 1) && (o.order_type == :Random)) && shuffle!(o.order)
    # i is the gradient to choose, either from the order or completely random
    j = o.order_type == :Random ? rand(1:length(o.order)) : o.order[o.k]
    return o.stepsize(p, o, iter), get_gradient!(p, s.dir, j, o.x)
end
function MomentumGradient(
    p::StochasticGradientProblem,
    x0::P,
    s::DirectionUpdateRule=StochasticGradient(zero_tangent_vector(p.M, x0));
    gradient=zero_tangent_vector(p.M, x0),
    momentum=0.2,
    vector_transport_method::VTM=ParallelTransport(),
) where {P,VTM<:AbstractVectorTransportMethod}
    return MomentumGradient{P,typeof(gradient),typeof(momentum),VTM}(
        deepcopy(x0), gradient, momentum, s, vector_transport_method
    )
end
function AverageGradient(
    p::StochasticGradientProblem,
    x0::P,
    n::Int=10,
    s::DirectionUpdateRule=StochasticGradient(zero_tangent_vector(p.M, x0));
    gradients=fill(zero_tangent_vector(p.M, x0), n),
    vector_transport_method::VTM=ParallelTransport(),
) where {P,VTM}
    return AverageGradient{P,eltype(gradients),VTM}(
        gradients, deepcopy(x0), s, vector_transport_method
    )
end
