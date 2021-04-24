@doc raw"""
    AlternatingGradientProblem <: Problem

An alternating gradient problem consists of
* a `ProductManifold M`
* a cost function ``f(x)``
* an array of gradients, i.e. a function that returns and array or an array of functions
  ``\{\operatorname{grad}f_i\}_{i=1}^n``, where both variants can be given in the allocating
  variant and the array of function may also be provided as mutating functions `(M, X_i, x) -> X_i`.
  Each component of the array corresponds to a component of the product manifold.

!!! Note
    This Problem requires the `ProductManifold` to be loaded from `Manifolds`.

!!! Note
    The input of each of the (component) gradients is still the whole vector `x`, just that
    up to the `i`th component all other values are assumed to be fixed.


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
    return ProductRepr(p.gradient!!(p.M, x)...)
end
function get_gradient(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector}, x
) where {TC}
    Y = ProductRepr([gi(M, x) for gi ∈ p.gradient!!]...)
    return Y
end
function get_gradient!(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:Function}, X, x
) where {TC}
    copyto!(M, X, get_gradient(p, X, x))
    return X
end
function get_gradient!(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector}, X, x
) where {TC}
    copyto!(M, X, get_gradient(p, x))
    return X
end
function get_gradient(
    p::AlternatingGradientProblem{MutatingEvaluation,<:Manifold,TC,<:Function}, x
) where {TC}
    Y = zero_tangent_vector(p.M, x)
    return p.gradient!!(M, Y, x)
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
    for gi ∈ p.gradient!!
        gi(p.M, X, x)
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
    return p.gradient!!(p.M, x)[k]
end
function get_gradient(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector}, k, x
) where {TC}
    return p.gradient!![k](p.M, x)
end
function get_gradient!(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:Function}, X, k, x
) where {TC}
    copyto!(M.manifolds[k], X, p.gradient!!(p.M, x)[k])
    return X
end
function get_gradient!(
    p::AlternatingGradientProblem{AllocatingEvaluation,<:Manifold,TC,<:AbstractVector},
    X,
    k,
    x,
) where {TC}
    copyto!(X, p.gradient!![k](p.M, x))
    return X
end
function get_gradient(
    p::AlternatingGradientProblem{MutatingEvaluation,<:Manifold,TC}, k, x
) where {TC}
    X = zero_tangent_vector(p.M, x)
    return get_gradient!(p, X, k, x)
end
function get_gradient!(
    ::AlternatingGradientProblem{MutatingEvaluation,<:Manifold,TC,<:Function},
    ::Any,
    ::Any,
    ::Any,
) where {TC}
    return error(
        "A mutating variant of the alternating gradient as a single function is not implemented.",
    )
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
* `evaluation_order` – (`:Random`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Linear`) or the default `:Random` one.
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
    order_type::Symbol=:RandomOrder,
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
    if o.i == 1 # at begin of inner iterations.
        # for each new epoche choose new order if we are at random order
        ((o.k == 1) && (o.order_type == :Random)) && shuffle!(o.order)
        # i is the gradient to choose, either from the order or completely random
        zero_tangent_vector!(p.M, s.dir, o.x) # reset internal vector to zero
    end
    j = o.order_type == :Random ? rand(1:length(o.order)) : o.order[o.k]
    # update jth component inplace
    get_gradient!(p, s.dir, j, o.x)
    return o.stepsize(p, o, iter), s.dir # return jth component
end
