@doc raw"""
    StochasticGradientProblem <: Problem

A stochastic gradient problem consists of
* a `Manifold M`
* a(n optional) cost function ``f(x) = \displaystyle\sum_{i=1}^n f_i(x)
* an array of gradients, i.e. a function that returns and array or an array of functions ``\{∇f_i\}_{i=1}^n``.

# Constructors
    StochasticGradientProblem(M::Manifold, ∇::Function)
    StochasticGradientProblem(M::Manifold, cost::Function, ∇::Function)
    StochasticGradientProblem(M::Manifold, ∇::AbstractVector{<:Function})
    StochasticGradientProblem(M::Manifold, cost::Function, ∇::::AbstractVector{<:Function})

Create a Stochastic gradient problem with an optional `cost` and the gradient either as one
function (returning an array) or a vector of functions.
"""
struct StochasticGradientProblem{MT<:Manifold,TCost,TGradient} <: Problem
    M::MT
    cost::TCost
    ∇::TGradient
end
function StochasticGradientProblem(M::TM, ∇::Function) where {TM<:Manifold}
    return StochasticGradientProblem{TM,Missing,Function}(M, Missing(), ∇)
end
function StochasticGradientProblem(M::TM, cost::Function, ∇::Function) where {TM<:Manifold}
    return StochasticGradientProblem{TM,Function,Function}(M, cost, ∇)
end
function StochasticGradientProblem(
    M::TM, ∇::AbstractVector{<:Function}
) where {TM<:Manifold}
    return StochasticGradientProblem{TM,Missing,Function}(M, Missing(), ∇)
end
function StochasticGradientProblem(
    M::TM, cost::Function, ∇::AbstractVector{<:Function}
) where {TM<:Manifold}
    return StochasticGradientProblem{TM,Function,typeof(∇)}(M, cost, ∇)
end

@doc raw"""
    get_gradients(P::StochasticGradientProblem, x)

Evaluate all summands gradients ``\{∇f_i\}_{i=1}^n`` at `x`.
"""
function get_gradients(P::StochasticGradientProblem{<:Manifold,TC,<:Function}, x) where {TC}
    return P.∇(x)
end
function get_gradients(
    P::StochasticGradientProblem{<:Manifold,TC,<:AbstractVector}, x
) where {TC}
    return [∇i(x) for ∇i in P.∇]
end

@doc raw"""
    get_gradient(P::StochasticGradientProblem, k, x)

Evaluate one of the summands gradients ``∇f_k``, ``k\in \{1,…,n\}``, at `x`.
"""
function get_gradient(
    P::StochasticGradientProblem{<:Manifold,TC,<:Function}, k, x
) where {TC}
    return P.∇(x)[k]
end
function get_gradient(
    P::StochasticGradientProblem{<:Manifold,TC,<:AbstractVector}, k, x
) where {TC}
    return P.∇[k](x)
end

"""
    AbstractStochasticGradientOptions <: Options

A generic type for all options related to stochastic gradient descent methods
"""
abstract type AbstractStochasticGradientOptions <: Options end

"""
    StochasticGradientOptions <: AbstractStochasticGradientOptions

Store the following fields for a default stochastic gradient descent algorithm,
see also [`StochasticGradientProblem`](@ref) and [`stochastic_gradient_descent`](@ref).

# Fields
* `x` the current iterate
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ConstantStepsize`](@ref)`(1.0)`) a [`Stepsize`](@ref)
* `evaluation_order` – (`:RandomOrder`) how to cycle through the gradients. Possible values are
  * `:RandomOrder` choose a random permutation per epoche (cycle through all gradients)
  * `:FixedRandomOrder` choose one permutation for all epoches
  * `:LinearOrder` – cycle through the gradients in a linear fashion
  * `:Random` – choose a random gradient each step
* `order` the current permutation
* `retraction_method` – (`ExponentialRetraction()`) a `retraction(M,x,ξ)` to use.

# Constructor
    StochasticGradientOptions(x)

Create a [`StochasticGradientOptions`](@ref) with start point `x`.
all other fields are optional keyword arguments.
"""
struct StochasticGradientOptions{
    TX,TStop<:StoppingCriterion,TStep<:Stepsize,RM<:AbstractRetractionMethod
} <: AbstractStochasticGradientOptions
    x::TX
    stopping_criterion::TStop
    stepsize::TStep
    order_type::Symbol
    order::Vector{Int}
    retraction_method::RM
    k::Int # current iterate
end
function StochasticGradientOptions(
    x;
    stoping_criterion::StoppingCriterion=StopAfterIteration(1000),
    stepsize::Stepsize=ConstantStepsize(0.1),
    order_type::Symbol=:RandomOrder,
    order=[],
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
)
    return StochasticGradientOptions{
        typeof(x),typeof(stoping_criterion),typeof(step_size),typeof(retraction_method)
    }(
        x, stoping_criterion, stepsize, order_type, order, retraction_method, 0
    )
end

@doc raw"""
    MomentumStochasticGradientOptions <: AbstractStochasticGradientOptions

Store the following fields for a default stochastic gradient descent algorithm,
see also [`StochasticGradientProblem`](@ref) and [`stochastic_gradient_descent`](@ref).

Compared to the classic [`StochasticGradientOptions`](@ref) these options further
store the last direction as `∇` and update the new direction as
```math
    ∇_{k} = αP_{x_k\gets x_{k-1}}∇_{k-1} - η∇f_j(x_k)
```
where ``η`` is the `stepsize`, ``α`` is the `momentum` and the last direktion
``∇_{k-1}`` is also transported to the current tangent space (see `vector_transport_method`)

# Fields
* `x` - the current iterate
* `∇` - the last update direction
* `momentum` (`0.2`) the effect of the previous direction to the current
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ConstantStepsize`](@ref)`(1.0)`) a [`Stepsize`](@ref)
* `evaluation_order` – (`:RandomOrder`) how to cycle through the gradients. Possible values are
  * `:RandomOrder` choose a random permutation per epoche (cycle through all gradients)
  * `:FixedRandomOrder` choose one permutation for all epoches
  * `:LinearOrder` – cycle through the gradients in a linear fashion
  * `:Random` – choose a random gradient each step
* `order` the current permutation
* `retraction_method` – (`ExponentialRetraction()`) a `retraction(M,x,ξ)` to use.
* `vector_transport_method` – (`ParallelTransport()`) vector transport for the old direction

# Constructor
    StochasticGradientOptions(x)

Create a [`StochasticGradientOptions`](@ref) with start point `x`.
all other fields are optional keyword arguments.
"""
struct MomentumStochasticGradientOptions{
    TX,
    TN,
    R<:Real,
    TStop<:StoppingCriterion,
    TStep<:Stepsize,
    RM<:AbstractRetractionMethod,
    VTM<:AbstractVectorTransportMethod,
} <: AbstractStochasticGradientOptions
    x::TX
    ∇::TN
    momentum::R
    stopping_criterion::TStop
    stepsize::TStep
    order_type::Symbol
    order::Vector{Int}
    retraction_method::RM
    vector_transport_method::VTM
    k::Int # current iterate
end
function MomentumStochasticGradientOptions(
    x,
    ∇;
    stoping_criterion::StoppingCriterion=StopAfterIteration(1000),
    stepsize::Stepsize=ConstantStepsize(0.1),
    order_type::Symbol=:RandomOrder,
    order=[],
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
    momentum::Real=0.2,
)
    return StochasticGradientOptions{
        typeof(x),
        typeof(∇),
        typeof(momentum).typeof(stoping_criterion),
        typeof(step_size),
        typeof(retraction_method),
        typeof(vector_transport_method),
    }(
        x,
        ∇,
        stoping_criterion,
        stepsize,
        order_type,
        order,
        retraction_method,
        vector_transport_method,
        0,
    )
end

struct AdaStochasticGradientOptions <: AbstractStochasticGradientOptions end
struct AveragingStochasticGradientOptions <: AbstractStochasticGradientOptions end