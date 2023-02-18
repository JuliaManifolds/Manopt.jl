
"""
    StochasticGradientDescentState <: AbstractGradientDescentSolverState

Store the following fields for a default stochastic gradient descent algorithm,
see also [`ManifoldStochasticGradientObjective`](@ref) and [`stochastic_gradient_descent`](@ref).

# Fields

* `p` the current iterate
* `direction` ([`StochasticGradient`](@ref)) a direction update to use
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ConstantStepsize`](@ref)`(1.0)`) a [`Stepsize`](@ref)
* `evaluation_order` – (`:Random`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Linear`) or the default `:Random` one.
* `order` the current permutation
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) a `retraction(M, p, X)` to use.

# Constructor
    StochasticGradientDescentState(M, p)

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
    retraction_method::RM=default_retraction_method(M, typeof(p)),
    stopping_criterion::SC=StopAfterIteration(1000),
    stepsize::S=default_stepsize(M, StochasticGradientDescentState),
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
function show(io::IO, sgds::StochasticGradientDescentState)
    i = get_count(sgds, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(sgds.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Stochastic Gradient Descent
    $Iter
    ## Parameters
    * order: $(sgds.order_type)
    * retraction method: $(sgds.retraction_method)

    ## Stepsize
    $(sgds.stepsize)

    ## Stopping Criterion
    $(status_summary(sgds.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
"""
    StochasticGradient <: AbstractGradientGroupProcessor

The default gradient processor, which just evaluates the (stochastic) gradient or a subset
thereof.

# Constructor

    StochasticGradient(M::AbstractManifold; p=rand(M), X=zero_vector(M, p))

Initialize the stochastic Gradient processor with `X`, i.e. both `M` and `p` are just
help variables, though `M` is mandatory by convention.
"""
struct StochasticGradient{T} <: AbstractGradientGroupProcessor
    dir::T
end
function StochasticGradient(M::AbstractManifold; p=rand(M), X=zero_vector(M, p))
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

function default_stepsize(M::AbstractManifold, ::Type{StochasticGradientDescentState})
    return ConstantStepsize(M)
end

@doc raw"""
    stochastic_gradient_descent(M, gradF, x)

perform a stochastic gradient descent

# Input

* `M` a manifold ``\mathcal M``
* `gradF` – a gradient function, that either returns a vector of the subgradients
  or is a vector of gradients
* `x` – an initial value ``x ∈ \mathcal M``

# Optional
* `cost` – (`missing`) you can provide a cost function for example to track the function value
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient(s) works by
   allocation (default) form `gradF(M, x)` or [`InplaceEvaluation`](@ref) in place, i.e.
   is of the form `gradF!(M, X, x)` (elementwise).
* `evaluation_order` – (`:Random`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Linear`) or the default `:Random` one.
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ConstantStepsize`](@ref)`(1.0)`) a [`Stepsize`](@ref)
* `order_type` (`:RandomOder`) a type of ordering of gradient evaluations.
  values are `:RandomOrder`, a `:FixedPermutation`, `:LinearOrder`
* `order` - (`[1:n]`) the initial permutation, where `n` is the number of gradients in `gradF`.
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) a retraction to use.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details
"""
function stochastic_gradient_descent(
    M::AbstractManifold, gradF::TDF, p; kwargs...
) where {TDF}
    q = allocate(p)
    copyto!(M, q, p)
    return stochastic_gradient_descent!(M, gradF, q; kwargs...)
end

@doc raw"""
    stochastic_gradient_descent!(M, gradF, x)

perform a stochastic gradient descent in place of `x`.

# Input

* `M` a manifold ``\mathcal M``
* `gradF` – a gradient function, that either returns a vector of the subgradients
  or is a vector of gradients
* `x` – an initial value ``x ∈ \mathcal M``

for all optional parameters, see [`stochastic_gradient_descent`](@ref).
"""
function stochastic_gradient_descent!(
    M::AbstractManifold,
    grad_f::TDF,
    p;
    cost::TF=Missing(),
    direction::DirectionUpdateRule=StochasticGradient(zero_vector(M, p)),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    stopping_criterion::StoppingCriterion=StopAfterIteration(10000) |
                                          StopWhenGradientNormLess(1e-9),
    stepsize::Stepsize=default_stepsize(M, StochasticGradientDescentState),
    order_type::Symbol=:Random,
    order=collect(1:(grad_f isa Function ? length(grad_f(M, p)) : length(grad_f))),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
) where {TDF,TF}
    msgo = ManifoldStochasticGradientObjective(grad_f; cost=cost, evaluation=evaluation)
    dmsgo = decorate_objective!(M, msgo; kwargs...)
    mp = DefaultManoptProblem(M, dmsgo)
    sgds = StochasticGradientDescentState(
        M,
        p,
        zero_vector(M, p);
        direction=direction,
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        order_type=order_type,
        order=order,
        retraction_method=retraction_method,
    )
    sgds = decorate_state!(sgds; kwargs...)
    return get_solver_return(solve!(mp, sgds))
end
function initialize_solver!(::AbstractManoptProblem, s::StochasticGradientDescentState)
    s.k = 1
    (s.order_type == :FixedRandom) && (shuffle!(s.order))
    return s
end
function step_solver!(mp::AbstractManoptProblem, s::StochasticGradientDescentState, iter)
    step, s.X = s.direction(mp, s, iter)
    retract!(get_manifold(mp), s.p, s.p, -step * s.X)
    s.k = ((s.k) % length(s.order)) + 1
    return s
end
