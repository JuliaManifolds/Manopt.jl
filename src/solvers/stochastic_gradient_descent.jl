
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

    ## Stopping criterion

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
    stochastic_gradient_descent(M, grad_f, p; kwargs...)
    stochastic_gradient_descent(M, msgo, p; kwargs...)

perform a stochastic gradient descent

# Input

* `M` a manifold ``\mathcal M``
* `grad_f` – a gradient function, that either returns a vector of the subgradients
  or is a vector of gradients
* `p` – an initial value ``x ∈ \mathcal M``

alternatively to the gradient you can provide an [`ManifoldStochasticGradientObjective`](@ref) `msgo`,
then using the `cost=` keyword does not have any effect since if so, the cost is already within the objective.

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

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
stochastic_gradient_descent(M::AbstractManifold, args...; kwargs...)
function stochastic_gradient_descent(M::AbstractManifold, grad_f; kwargs...)
    return stochastic_gradient_descent(M, grad_f, rand(M); kwargs...)
end
function stochastic_gradient_descent(
    M::AbstractManifold,
    grad_f,
    p;
    cost=Missing(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    msgo = ManifoldStochasticGradientObjective(grad_f; cost=cost, evaluation=evaluation)
    return stochastic_gradient_descent(M, msgo, p; evaluation=evaluation, kwargs...)
end
function stochastic_gradient_descent(
    M::AbstractManifold,
    grad_f,
    p::Number;
    cost=Missing(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    q = [p]
    f_ = ismissing(cost) ? cost : (M, p) -> cost(M, p[])
    if grad_f isa Function
        n = grad_f(M, p) isa Number
        grad_f_ = (M, p) -> [[X] for X in (n ? [grad_f(M, p[])] : grad_f(M, p[]))]
    else
        if evaluation isa AllocatingEvaluation
            grad_f_ = [(M, p) -> [f(M, p[])] for f in grad_f]
        else
            grad_f_ = [(M, X, p) -> (X .= [f(M, p[])]) for f in grad_f]
        end
    end
    rs = stochastic_gradient_descent(
        M, grad_f_, q; cost=f_, evaluation=evaluation, kwargs...
    )
    #return just a number if  the return type is the same as the type of q
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function stochastic_gradient_descent(
    M::AbstractManifold, msgo::O, p; kwargs...
) where {O<:Union{ManifoldStochasticGradientObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return stochastic_gradient_descent!(M, msgo, q; kwargs...)
end

@doc raw"""
    stochastic_gradient_descent!(M, grad_f, p)
    stochastic_gradient_descent!(M, msgo, p)

perform a stochastic gradient descent in place of `p`.

# Input

* `M` a manifold ``\mathcal M``
* `grad_f` – a gradient function, that either returns a vector of the subgradients
  or is a vector of gradients
* `p` – an initial value ``p ∈ \mathcal M``

Alternatively to the gradient you can provide an [`ManifoldStochasticGradientObjective`](@ref) `msgo`,
then using the `cost=` keyword does not have any effect since if so, the cost is already within the objective.

for all optional parameters, see [`stochastic_gradient_descent`](@ref).
"""
stochastic_gradient_descent!(::AbstractManifold, args...; kwargs...)
function stochastic_gradient_descent!(
    M::AbstractManifold,
    grad_f,
    p;
    cost=Missing(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    msgo = ManifoldStochasticGradientObjective(grad_f; cost=cost, evaluation=evaluation)
    return stochastic_gradient_descent!(M, msgo, p; evaluation=evaluation, kwargs...)
end
function stochastic_gradient_descent!(
    M::AbstractManifold,
    msgo::O,
    p;
    direction::DirectionUpdateRule=StochasticGradient(zero_vector(M, p)),
    stopping_criterion::StoppingCriterion=StopAfterIteration(10000) |
                                          StopWhenGradientNormLess(1e-9),
    stepsize::Stepsize=default_stepsize(M, StochasticGradientDescentState),
    order=collect(1:length(get_gradients(M, msgo, p))),
    order_type::Symbol=:Random,
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
) where {O<:Union{ManifoldStochasticGradientObjective,AbstractDecoratedManifoldObjective}}
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
    dsgds = decorate_state!(sgds; kwargs...)
    solve!(mp, dsgds)
    return get_solver_return(get_objective(mp), dsgds)
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
