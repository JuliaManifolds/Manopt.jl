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
* `retraction_method` – (`default_retraction_method(M)`) a `retraction(M,x,ξ)` to use.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details
"""
function stochastic_gradient_descent(
    M::AbstractManifold, gradF::TDF, x; kwargs...
) where {TDF}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return stochastic_gradient_descent!(M, gradF, x_res; kwargs...)
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
    gradF::TDF,
    x;
    cost::TF=Missing(),
    direction::DirectionUpdateRule=StochasticGradient(zero_vector(M, x)),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    stopping_criterion::StoppingCriterion=StopAfterIteration(10000) |
                                          StopWhenGradientNormLess(1e-9),
    stepsize::Stepsize=ConstantStepsize(M),
    order_type::Symbol=:Random,
    order=collect(1:(gradF isa Function ? length(gradF(M, x)) : length(gradF))),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    kwargs...,
) where {TDF,TF}
    p = ManifoldStochasticGradientObjective(M, gradF; cost=cost, evaluation=evaluation)
    mp = DefaultManoptProblem(M, os)
    o = StochasticGradientDescentState(
        M,
        x,
        zero_vector(M, x);
        direction=direction,
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        order_type=order_type,
        order=order,
        retraction_method=retraction_method,
    )
    o = decorate_state(o; kwargs...)
    return get_solver_return(solve!(mp, o))
end
function initialize_solver!(::AbstractManoptProblem, s::StochasticGradientDescentState)
    s.k = 1
    (s.order_type == :FixedRandom) && (shuffle!(s.order))
    return s
end
function step_solver!(mp::AbstractManoptProblem, s::StochasticGradientDescentState, iter)
    step, s.gradient = s.direction(mp, s, iter)
    retract!(mp.M, s.x, s.x, -step * s.gradient)
    s.k = ((s.k) % length(s.order)) + 1
    return s
end
