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
   allocation (default) form `gradF(M, x)` or [`MutatingEvaluation`](@ref) in place, i.e.
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
* `x_opt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
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
    stepsize::Stepsize=ConstantStepsize(1.0),
    order_type::Symbol=:Random,
    order=collect(1:(gradF isa Function ? length(gradF(M, x)) : length(gradF))),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    return_options=false,
    kwargs...,
) where {TDF,TF}
    p = StochasticGradientProblem(M, gradF; cost=cost, evaluation=evaluation)
    o = StochasticGradientDescentOptions(
        x,
        zero_vector(M, x),
        direction;
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        order_type=order_type,
        order=order,
        retraction_method=retraction_method,
    )
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end
function initialize_solver!(
    ::StochasticGradientProblem, o::StochasticGradientDescentOptions
)
    o.k = 1
    (o.order_type == :FixedRandom) && (shuffle!(o.order))
    return o
end
function step_solver!(
    p::StochasticGradientProblem, o::StochasticGradientDescentOptions, iter
)
    s, o.gradient = o.direction(p, o, iter)
    retract!(p.M, o.x, o.x, -s * o.gradient)
    o.k = ((o.k) % length(o.order)) + 1
    return o
end
get_solver_result(o::StochasticGradientDescentOptions) = o.x
