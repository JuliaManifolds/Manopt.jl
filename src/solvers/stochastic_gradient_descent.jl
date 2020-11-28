@doc raw"""
    stochastic_gradient_descent(M, ∇F, x)

perform a stochastic gradient descent

# Input

* `M` a manifold ``\mathcal M``
* `∇F` – a gradient function, that either returns a vector of the subgradients
  or is a vector of gradients
* `x` – an initial value $x ∈ \mathcal M$

# Optional
* `cost` – (`missing`) you can provide a cost function for example to track the function value
* `evaluation_order` – (`:Random`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Linear`) or the default `:Random` one.
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ConstantStepsize`](@ref)`(1.0)`) a [`Stepsize`](@ref)
* `order_type` (`:RandomOder`) a type of ordering of gradient evaluations.
  values are `:RandomOrder`, a `:FixedPermutation`, `:LinearOrder`
* `order` - (`[1:n]`) the initial permutation, where `n` is the number of gradients in `∇F`.
* `retraction_method` – (`ExponentialRetraction()`) a `retraction(M,x,ξ)` to use.

# Output
* `x_opt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function stochastic_gradient_descent(
    M::Manifold,
    ∇F::Union{Function,AbstractVector{<:Function}},
    x0;
    cost::Union{Function,Missing}=Missing(),
    direction::AbstractGradientProcessor=StochasticGradtient(),
    stoping_criterion::StoppingCriterion=StopAfterIteration(1000),
    stepsize::Stepsize=ConstantStepsize(0.1),
    order_type::Symbol=:Random,
    order=collect(1:(∇F isa Function ? length(∇F(x)) : length(∇F))),
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
    momentum::Float64=0.0,
)
    p = StochasticGradientProblem(M, cost, ∇F)
    if ((momentum) > 1.0 || (momentum < 0.0))
        error("Momentum hast to be in [0,1] not $momentum.")
    end
    o = StochasticGradientDescentOptions(
        x0;
        stoping_criterion=stoping_criterion,
        stepsize=stepsize,
        order_type=order_type,
        order=order,
        direction=direction,
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
    s, η = o.direction(p, o, i)
    retract!(p.M, o.x, o.x, -s * η)
    # move forward in cycle
    o.k = ((o.k) % length(o.order)) + 1
    return o
end
get_solver_result(o::StochasticGradientDescentOptions) = o.x
