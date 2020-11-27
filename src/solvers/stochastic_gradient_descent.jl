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
* `evaluation_order` – ([`RandomEvalOrder`](@ref)`()`) how to cycle through the gradients.
  Other values are [`LinearEvalOrder`](@ref)`()` that takes a new random order each
  iteration, and [`FixedRandomEvalOrder`](@ref)`()` that fixes a random cycle for all iterations.
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
    stoping_criterion::StoppingCriterion=StopAfterIteration(1000),
    stepsize::Stepsize=ConstantStepsize(0.1),
    order_type::Symbol=:RandomOrder,
    order=collect(1:(∇F isa Function ? length(∇F(x)) : length(∇F))),
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
    momentum::Float64=0.0,
)
    p = StochasticGradientProblem(M, cost, ∇F)
    if ((momentum) > 1.0 || (momentum < 0.0))
        error("Momentum hast to be in [0,1] not $momentum.")
    end
    if momentum == 0.0
        o = StochasticGradientOptions(
            x0;
            stoping_criterion=stoping_criterion,
            stepsize=stepsize,
            order_type=order_type,
            order=order,
            retraction_method=retraction_method,
        )
    else
        o = MomentumStochasticGradientOptions(
            x0,
            zero_tangent_vector(M, x0);
            stoping_criterion=stoping_criterion,
            stepsize=stepsize,
            order_type=order_type,
            order=order,
            retraction_method=retraction_method,
            vector_transport_method=vector_transport_method,
        )
    end
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end
function initialize_solver!(
    ::StochasticGradientProblem, o::AbstractStochasticGradientOptions
)
    o.k = 1
    (o.order_type == :FixedRandomOrder) && (shuffle!(o.order))
    return o
end
function step_solver!(p::StochasticGradientProblem, o::StochasticGradientOptions, iter)
    # for each new epoche choose new order if we are at random order
    ((k == 1) && (o.order_type == :RandomOrder)) && shuffle!(o.order)
    # i is the gradient to choose, either from the order or completely random
    j = o.order_type == :Random ? rand(1:length(o.order)) : o.order[k]
    # evaluate the gradient and do step
    retract!(p.M, o.x, o.x, -o.stepsize(p, o, iter) .* get_gradient(p, j, o.x))
    # move forward in cycle
    return o.k = ((o.k) % length(o.order)) + 1
end
function step_solver!(
    p::StochasticGradientProblem, o::MomentumStochasticGradientOptions, iter
)
    # for each new epoche choose new order if we are at random order
    ((k == 1) && (o.order_type == :RandomOrder)) && shuffle!(o.order)
    # i is the gradient to choose, either from the order or completely random
    j = o.order_type == :Random ? rand(1:length(o.order)) : o.order[k]
    x_old = deepcopy(o.x)
    o.∇ .= o.momentum .* o.∇ - o.stepsize(p, o, iter) .* get_gradient(p, j, o.x)
    # evaluate the gradient and do step
    retract!(p.M, o.x, x_old, -o.stepsize(p, o, iter) .* get_gradient(p, j, o.x))
    # parallel transport direction to new tangent space
    vector_transport_to!(p.M, o.∇, x_old, o.∇, o.x, o.vector_transport_method)
    # move forward in cycle
    return o.k = ((o.k) % length(o.order)) + 1
end
get_solver_result(o::AbstractStochasticGradientOptions) = o.x
