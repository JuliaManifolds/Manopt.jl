@doc raw"""
    alternating_gradient_descent(M, F, gradF, x)

perform an alternating gradient descent

# Input

* `M` a product manifold ``\mathcal M = \mathcal M_1 × \mathcal M_2 × ⋯ ×\mathcal M_n``
* `F` – the objective functioN (cost)
* `gradF` – a gradient function, that either returns a `ProductRepr` of the component gradients
  or is a vector of gradient functions per component
* `x` – an initial value ``x ∈ \mathcal M``

# Optional
* `cost` – (`missing`) you can provide a cost function for example to track the function value
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient(s) works by
   allocation (default) form `gradF(M, x)` or [`MutatingEvaluation`](@ref) in place, i.e.
   is of the form `gradF!(M, X, x)` (elementwise).
* `evaluation_order` – (`:Random`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Linear`) or the default `:Random` one.
* `inner_iterations`– (`5`) how many gradient steps to take in a component before alternating to the next
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ConstantStepsize`](@ref)`(1.0)`) a [`Stepsize`](@ref)
* `order_type` (`:RandomOder`) a type of ordering of gradient evaluations.
  values are `:RandomOrder`, a `:FixedPermutation`, `:LinearOrder`
* `order` - (`[1:n]`) the initial permutation, where `n` is the number of gradients in `gradF`.
* `retraction_method` – (`ExponentialRetraction()`) a `retraction(M,x,ξ)` to use.

# Output
* `x_opt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function alternating_gradient_descent(
    M::ProductManifold, F, gradF::Union{TgF,AbstractVector{<:TgF}}, x; kwargs...
) where {TgF}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return alternating_gradient_descent!(M, F, gradF, x_res; kwargs...)
end
@doc raw"""
    alternating_gradient_descent!(M, F, gradF, x)

perform a alternating gradient descent in place of `x`.

# Input

* `M` a manifold ``\mathcal M``
* `F` – the objective functioN (cost)
* `gradF` – a gradient function, that either returns a vector of the subgradients
  or is a vector of gradients
* `x` – an initial value ``x ∈ \mathcal M``

for all optional parameters, see [`alternating_gradient_descent`](@ref).
"""
function alternating_gradient_descent!(
    M::ProductManifold,
    F,
    gradF::Union{TgF,AbstractVector{<:TgF}},
    x;
    direction::DirectionUpdateRule=AlternatingGradient(zero_tangent_vector(M, x)),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inner_iterations::Int=5,
    stoping_criterion::StoppingCriterion=StopAfterIteration(100) |
                                         StopWhenGradientNormLess(1e-9),
    stepsize::Stepsize=ConstantStepsize(1.0),
    order_type::Symbol=:Random,
    order=collect(1:(gradF isa Function ? length(gradF(M, x)) : length(gradF))),
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
    return_options=false,
    kwargs...,
) where {TgF}
    p = AlternatingGradientProblem(M, F, gradF; evaluation=evaluation)
    o = AlternatingGradientDescentOptions(
        x,
        get_gradient(p, x),
        direction;
        inner_iterations=inner_iterations,
        stoping_criterion=stoping_criterion,
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
    p::AlternatingGradientProblem, o::AlternatingGradientDescentOptions
)
    o.k = 1
    o.i = 1
    (o.order_type == :FixedRandom) && (shuffle!(o.order))
    return o
end
function step_solver!(
    p::AlternatingGradientProblem, o::AlternatingGradientDescentOptions, iter
)
    s, o.gradient = o.direction(p, o, iter)
    retract!(p.M[o.k], o.x[p.M, o.k], o.x[p.M, o.k], -s * o.gradient[p.M, o.k])
    o.i += 1
    if o.i > o.inner_iterations
        o.k = ((o.k) % length(o.order)) + 1
        o.i = 1
    end
    return o
end
get_solver_result(o::AlternatingGradientDescentOptions) = o.x
