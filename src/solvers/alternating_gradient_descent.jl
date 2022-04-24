@doc raw"""
    alternating_gradient_descent(M, F, gradF, x)

perform an alternating gradient descent

# Input

* `M` – the product manifold ``\mathcal M = \mathcal M_1 × \mathcal M_2 × ⋯ ×\mathcal M_n``
* `F` – the objective function (cost) defined on `M`.
* `gradF` – a gradient, that can be of two cases
  * is a single function returning a `ProductRepr` or
  * is a vector functions each returning a component part of the whole gradient
* `x` – an initial value ``x ∈ \mathcal M``

# Optional
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient(s) works by
   allocation (default) form `gradF(M, x)` or [`MutatingEvaluation`](@ref) in place, i.e.
   is of the form `gradF!(M, X, x)` (elementwise).
* `evaluation_order` – (`:Linear`) – whether
  to use a randomly permuted sequence (`:FixedRandom`), a per
  cycle permuted sequence (`:Random`) or the default `:Linear` one.
* `inner_iterations`– (`5`) how many gradient steps to take in a component before alternating to the next
* `stopping_criterion` ([`StopAfterIteration`](@ref)`(1000)`)– a [`StoppingCriterion`](@ref)
* `stepsize` ([`ArmijoLinesearch`](@ref)`()`) a [`Stepsize`](@ref)
* `order` - (`[1:n]`) the initial permutation, where `n` is the number of gradients in `gradF`.
* `retraction_method` – (`default_retraction_method(M)`) a `retraction(M, p, X)` to use.

# Output
* `x_opt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)

!!! note

    This Problem requires the `ProductManifold` from `Manifolds.jl`, so `Manifolds.jl` to be loaded.

!!! note

    The input of each of the (component) gradients is still the whole vector `x`,
    just that all other then the `i`th input component are assumed to be fixed and just
    the `i`th components gradient is computed / returned.

"""
function alternating_gradient_descent(
    M::ProductManifold, F, gradF::Union{TgF,AbstractVector{<:TgF}}, x; kwargs...
) where {TgF}
    x_res = copy(M, x)
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
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    inner_iterations::Int=5,
    stopping_criterion::StoppingCriterion=StopAfterIteration(100) |
                                          StopWhenGradientNormLess(1e-9),
    stepsize::Stepsize=ArmijoLinesearch(M),
    order_type::Symbol=:Linear,
    order=collect(1:(gradF isa Function ? length(gradF(M, x)) : length(gradF))),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    return_options=false,
    kwargs...,
) where {TgF}
    p = AlternatingGradientProblem(M, F, gradF; evaluation=evaluation)
    o = AlternatingGradientDescentOptions(
        M,
        x;
        inner_iterations=inner_iterations,
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
    ::AlternatingGradientProblem, o::AlternatingGradientDescentOptions
)
    o.k = 1
    o.i = 1
    (o.order_type == :FixedRandom || o.order_type == :Random) && (shuffle!(o.order))
    return o
end
function step_solver!(
    p::AlternatingGradientProblem, o::AlternatingGradientDescentOptions, iter
)
    s, o.gradient = o.direction(p, o, iter)
    j = o.order[o.k]
    retract!(p.M[j], o.x[p.M, j], o.x[p.M, j], -s * o.gradient[p.M, j])
    o.i += 1
    if o.i > o.inner_iterations
        o.k = ((o.k) % length(o.order)) + 1
        (o.order_type == :Random) && (shuffle!(o.order))
        o.i = 1
    end
    return o
end
get_solver_result(o::AlternatingGradientDescentOptions) = o.x
