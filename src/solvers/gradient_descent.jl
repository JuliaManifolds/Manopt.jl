@doc raw"""
    gradient_descent(M, F, gradF, x)

perform a gradient descent

```math
x_{k+1} = \operatorname{retr}_{x_k}\bigl( s_k\operatorname{grad}f(x_k) \bigr)
```

with different choices of ``s_k`` available (see `stepsize` option below).

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F: \mathcal M→ℝ`` to minimize
* `gradF` – the gradient ``\operatorname{grad}F: \mathcal M → T\mathcal M`` of F
* `x` – an initial value ``x ∈ \mathcal M``

# Optional
* `direction` – [`IdentityUpdateRule`](@ref) perform a processing of the direction, e.g.
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `gradF(M, x)`
  or [`MutatingEvaluation`](@ref) in place, i.e. is of the form `gradF!(M, X, x)`.
* `retraction_method` – (`default_retraction_method(M)`) a `retraction(M,x,ξ)` to use.
* `return_options` – (`false`) – if activated, the extended result, i.e. the
    complete [`Options`](@ref) are returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `x_opt` if returned
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
  functor.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(200), `[`StopWhenGradientNormLess`](@ref)`(10.0^-8))`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `x_opt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function gradient_descent(
    M::AbstractManifold, F::TF, gradF::TDF, x; kwargs...
) where {TF,TDF}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return gradient_descent!(M, F, gradF, x_res; kwargs...)
end
@doc raw"""
    gradient_descent!(M, F, gradF, x)

perform a gradient_descent

```math
x_{k+1} = \operatorname{retr}_{x_k}\bigl( s_k\operatorname{grad}f(x_k) \bigr)
```

in place of `x` with different choices of ``s_k`` available.

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize
* `gradF` – the gradient ``\operatorname{grad}F:\mathcal M→ T\mathcal M`` of F
* `x` – an initial value ``x ∈ \mathcal M``

For more options, especially [`Stepsize`](@ref)s for ``s_k``, see [`gradient_descent`](@ref)
"""
function gradient_descent!(
    M::AbstractManifold,
    F::TF,
    gradF::TDF,
    x;
    stepsize::Stepsize=ConstantStepsize(1.0),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(10.0^-8),
    direction=IdentityUpdateRule(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    return_options=false,
    kwargs..., #collect rest
) where {TF,TDF}
    p = GradientProblem(M, F, gradF; evaluation=evaluation)
    o = GradientDescentOptions(
        M,
        x;
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
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
#
# Solver functions
#
function initialize_solver!(p::GradientProblem, o::GradientDescentOptions)
    o.gradient = get_gradient(p, o.x)
    return o
end
function step_solver!(p::GradientProblem, o::GradientDescentOptions, iter)
    s, o.gradient = o.direction(p, o, iter)
    retract!(p.M, o.x, o.x, -s * o.gradient, o.retraction_method)
    return o
end
get_solver_result(o::GradientDescentOptions) = o.x
