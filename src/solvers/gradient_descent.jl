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
  or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `gradF!(M, X, x)`.
* `retraction_method` – (`default_retraction_method(M)`) a `retraction(M,x,ξ)` to use.
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
  functor.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(200), `[`StopWhenGradientNormLess`](@ref)`(10.0^-8))`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
...
and the ones that are passed to [`decorate_state`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details
"""
function gradient_descent(
    M::AbstractManifold, F::TF, gradF::TDF, x; kwargs...
) where {TF,TDF}
    x_res = copy(M, x)
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
    stepsize::Stepsize=ConstantStepsize(M),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(10.0^-8),
    debug=[DebugWarnIfCostIncreases()],
    direction=IdentityUpdateRule(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs..., #collect rest
) where {TF,TDF}
    p = GradientProblem(M, F, gradF; evaluation=evaluation)
    o = GradientDescentState(
        M,
        x;
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        direction=direction,
        retraction_method=retraction_method,
    )
    o = decorate_state(o; debug=debug, kwargs...)
    return get_solver_return(solve!(p, o))
end
#
# Solver functions
#
function initialize_solver!(p::AbstractManoptProblem, s::GradientDescentState)
    s.gradient = get_gradient(p, s.x)
    return s
end
function step_solver!(p::AbstractManoptProblem, s::GradientDescentState, iter)
    step, s.gradient = s.direction(p, s, iter)
    retract!(p.M, s.x, s.x, -step * s.gradient, s.retraction_method)
    return s
end
