@doc raw"""
    gradient_descent(M, F, ∇F, x)

perform a gradient_descent ``x_{k+1} = \mathrm{retr}_{x_k} s_k\nabla f(x_k)`` with
different choices of ``s_k`` available (see `stepsize` option below).

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F\colon\mathcal M\to\mathbb R`` to minimize
* `∇F` – the gradient ``\nabla F\colon\mathcal M\to T\mathcal M`` of F
* `x` – an initial value ``x ∈ \mathcal M``

# Optional
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
  functor.
* `retraction_method` – (`ExponentialRetraction()`) a `retraction(M,x,ξ)` to use.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(200), `[`StopWhenGradientNormLess`](@ref)`(10.0^-8))`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `return_options` – (`false`) – if activated, the extended result, i.e. the
    complete [`Options`](@ref) are returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `x_opt` if returned
...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `x_opt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function gradient_descent(M::Manifold, F::TF, ∇F::TDF, x; kwargs...) where {TF,TDF}
    x_res = allocate(x)
    copyto!(x_res, x)
    return gradient_descent!(M, F, ∇F, x_res; kwargs...)
end
@doc raw"""
    gradient_descent!(M, F, ∇F, x)

perform a gradient_descent ``x_{k+1} = \mathrm{retr}_{x_k} s_k\nabla f(x_k)`` inplace of `x`
with different choices of ``s_k`` available.

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F\colon\mathcal M\to\mathbb R`` to minimize
* `∇F` – the gradient ``\nabla F\colon\mathcal M\to T\mathcal M`` of F
* `x` – an initial value ``x ∈ \mathcal M``

For more options, especially [`Stepsize`](@ref)s for ``s_k``, see [`gradient_descent`](@ref)
"""
function gradient_descent!(
    M::Manifold,
    F::TF,
    ∇F::TDF,
    x;
    stepsize::Stepsize=ConstantStepsize(1.0),
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(200), StopWhenGradientNormLess(10.0^-8)
    ),
    direction=IdentityUpdateRule(),
    return_options=false,
    kwargs..., #collect rest
) where {TF,TDF}
    p = GradientProblem(M, F, ∇F)
    o = GradientDescentOptions(
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
function initialize_solver!(p::P, o::O) where {P<:GradientProblem,O<:GradientDescentOptions}
    o.∇ = get_gradient(p, o.x)
    return o
end
function step_solver!(p::P, o::O, iter) where {P<:GradientProblem,O<:GradientDescentOptions}
    s, o.∇ = o.direction(p, o, iter)
    o.x = retract(p.M, o.x, -s .* o.∇, o.retraction_method)
    return o
end
get_solver_result(o::GradientDescentOptions) = o.x
