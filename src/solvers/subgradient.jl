@doc raw"""
    subgradient_method(M, F, ∂F, x)
perform a subgradient method $x_{k+1} = \mathrm{retr}(x_k, s_k∂F(x_k))$,

where $\mathrm{retr}$ is a retraction, $s_k$ can be specified as a function but is
usually set to a constant value. Though the subgradient might be set valued,
the argument `∂F` should always return _one_ element from the subgradient.

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F\colon\mathcal M\to\mathbb R$ to minimize
* `∂F`: the (sub)gradient $\partial F\colon\mathcal M\to T\mathcal M$ of F
  restricted to always only returning one value/element from the subgradient
* `x` – an initial value $x ∈ \mathcal M$

# Optional
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
* `retraction` – (`exp`) a `retraction(M,x,ξ)` to use.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(200), `[`StopWhenGradientNormLess`](@ref)`(10.0^-8))`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.
* `return_options` – (`false`) – if actiavated, the extended result, i.e. the
    complete [`Options`](@ref) re returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `xOpt` if returned
...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `xOpt` – the resulting (approximately critical) point of gradientDescent
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function subgradient_method(
    M::Manifold,
    F::TF,
    ∂F::TdF,
    x;
    retraction::TRetr = ExponentialRetraction(),
    stepsize::Stepsize = DecreasingStepsize(injectivity_radius(M, x) / 5),
    stopping_criterion::StoppingCriterion = StopAfterIteration(5000),
    return_options = false,
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr}
    p = SubGradientProblem(M, F, ∂F)
    o = SubGradientMethodOptions(M, x, stopping_criterion, stepsize, retraction)
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end
function initialize_solver!(p::SubGradientProblem, o::SubGradientMethodOptions)
    o.xOptimal = o.x
    return o.∂ = zero_tangent_vector(p.M, o.x)
end
function step_solver!(p::SubGradientProblem, o::SubGradientMethodOptions, iter)
    o.∂ = get_subgradient(p, o.x)
    s = get_stepsize(p, o, iter)
    retract!(p.M, o.x, o.x, -s * o.∂, o.retraction_method)
    if get_cost(p, o.x) < get_cost(p, o.xOptimal)
        o.xOptimal = o.x
    end
end
get_solver_result(o::SubGradientMethodOptions) = o.xOptimal
