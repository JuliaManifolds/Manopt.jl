@doc raw"""
    subgradient_method(M, F, ∂F, x)

perform a subgradient method ``x_{k+1} = \mathrm{retr}(x_k, s_k∂F(x_k))``,

where ``\mathrm{retr}`` is a retraction, ``s_k`` can be specified as a function but is
usually set to a constant value. Though the subgradient might be set valued,
the argument `∂F` should always return _one_ element from the subgradient, but
not necessarily determistic.

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F\colon\mathcal M→ℝ`` to minimize
* `∂F!!`– the (sub)gradient ``\partial F\colon\mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subgradient.
  This function can be passed as an allocation function `(M, y) -> X` or
  a mutating function `(M, X, y) -> X`, see `evaluation`.
* `x` – an initial value ``x ∈ \mathcal M``

# Optional
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the subgradient works by allocation (default) form `∂F(M, y)`
  or [`MutatingEvaluation`](@ref) in place, i.e. is of the form `∂F!(M, X, x)`.
* `stepsize` – ([`ConstantStepsize`](@ref)`(1.)`) specify a [`Stepsize`](@ref)
* `retraction` – (`exp`) a `retraction(M,x,ξ)` to use.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(5000)`)
  a functor, see[`StoppingCriterion`](@ref), indicating when to stop.
* `return_options` – (`false`) – if activated, the extended result, i.e. the
  complete [`Options`](@ref) re returned. This can be used to access recorded values.
  If set to false (default) just the optimal value `x_opt` if returned
...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `x_opt` – the resulting (approximately critical) point of the subgradient method
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function subgradient_method(M::Manifold, F::TF, ∂F::TdF, x; kwargs...) where {TF,TdF}
    x_res = deepcopy(x)
    return subgradient_method!(M, F, ∂F, x_res; kwargs...)
end
@doc raw"""
    subgradient_method!(M, F, ∂F, x)

perform a subgradient method ``x_{k+1} = \mathrm{retr}(x_k, s_k∂F(x_k))`` in place of `x`

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F\colon\mathcal M→ℝ`` to minimize
* `∂F!!`- the (sub)gradient ``\partial F\colon\mathcal M→ T\mathcal M`` of F
  restricted to always only returning one value/element from the subgradient.
  This function can be passed as an allocation function `(M, y) -> X` or
  a mutating function `(M, X, y) -> X`, see `evaluation`.
* `x` – an initial value ``x ∈ \mathcal M``

for more details and all optional parameters, see [`subgradient_method`](@ref).
"""
function subgradient_method!(
    M::Manifold,
    F::TF,
    ∂F!!::TdF,
    x;
    retraction::TRetr=ExponentialRetraction(),
    stepsize::Stepsize=ConstantStepsize(1.0),
    stopping_criterion::StoppingCriterion=StopAfterIteration(5000),
    return_options=false,
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs..., #especially may contain debug
) where {TF,TdF,TRetr}
    p = SubGradientProblem(M, F, ∂F!!; evaluation=evaluation)
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
    o.x_optimal = o.x
    o.∂ = zero_tangent_vector(p.M, o.x)
    return o
end
function step_solver!(p::SubGradientProblem, o::SubGradientMethodOptions, iter)
    get_subgradient!(p, o.∂, o.x)
    s = get_stepsize(p, o, iter)
    retract!(p.M, o.x, o.x, -s * o.∂, o.retraction_method)
    (get_cost(p, o.x) < get_cost(p, o.x_optimal)) && (o.x_optimal = o.x)
    return o
end
get_solver_result(o::SubGradientMethodOptions) = o.x_optimal
