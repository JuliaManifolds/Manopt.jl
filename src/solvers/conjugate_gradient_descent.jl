function default_stepsize(
    M::AbstractManifold,
    ::Type{<:ConjugateGradientDescentState};
    retraction_method=default_retraction_method(M),
)
    # take a default with a slightly defensive initial step size.
    return ArmijoLinesearch(M; retraction_method=retraction_method, initial_stepsize=1.0)
end
@doc raw"""
    conjugate_gradient_descent(M, F, gradF, x)

perform a conjugate gradient based descent

````math
p_{k+1} = \operatorname{retr}_{p_k} \bigl( s_kδ_k \bigr),
````

where ``\operatorname{retr}`` denotes a retraction on the `Manifold` `M`
and one can employ different rules to update the descent direction ``δ_k`` based on
the last direction ``δ_{k-1}`` and both gradients ``\operatorname{grad}f(x_k)``,``\operatorname{grad}f(x_{k-1})``.
The [`Stepsize`](@ref) ``s_k`` may be determined by a [`Linesearch`](@ref).

Available update rules are [`SteepestDirectionUpdateRule`](@ref), which yields a [`gradient_descent`](@ref),
[`ConjugateDescentCoefficient`](@ref) (the default), [`DaiYuanCoefficient`](@ref), [`FletcherReevesCoefficient`](@ref),
[`HagerZhangCoefficient`](@ref), [`HestenesStiefelCoefficient`](@ref),
[`LiuStoreyCoefficient`](@ref), and [`PolakRibiereCoefficient`](@ref).
These can all be combined with a [`ConjugateGradientBealeRestart`](@ref) rule.

They all compute ``β_k`` such that this algorithm updates the search direction as
````math
\delta_k=\operatorname{grad}f(p_k) + β_k \delta_{k-1}
````

# Input
* `M` : a manifold ``\mathcal M``
* `f` : a cost function ``F:\mathcal M→ℝ`` to minimize implemented as a function `(M,p) -> v`
* `grad_f`: the gradient ``\operatorname{grad}F:\mathcal M → T\mathcal M`` of ``F`` implemented also as `(M,x) -> X`
* `p` : an initial value ``x∈\mathcal M``

# Optional
* `coefficient` : ([`ConjugateDescentCoefficient`](@ref) `<:` [`DirectionUpdateRule`](@ref))
  rule to compute the descent direction update coefficient ``β_k``,
  as a functor, i.e. the resulting function maps `(amp, cgs, i) -> β`, where
  `amp` is an [`AbstractManoptProblem`](@ref), `cgs` are the
  [`ConjugateGradientDescentState`](@ref) `o` and `i` is the current iterate.
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `gradF(M, x)`
  or [`InplaceEvaluation`](@ref) in place, i.e. is of the form `gradF!(M, X, x)`.
* `retraction_method` - (`default_retraction_method(M, typeof(p))`) a retraction method to use.
* `stepsize` - ([`ArmijoLinesearch`](@ref) via [`default_stepsize`](@ref)) A [`Stepsize`](@ref) function applied to the
  search direction. The default is a constant step size 1.
* `stopping_criterion` : (`stopWhenAny( stopAtIteration(200), stopGradientNormLess(10.0^-8))`)
  a function indicating when to stop.
* `vector_transport_method` – (`default_vector_transport_method(M, typeof(p))`) vector transport method to transport
  the old descent direction when computing the new descent direction.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details
"""
function conjugate_gradient_descent(
    M::AbstractManifold, F::TF, gradF::TDF, x; kwargs...
) where {TF,TDF}
    x_res = copy(M, x)
    return conjugate_gradient_descent!(M, F, gradF, x_res; kwargs...)
end
@doc raw"""
    conjugate_gradient_descent!(M, F, gradF, x)

perform a conjugate gradient based descent in place of `x`, i.e.
````math
p_{k+1} = \operatorname{retr}_{p_k} \bigl( s_k\delta_k \bigr),
````
where ``\operatorname{retr}`` denotes a retraction on the `Manifold` `M`

# Input
* `M` : a manifold ``\mathcal M``
* `f` : a cost function ``F:\mathcal M→ℝ`` to minimize
* `grad_f`: the gradient ``\operatorname{grad}F:\mathcal M→ T\mathcal M`` of F
* `p` : an initial value ``p∈\mathcal M``

for more details and options, especially the [`DirectionUpdateRule`](@ref)s,
 see [`conjugate_gradient_descent`](@ref).
"""
function conjugate_gradient_descent!(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    p;
    coefficient::DirectionUpdateRule=ConjugateDescentCoefficient(),
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    stepsize::Stepsize=default_stepsize(
        M, ConjugateGradientDescentState; retraction_method=retraction_method
    ),
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(500), StopWhenGradientNormLess(10^(-8))
    ),
    vector_transport_method=default_vector_transport_method(M, typeof(p)),
    kwargs...,
) where {TF,TDF}
    mgo = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    dmgo = decorate_objective!(M, mgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    X = zero_vector(M, p)
    cgs = ConjugateGradientDescentState(
        M,
        p,
        stopping_criterion,
        stepsize,
        DirectionUpdateRuleStorage(M, coefficient, p, X),
        retraction_method,
        vector_transport_method,
        X,
    )
    cgs = decorate_state!(cgs; kwargs...)
    return get_solver_return(solve!(dmp, cgs))
end
function initialize_solver!(amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState)
    cgs.X = get_gradient(amp, cgs.p)
    cgs.δ = -copy(get_manifold(amp), cgs.p, cgs.X)
    # remember the first gradient in coefficient calculation
    cgs.coefficient(amp, cgs, 0)
    cgs.β = 0.0
    return cgs
end
function step_solver!(amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, i)
    M = get_manifold(amp)
    p_old = copy(M, cgs.p)
    current_stepsize = get_stepsize(amp, cgs, i, cgs.δ)
    retract!(M, cgs.p, cgs.p, cgs.δ, current_stepsize, cgs.retraction_method)
    get_gradient!(amp, cgs.X, cgs.p)
    cgs.β = cgs.coefficient(amp, cgs, i)
    vector_transport_to!(M, cgs.δ, p_old, cgs.δ, cgs.p, cgs.vector_transport_method)
    cgs.δ .*= cgs.β
    cgs.δ .-= cgs.X
    return cgs
end
