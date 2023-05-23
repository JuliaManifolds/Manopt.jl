function default_stepsize(
    M::AbstractManifold,
    ::Type{<:ConjugateGradientDescentState};
    retraction_method=default_retraction_method(M),
)
    # take a default with a slightly defensive initial step size.
    return ArmijoLinesearch(M; retraction_method=retraction_method, initial_stepsize=1.0)
end
function show(io::IO, cgds::ConjugateGradientDescentState)
    i = get_count(cgds, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(cgds.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Conjugate Gradient Descent Solver
    $Iter
    ## Parameters
    * conjugate gradient coefficient: $(cgds.coefficient) (last β=$(cgds.β))
    * retraction method: $(cgds.retraction_method)
    * vector transport method: $(cgds.vector_transport_method)

    ## Stepsize
    $(cgds.stepsize)

    ## Stopping Criterion
    $(status_summary(cgds.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    conjugate_gradient_descent(M, F, gradF, p=rand(M))
    conjugate_gradient_descent(M, gradient_objective, p)

perform a conjugate gradient based descent

````math
p_{k+1} = \operatorname{retr}_{p_k} \bigl( s_kδ_k \bigr),
````

where ``\operatorname{retr}`` denotes a retraction on the `Manifold` `M`
and one can employ different rules to update the descent direction ``δ_k`` based on
the last direction ``δ_{k-1}`` and both gradients ``\operatorname{grad}f(x_k)``,``\operatorname{grad}f(x_{k-1})``.
The [`Stepsize`](@ref) ``s_k`` may be determined by a [`Linesearch`](@ref).

Alternatively to `f` and `grad_f` you can prodive
the [`AbstractManifoldGradientObjective`](@ref) `gradient_objective` directly.

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

If you provide the [`ManifoldGradientObjective`](@ref) directly, `evaluation` is ignored.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details
"""
conjugate_gradient_descent(M::AbstractManifold, args...; kwargs...)
function conjugate_gradient_descent(M::AbstractManifold, f, grad_f; kwargs...)
    return conjugate_gradient_descent(M, f, grad_f, rand(M); kwargs...)
end
function conjugate_gradient_descent(
    M::AbstractManifold, f::TF, grad_f::TDF, p; evaluation=AllocatingEvaluation(), kwargs...
) where {TF,TDF}
    mgo = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    return conjugate_gradient_descent(M, mgo, p; evaluation=evaluation, kwargs...)
end
function conjugate_gradient_descent(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TDF}
    # redefine our initial point
    q = [p]
    f_(M, p) = f(M, p[])
    grad_f_ = _to_mutating_gradient(grad_f, evaluation)
    rs = conjugate_gradient_descent(M, f_, grad_f_, q; evaluation=evaluation, kwargs...)
    #return just a number if  the return type is the same as the type of q
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function conjugate_gradient_descent(
    M::AbstractManifold, mgo::O, p=rand(M); kwargs...
) where {O<:Union{ManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return conjugate_gradient_descent!(M, mgo, q; kwargs...)
end

@doc raw"""
    conjugate_gradient_descent!(M, F, gradF, x)
    conjugate_gradient_descent!(M, gradient_objective, p; kwargs...)

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

Alternatively to `f` and `grad_f` you can prodive
the [`AbstractManifoldGradientObjective`](@ref) `gradient_objective` directly.

for more details and options, especially the [`DirectionUpdateRule`](@ref)s,
 see [`conjugate_gradient_descent`](@ref).
"""
conjugate_gradient_descent!(M::AbstractManifold, params...; kwargs...)
function conjugate_gradient_descent!(
    M::AbstractManifold,
    f::TF,
    grad_f::TDF,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
) where {TF,TDF}
    mgo = ManifoldGradientObjective(f, grad_f; evaluation=evaluation)
    dmgo = decorate_objective!(M, mgo; kwargs...)
    return conjugate_gradient_descent!(M, dmgo, p; kwargs...)
end
function conjugate_gradient_descent!(
    M::AbstractManifold,
    mgo::O,
    p;
    coefficient::DirectionUpdateRule=ConjugateDescentCoefficient(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    stepsize::Stepsize=default_stepsize(
        M, ConjugateGradientDescentState; retraction_method=retraction_method
    ),
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(500), StopWhenGradientNormLess(10^(-8))
    ),
    vector_transport_method=default_vector_transport_method(M, typeof(p)),
    kwargs...,
) where {O<:Union{ManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    dmgo = decorate_objective!(M, mgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    X = zero_vector(M, p)
    cgs = ConjugateGradientDescentState(
        M,
        p,
        stopping_criterion,
        stepsize,
        coefficient,
        retraction_method,
        vector_transport_method,
        X,
    )
    dcgs = decorate_state!(cgs; kwargs...)
    solve!(dmp, dcgs)
    return get_solver_return(get_objective(dmp), dcgs)
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
    copyto!(M, cgs.p_old, cgs.p)
    current_stepsize = get_stepsize(amp, cgs, i, cgs.δ)
    retract!(M, cgs.p, cgs.p, cgs.δ, current_stepsize, cgs.retraction_method)
    get_gradient!(amp, cgs.X, cgs.p)
    cgs.β = cgs.coefficient(amp, cgs, i)
    vector_transport_to!(M, cgs.δ, cgs.p_old, cgs.δ, cgs.p, cgs.vector_transport_method)
    cgs.δ .*= cgs.β
    cgs.δ .-= cgs.X
    return cgs
end
