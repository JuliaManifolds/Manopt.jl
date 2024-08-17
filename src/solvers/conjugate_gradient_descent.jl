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

    ## Stopping criterion

    $(status_summary(cgds.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

_doc_CG_formula = raw"""
````math
p_{k+1} = \operatorname{retr}_{p_k} \bigl( s_kδ_k \bigr),
````
"""
_doc_update_delta_k = raw"""
````math
\delta_k=\operatorname{grad}f(p_k) + β_k \delta_{k-1}
````
"""

_doc_CG = """
    conjugate_gradient_descent(M, f, grad_f, p=rand(M))
    conjugate_gradient_descent!(M, f, grad_f, p)
    conjugate_gradient_descent(M, gradient_objective, p)
    conjugate_gradient_descent!(M, gradient_objective, p; kwargs...)

perform a conjugate gradient based descent-

$(_doc_CG_formula)

where ``$(_l_retr)`` denotes a retraction on the `Manifold` `M`
and one can employ different rules to update the descent direction ``δ_k`` based on
the last direction ``δ_{k-1}`` and both gradients ``$(_l_grad)f(x_k)``,``$(_l_grad) f(x_{k-1})``.
The [`Stepsize`](@ref) ``s_k`` may be determined by a [`Linesearch`](@ref).

Alternatively to `f` and `grad_f` you can provide
the [`AbstractManifoldGradientObjective`](@ref) `gradient_objective` directly.

Available update rules are [`SteepestDirectionUpdateRule`](@ref), which yields a [`gradient_descent`](@ref),
[`ConjugateDescentCoefficient`](@ref) (the default), [`DaiYuanCoefficientRule`](@ref), [`FletcherReevesCoefficient`](@ref),
[`HagerZhangCoefficient`](@ref), [`HestenesStiefelCoefficient`](@ref),
[`LiuStoreyCoefficient`](@ref), and [`PolakRibiereCoefficient`](@ref).
These can all be combined with a [`ConjugateGradientBealeRestart`](@ref) rule.

They all compute ``β_k`` such that this algorithm updates the search direction as

$(_doc_update_delta_k)

# Input

$(_arg_M)
$(_arg_f)
$(_arg_grad_f)
$(_arg_p)

# Keyword arguments

* `coefficient::DirectionUpdateRule=[`ConjugateDescentCoefficient`](@ref)`()`:
  rule to compute the descent direction update coefficient ``β_k``, as a functor, where
  the resulting function maps are `(amp, cgs, k) -> β` with `amp` an [`AbstractManoptProblem`](@ref),
  `cgs` is the [`ConjugateGradientDescentState`](@ref), and `i` is the current iterate.
* $(_kw_evaluation_default): $(_kw_evaluation)
* $(_kw_retraction_method_default): $(_kw_retraction_method)
* `stepsize=[`ArmijoLinesearch`](@ref)`(M)`: $_kw_stepsize
  via [`default_stepsize`](@ref)) passing on the `default_retraction_method`
* `stopping_criterion=`[`StopAfterIteration`](@ref)`(500)`$(_sc_any)[`StopWhenGradientNormLess`](@ref)`(1e-8)`:
  $(_kw_stopping_criterion)
* $(_kw_vector_transport_method_default): $(_kw_vector_transport_method)

If you provide the [`ManifoldGradientObjective`](@ref) directly, the `evaluation=` keyword is ignored.
The decorations are still applied to the objective.

$(_doc_sec_output)
"""

@doc "$(_doc_CG)"
conjugate_gradient_descent(M::AbstractManifold, args...; kwargs...)
function conjugate_gradient_descent(M::AbstractManifold, f, grad_f; kwargs...)
    return conjugate_gradient_descent(M, f, grad_f, rand(M); kwargs...)
end
function conjugate_gradient_descent(
    M::AbstractManifold, f::TF, grad_f::TDF, p; evaluation=AllocatingEvaluation(), kwargs...
) where {TF,TDF}
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    grad_f_ = _ensure_mutating_gradient(grad_f, p, evaluation)
    mgo = ManifoldGradientObjective(f_, grad_f_; evaluation=evaluation)
    rs = conjugate_gradient_descent(M, mgo, p_; evaluation=evaluation, kwargs...)
    return _ensure_matching_output(p, rs)
end
function conjugate_gradient_descent(
    M::AbstractManifold, mgo::O, p=rand(M); kwargs...
) where {O<:Union{ManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return conjugate_gradient_descent!(M, mgo, q; kwargs...)
end

@doc "$(_doc_CG)"
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
    stopping_criterion::StoppingCriterion=StopAfterIteration(500) |
                                          StopWhenGradientNormLess(1e-8),
    vector_transport_method=default_vector_transport_method(M, typeof(p)),
    initial_gradient=zero_vector(M, p),
    kwargs...,
) where {O<:Union{ManifoldGradientObjective,AbstractDecoratedManifoldObjective}}
    dmgo = decorate_objective!(M, mgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    cgs = ConjugateGradientDescentState(
        M;
        p=p,
        stopping_criterion=stopping_criterion,
        stepsize=stepsize,
        coefficient=coefficient,
        retraction_method=retraction_method,
        vector_transport_method=vector_transport_method,
        initial_gradient=initial_gradient,
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
function step_solver!(amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, k)
    M = get_manifold(amp)
    copyto!(M, cgs.p_old, cgs.p)
    current_stepsize = get_stepsize(amp, cgs, k, cgs.δ)
    retract!(M, cgs.p, cgs.p, cgs.δ, current_stepsize, cgs.retraction_method)
    get_gradient!(amp, cgs.X, cgs.p)
    cgs.β = cgs.coefficient(amp, cgs, k)
    vector_transport_to!(M, cgs.δ, cgs.p_old, cgs.δ, cgs.p, cgs.vector_transport_method)
    cgs.δ .*= cgs.β
    cgs.δ .-= cgs.X
    return cgs
end
