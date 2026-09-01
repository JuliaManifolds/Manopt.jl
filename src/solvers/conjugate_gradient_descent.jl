function default_stepsize(
        M::AbstractManifold,
        ::Type{<:ConjugateGradientDescentState};
        retraction_method = default_retraction_method(M),
    )
    # take a default with a slightly defensive initial step size.
    return ArmijoLinesearch(
        M; retraction_method = retraction_method, initial_stepsize = 1.0
    )
end
function status_summary(cgds::ConjugateGradientDescentState; context::Symbol = :default)
    (context === :short) && (return repr(cgds))
    i = get_count(cgds, :Iterations)
    conv_inl = (i > 0) ? (has_converged(cgds.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the conjugate gradient descent solver$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(cgds.stop) ? "Yes" : "No"
    as = _callbacks_summary(cgds)
    return """
    # Solver state for `Manopt.jl`s Conjugate Gradient Descent Solver
    $Iter
    ## Parameters$(as)
    * conjugate gradient coefficient:$(_MANOPT_INDENT)$(cgds.coefficient) (last β=$(cgds.β))
    * restart condition:             $(_MANOPT_INDENT)$(cgds.restart_condition)
    * retraction method:             $(_MANOPT_INDENT)$(cgds.retraction_method)
    * vector transport method:       $(_MANOPT_INDENT)$(cgds.vector_transport_method)

    ## Stepsize
    $(_in_str(status_summary(cgds.stepsize; context = context); indent = 0, headers = 1))

    ## Stopping criterion
    $(_in_str(status_summary(cgds.stop; context = context); indent = 0, headers = 1))
    The algorithm converged: $Conv"""
end

_doc_CG_formula = raw"""
````math
p_{k+1} = \operatorname{retr}_{p_k} \bigl( s_kδ_k \bigr),
````
"""
_doc_update_delta_k = raw"""
````math
δ_k=\operatorname{grad}f(p_k) + β_k \delta_{k-1}
````
"""

_doc_CG = """
    conjugate_gradient_descent(M, f, grad_f, p=rand(M))
    conjugate_gradient_descent!(M, f, grad_f, p)
    conjugate_gradient_descent(M, gradient_objective, p)
    conjugate_gradient_descent!(M, gradient_objective, p; kwargs...)

perform a conjugate gradient based descent-

$(_doc_CG_formula)

where ``$(_tex(:retr))`` denotes a retraction on the `Manifold` `M`
and one can employ different rules to update the descent direction ``δ_k`` based on
the last direction ``δ_{k-1}`` and both gradients ``$(_tex(:grad))f(x_k)``,``$(_tex(:grad)) f(x_{k-1})``.
The [`Stepsize`](@ref) ``s_k`` may be determined by a [`Linesearch`](@ref).

Alternatively to `f` and `grad_f` you can provide
the [`AbstractManifoldFirstOrderObjective`](@ref) `gradient_objective` directly.

Available update rules are [`SteepestDescentCoefficientRule`](@ref), which yields a [`gradient_descent`](@ref),
[`ConjugateDescentCoefficient`](@ref) (the default), [`DaiYuanCoefficientRule`](@ref), [`FletcherReevesCoefficient`](@ref),
[`HagerZhangCoefficient`](@ref), [`HestenesStiefelCoefficient`](@ref),
[`LiuStoreyCoefficient`](@ref), and [`PolakRibiereCoefficient`](@ref).
These can all be combined with a [`ConjugateGradientBealeRestartRule`](@ref) rule.

They all compute ``β_k`` such that this algorithm updates the search direction as

$(_doc_update_delta_k)

# Input

$(_args([:M, :f, :grad_f, :p]))

# Keyword arguments

$(_kwargs(:callbacks; add_properties = [:process_note]))
* `coefficient::DirectionUpdateRule=`[`ConjugateDescentCoefficient`](@ref)`()`:
  rule to compute the descent direction update coefficient ``β_k``, as a functor, where
  the resulting function maps are `(amp, cgs, k) -> β` with `amp` an [`AbstractManoptProblem`](@ref),
  `cgs` is the [`ConjugateGradientDescentState`](@ref), and `k` is the current iterate.
$(_kwargs([:differential, :evaluation]))
* `restart_condition::AbstractRestartCondition=`[`RestartOnNonDescent`](@ref)`()`:
  rule when the algorithm should restart, i.e. use the negative gradient instead of the computed direction,
  as a functor where the resulting function maps are `(amp, cgs, k) -> corr::Bool` with `amp` an [`AbstractManoptProblem`](@ref),
  `cgs` is the [`ConjugateGradientDescentState`](@ref), and `k` is the current iterate.
$(_kwargs(:retraction_method))
$(_kwargs(:stepsize; default = "`[`ArmijoLinesearch`](@ref)`()"))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(500)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-8)"))
$(_kwargs(:vector_transport_method))

If you provide the [`ManifoldFirstOrderObjective`](@ref) directly, the `evaluation=` keyword is ignored.
The decorations are still applied to the objective.

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_CG)"
conjugate_gradient_descent(M::AbstractManifold, args...; kwargs...)
function conjugate_gradient_descent(M::AbstractManifold, f, grad_f; kwargs...)
    return conjugate_gradient_descent(M, f, grad_f, rand(M); kwargs...)
end
function conjugate_gradient_descent(
        M::AbstractManifold, f::TF, grad_f::TDF, p;
        differential = missing,
        evaluation = AllocatingEvaluation(),
        kwargs...,
    ) where {TF, TDF}
    p_ = maybe_wrap_variable(p)
    mgo = ManifoldGradientObjective(
        f, grad_f; differential = differential, evaluation = evaluation, p = p
    )
    rs = conjugate_gradient_descent(M, mgo, p_; evaluation = evaluation, kwargs...)
    return maybe_unwrap_variable(p, rs)
end
function conjugate_gradient_descent(
        M::AbstractManifold, mgo::O, p = rand(M); kwargs...
    ) where {O <: Union{AbstractManifoldFirstOrderObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(conjugate_gradient_descent; kwargs...)
    q = copy(M, p)
    return conjugate_gradient_descent!(M, mgo, q; kwargs...)
end
calls_with_kwargs(::typeof(conjugate_gradient_descent)) = (conjugate_gradient_descent!,)

@doc "$(_doc_CG)"
conjugate_gradient_descent!(M::AbstractManifold, params...; kwargs...)
function conjugate_gradient_descent!(
        M::AbstractManifold, f::TF, grad_f::TDF, p;
        differential = missing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    ) where {TF, TDF}
    mgo = ManifoldGradientObjective(
        f, grad_f; differential = differential, evaluation = evaluation, p = p
    )
    return conjugate_gradient_descent!(M, mgo, p; kwargs...)
end
function conjugate_gradient_descent!(
        M::AbstractManifold,
        mgo::O,
        p;
        callbacks = Dict{Symbol, Function}(),
        coefficient::Union{DirectionUpdateRule, ManifoldDefaultsFactory} = ConjugateDescentCoefficient(),
        restart_condition::AbstractRestartCondition = RestartOnNonDescent(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = default_stepsize(
            M, ConjugateGradientDescentState; retraction_method = retraction_method
        ),
        stopping_criterion::StoppingCriterion = StopAfterIteration(500) |
            StopWhenGradientNormLess(1.0e-8),
        vector_transport_method = default_vector_transport_method(M, typeof(p)),
        initial_gradient = zero_vector(M, p),
        kwargs...,
    ) where {O <: Union{AbstractManifoldFirstOrderObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(conjugate_gradient_descent!; kwargs...)
    dmgo = decorate_objective!(M, mgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    cgs = ConjugateGradientDescentState(
        M;
        callbacks = process_callbacks_arg(callbacks, ConjugateGradientDescentState),
        p = p,
        stopping_criterion = stopping_criterion,
        stepsize = _produce_type(stepsize, M, p),
        coefficient = _produce_type(coefficient, M, p),
        restart_condition = restart_condition,
        retraction_method = retraction_method,
        vector_transport_method = vector_transport_method,
        initial_gradient = initial_gradient,
    )
    dcgs = decorate_state!(cgs; kwargs...)
    solve!(dmp, dcgs)
    return get_solver_return(get_objective(dmp), dcgs)
end
calls_with_kwargs(::typeof(conjugate_gradient_descent!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState)
    cgs.X = get_gradient(amp, cgs.p)
    cgs.δ = -copy(get_manifold(amp), cgs.p, cgs.X)
    # remember the first gradient in coefficient calculation
    cgs.coefficient(amp, cgs, 0)
    initialize_stepsize!(cgs.stepsize)
    cgs.β = 0.0
    return cgs
end
function step_solver!(amp::AbstractManoptProblem, cgs::ConjugateGradientDescentState, k)
    M = get_manifold(amp)
    copyto!(M, cgs.p_old, cgs.p)
    current_stepsize = get_stepsize(amp, cgs, k, cgs.δ; gradient = cgs.X)
    callback(:Stepsize, amp, cgs, k)
    ManifoldsBase.retract_fused!(
        M, cgs.p, cgs.p, cgs.δ, current_stepsize, cgs.retraction_method
    )
    get_gradient!(amp, cgs.X, cgs.p)
    cgs.β = cgs.coefficient(amp, cgs, k)
    vector_transport_to!(M, cgs.δ, cgs.p_old, cgs.δ, cgs.p, cgs.vector_transport_method)
    cgs.δ .*= cgs.β
    cgs.δ .-= cgs.X
    if (cgs.restart_condition(amp, cgs, k))
        # restart solver; set dir to -grad
        copyto!(M, cgs.δ, cgs.X)
        cgs.δ .*= -1
        update_storage!(cgs.coefficient.storage, amp, cgs)
        cgs.β = 0.0
    end
    return cgs
end
