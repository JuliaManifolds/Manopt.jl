# TODO: Update keywords in docs
_doc_LM = """
    LevenbergMarquardt(M, f, jacobian_f, p, num_components=-1; kwargs...)
    LevenbergMarquardt(M, vgf, p; kwargs...)
    LevenbergMarquardt(M, nlso, p; kwargs...)
    LevenbergMarquardt!(M, f, jacobian_f, p, num_components=-1; kwargs...)
    LevenbergMarquardt!(M, vgf, p; kwargs...)
    LevenbergMarquardt!(M, nlso, p; kwargs...)

compute the the Riemannian Levenberg-Marquardt algorithm [Peeters:1993, AdachiOkunoTakeda:2022, BaranBergmann:2026](@cite)
to solve

$(_problem(:NonLinearLeastSquares))

The second block of signatures perform the optimization in-place of `p`.

The regularization parameter is updated using a generalized scheme proposed in [Fan:2006](@cite),
Eq. (2.2). See also [Yuan:2015](@cite) for other schemes.
The generalized scheme offers separate thresholds for the acceptance of new points (`candidate_acceptance_threshold`),
decreasing the regularization parameter (`damping_reduction_threshold`) and increasing
the regularization parameter (`damping_increase_threshold`).

# Input

$(_args(:M))
* `f`: a residual function ``f: $(_math(:Manifold))→ℝ^m``.
  The residual function can be provided in two different ways
    * as a single function returning a vector ``f(p) ∈ ℝ^m``
    * as a vector of functions, where each single function returns a scalar ``f_i(p) ∈ ℝ``
  The type is determined by the `function_type=` keyword argument.
* `jacobian_f`:   the Jacobian of ``f``.
  The Jacobian can be provided in three different ways
  * as a single function returning a vector of gradient vectors ``$(_tex(:bigl))($(_tex(:grad)) f_i(p)$(_tex(:bigr)))_{i=1}^m``
  * as a vector of functions, where each single function returns a gradient vector ``$(_tex(:grad)) f_i(p)``, ``i=1,…,m``
  * as a single function returning a (coefficient) matrix ``J ∈ ℝ^{m×d}``, where ``d`` is the dimension of the manifold.
  These coefficients are given with respect to an [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) of the tangent space at `p`.
  The type is determined by the `jacobian_type=` keyword argument.
$(_args(:p))
* `num_components`: length ``m`` of the vector returned by the cost function.
  By default its value is -1 which means that it is determined automatically by
  calling `f` one additional time. This is only possible when `evaluation` is [`AllocatingEvaluation`](@ref),
  for mutating evaluation this value must be explicitly specified.

You can also provide the cost and its Jacobian already as a [`VectorGradientFunction`](@ref) `vgf` to indicate you are working on a single block,
Alternatively, passing a [`ManifoldNonlinearLeastSquaresObjective`](@ref) `nlso` also works.

# Keyword arguments

If you provide `f` and its jacobian

$(_kwargs(:evaluation))
* `function_type=`[`FunctionVectorialType`](@ref): an [`AbstractVectorialType`](@ref) specifying the type of cost function provided.
* `jacobian_type=`[`FunctionVectorialType`](@ref): an [`AbstractVectorialType`](@ref) specifying the type of Jacobian provided.
$(_kwargs(:evaluation))

as well as then these are already combined in a single [`VectorGradientFunction`](@ref) `vgf`

* `robustifier::`[`AbstractRobustifierFunction`](@ref)` = `[`IdentityRobustifier`](@ref)`()`:
  for the robust variant, specify how the robustification is meant to take place.
  - if you provide a single vectorial function and its Jacobian, a single robustifer is applied
    to every component function of this vectorial function (each component is a block in the sum)
  - if you provide a vector of [`VectorGradientFunction`](@ref)s, each needs a robustifier.
$(_kwargs(:evaluation))

as well as in general using the model imprevement parameter ``m_k`` in several places, cf [BaranBergmann:2026](@cite)

* `candidate_acceptance_threshold=0.2`: sufficient model improvement ``η ∈ (0,1)``, i.e. ``m_k > η`` to accept a candidate point
* `damping_increase_factor=5.0`:        factor ``β_{$(_tex(:text, "i"))}`` to increase damping, when the model is inaccurate
* `damping_increase_threshold=candidate_acceptance_threshold`: threshold ``η_{$(_tex(:text, "l"))}`` the value ``m_k``has to be below to increase damping.
  The default yields, that we increase damping when we reject a candidate.
* `damping_reduction_factor= 1 / damping_increase_factor`: factor ``β_{$(_tex(:text, "d"))}`` to reduce damping, when the model is accurate
* `damping_reduction_threshold=Inf`:    threshold ``β_{$(_tex(:text, "d"))}`` to reduce damping, when the model is accurate
  The default means, that we never reduce damping.
* `damping_term_min = 0.1`:             lower bound ``μ_{$(_tex(:text, "l"))}`` for the damping ``μ_k`` throughout the iterations
* `damping_term_max = Inf`:             upper bound ``μ_{$(_tex(:text, "u"))}`` for the damping ``μ_k`` throughout the iterations
* `initial_damping_term=damping_term_min`: initial damping ``μ_0``
* `initial_residual_values = zeros(m)`: a cache for the vector of residuals, `m` is the number of residual blocks
* `initial_jacobian_matrices`: a cache for the evaluated Jacobians (currently only used if `use_unified_basis = true`, then initialised to a vector of jacobian matrices, otherwise ignored)
$(_kwargs(:retraction_method))
* `scaling_threshold = 1.0e-6`:         a threshold `ε` to bound the scaling parameter `α` in the robust case away from `1`, see [`get_LevenbergMarquardt_scaling`](@ref)
* `scaling_mode = :Default`:            specify the scaling stabilization mode, see [`get_LevenbergMarquardt_scaling`](@ref)
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(500)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1.0e-12)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1.0e-12)"))
* `sub_objective`                      : specify the objective for the surrogate sub problem to solver in every iteration.
  This is set depending on the `use_unified_basis`
  - if `true` to the [`LevenbergMarquardtLinearSurrogateCoordinatesObjective`](@ref) which always works in coordinates of one single basis per tangent space and allows to cache Jacobian evaluations
  - if `false` to the [`LevenbergMarquardtLinearSurrogateObjective`](@ref) that can work either with linear operators or in (even different) coordinates.

  This keyword is ignored if you set the `sub_problem` and/or `sub_state` keyword directly
* `sub_problem = `[`DefaultManoptProblem`](@ref)`(`$(_link(:TangentSpace))`(M, p), sub_objective)`: specify the sub problem to be solved. This should usually be phrased on the tangent space at the current iterate
* `sub_state = `[`ConjugateResidualState`](@ref)`(`$(_link(:TangentSpace))`(M, p), sub_objective)`: specify the solver for the surrogate, see also [`conjugate_residual`](@ref)
* `use_unified_basis = false`:           specify to use a single basis for all Jacobian evaluations at a certain iterate, see `sub_objective`
  this requires that all Jacobians involved are of tupe [`CoefficientVectorialType`](@ref), since only then a jacobian can be represented as a matrix,
  and then here unified in the sense that all use the same basis.
$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_LM)"
LevenbergMarquardt(M::AbstractManifold, args...; kwargs...)
function LevenbergMarquardt(
        M::AbstractManifold, f, jacobian_f, num_components::Int = -1;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...,
    )
    return LevenbergMarquardt(
        M, f, jacobian_f, rand(M), num_components; evaluation = evaluation, kwargs...
    )
end
function LevenbergMarquardt(
        M::AbstractManifold, f, jacobian_f, p, num_components::Int = -1;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        function_type::AbstractVectorialType = FunctionVectorialType(),
        jacobian_type::AbstractVectorialType = CoefficientVectorialType(DefaultOrthonormalBasis()),
        kwargs...,
    )
    if num_components == -1
        if evaluation === AllocatingEvaluation()
            num_components = length(f(M, p))
        else
            throw(
                ArgumentError("For mutating evaluation num_components needs to be explicitly specified"),
            )
        end
    end
    vgf = VectorGradientFunction(
        f, jacobian_f, num_components;
        evaluation = evaluation, function_type = function_type, jacobian_type = jacobian_type,
    )
    return LevenbergMarquardt(M, vgf, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt(
        M::AbstractManifold, vgf::Union{VectorGradientFunction, VectorDifferentialFunction}, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        robustifier::AbstractRobustifierFunction = IdentityRobustifier(), kwargs...,
    )
    # For a single vector gradient function, we always treat robustification componentwise
    nlso = ManifoldNonlinearLeastSquaresObjective(vgf, ComponentwiseRobustifierFunction(robustifier))
    return LevenbergMarquardt(M, nlso, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt(
        M::AbstractManifold, vgf::Union{Vector{<:VectorGradientFunction}, Vector{<:VectorDifferentialFunction}}, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        robustifier::Vector{<:AbstractRobustifierFunction} = [IdentityRobustifier() for _ in 1:length(vgf)],
        kwargs...,
    )
    nlso = ManifoldNonlinearLeastSquaresObjective(vgf, robustifier)
    return LevenbergMarquardt(M, nlso, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt(
        M::AbstractManifold, nlso::O, p; kwargs...
    ) where {O <: Union{ManifoldNonlinearLeastSquaresObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(LevenbergMarquardt; kwargs...)
    q = copy(M, p)
    return LevenbergMarquardt!(M, nlso, q; kwargs...)
end
calls_with_kwargs(::typeof(LevenbergMarquardt)) = (LevenbergMarquardt!,)

function construct_lm_subobjective(use_fast_coordinate_subobjective::Bool, nlso, damping_term_min, threshold, mode, residuals, jacobian_matrices)
    if use_fast_coordinate_subobjective
        # If we just have one vector function, the jacobians in the following should be a [M,] of a single matrix
        # ...to make this a bit easier for a user, we also accept a matrix here and wrap it if necessary
        _jm = jacobian_matrices isa AbstractMatrix ? [jacobian_matrices] : jacobian_matrices
        return NormalEquationsObjective(
            LevenbergMarquardtLinearSurrogateCoordinatesObjective(
                nlso; penalty = damping_term_min, threshold = threshold, mode = mode,
                residuals = residuals, jacobian_cache = _jm,
            ),
        )
    else
        return NormalEquationsObjective(
            LevenbergMarquardtLinearSurrogateObjective(
                nlso; penalty = damping_term_min, threshold = threshold, mode = mode,
                residuals = residuals,
            ),
        )
    end
end

@doc "$(_doc_LM)"
LevenbergMarquardt!(M::AbstractManifold, args...; kwargs...)
function LevenbergMarquardt!(
        M::AbstractManifold, f, jacobian_f, p, num_components::Int = -1;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        jacobian_tangent_basis::AbstractBasis = default_basis(M, typeof(p)),
        jacobian_type::AbstractVectorialType = CoefficientVectorialType(jacobian_tangent_basis),
        function_type::AbstractVectorialType = FunctionVectorialType(), kwargs...,
    )
    if num_components == -1
        if evaluation === AllocatingEvaluation()
            num_components = length(f(M, p))
        else
            throw(
                ArgumentError("For mutating evaluation num_components needs to be explicitly specified"),
            )
        end
    end
    nlso = ManifoldNonlinearLeastSquaresObjective(
        f, jacobian_f, num_components;
        evaluation = evaluation, jacobian_type = jacobian_type, function_type = function_type,
    )
    return LevenbergMarquardt!(M, nlso, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt!(
        M::AbstractManifold, vgf::VectorGradientFunction, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        robustifier = IdentityRobustifier(),
        kwargs...,
    )
    nlso = ManifoldNonlinearLeastSquaresObjective(vgf, robustifier)
    return LevenbergMarquardt!(M, nlso, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt!(
        M::AbstractManifold, vgf::Vector{<:VectorGradientFunction}, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        robustifier::Vector{<:AbstractRobustifierFunction} = [IdentityRobustifier() for _ in 1:length(vgf)],
        kwargs...,
    )
    nlso = ManifoldNonlinearLeastSquaresObjective(vgf, robustifier)
    return LevenbergMarquardt!(M, nlso, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt!(
        M::AbstractManifold, nlso::O, p;
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        stopping_criterion::StoppingCriterion = StopAfterIteration(500) | StopWhenGradientNormLess(1.0e-12) | StopWhenStepsizeLess(1.0e-12),
        candidate_acceptance_threshold::Real = 0.2,
        damping_increase_factor::Real = 5.0,
        damping_increase_threshold::Real = candidate_acceptance_threshold,
        damping_reduction_threshold::Real = Inf,
        damping_reduction_factor::Real = 1 / damping_increase_factor,
        damping_term_min::Real = 0.1,
        damping_term_max::Real = Inf,
        initial_damping_term::Real = damping_term_min,
        debug = is_tutorial_mode() ? [DebugWarnIfCostIncreases()] : [],
        initial_residual_values = zeros(number_eltype(p), residuals_count(get_objective(nlso))),
        use_unified_basis::Bool = false,
        initial_jacobian_matrices = if use_unified_basis
            [Manopt.allocate_jacobian(M, vgf; T = eltype(p)) for vgf in get_objective(nlso).objective]
        else # one nothing per block
            fill(nothing, length(get_objective(nlso).objective))
        end,
        scaling_threshold::Real = 1.0e-6,
        scaling_mode::Symbol = :Default,
        minimum_acceptable_model_improvement::Real = eps(number_eltype(p)),
        sub_objective = construct_lm_subobjective(use_unified_basis, nlso, damping_term_min, scaling_threshold, scaling_mode, initial_residual_values, initial_jacobian_matrices),
        sub_problem = DefaultManoptProblem(TangentSpace(M, p), sub_objective),
        sub_state = ConjugateResidualState(TangentSpace(M, p), sub_objective),
        kwargs..., #collect rest
    ) where {O <: Union{ManifoldNonlinearLeastSquaresObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(LevenbergMarquardt!; kwargs...)
    dnlso = decorate_objective!(M, nlso; kwargs...)
    nlsp = DefaultManoptProblem(M, dnlso)
    sub_state_ = maybe_wrap_evaluation_type(sub_state)
    if has_anisotropic_max_stepsize(M)
        # This is how to recognize the box constraints as for example Hyperrectangle does
        sub_state_ = LevenbergMarquardtBoxSubsolver(M, sub_state_, p)
    end
    lms = LevenbergMarquardtState(
        M, sub_problem, sub_state_, initial_residual_values, initial_jacobian_matrices;
        p = p,
        damping_increase_factor = damping_increase_factor,
        damping_increase_threshold = damping_increase_threshold,
        damping_reduction_threshold = damping_reduction_threshold,
        damping_reduction_factor = damping_reduction_factor,
        candidate_acceptance_threshold = candidate_acceptance_threshold,
        damping_term = initial_damping_term,
        damping_term_min = damping_term_min, damping_term_max = damping_term_max,
        stopping_criterion = stopping_criterion,
        retraction_method = retraction_method,
        minimum_acceptable_model_improvement = minimum_acceptable_model_improvement,
    )
    dlms = decorate_state!(lms; debug = debug, kwargs...)
    solve!(nlsp, dlms)
    return get_solver_return(get_objective(nlsp), dlms)
end
calls_with_kwargs(::typeof(LevenbergMarquardt!)) = (decorate_objective!, decorate_state!)
#
# Solver functions
#
function initialize_solver!(
        dmp::DefaultManoptProblem, lms::LevenbergMarquardtState,
    )
    M = get_manifold(dmp)
    nlso = get_objective(dmp, true) # unwarp decorators
    get_residuals!(M, lms.residual_values, nlso, lms.p)
    for (o, jb) in zip(nlso.objective, lms.jacobian_matrices)
        !isnothing(jb) && get_jacobian!(M, jb, o, lms.p)
    end
    get_gradient!(M, lms.X, nlso, lms.p; value_cache = lms.residual_values, jacobian_cache = lms.jacobian_matrices)
    return lms
end

function step_solver!(
        dmp::DefaultManoptProblem,
        lms::LevenbergMarquardtState,
        ::Integer,
    )
    # Update damping term in the surrogate
    # should this be with (currently) or without robustifier?
    M = get_manifold(dmp)
    nlso = get_objective(dmp, true)
    FpSq = get_cost(M, nlso, lms.p)
    set_parameter!(lms.sub_problem, Val(:Objective), Val(:Penalty), lms.damping_term * FpSq)
    # update base point of the tangent space the subproblem works on
    set_parameter!(lms.sub_problem, Val(:Manifold), Val(:Basepoint), lms.p)
    # Subsolver result
    solve_LM_subproblem!(M, lms.direction, lms.p, lms.sub_problem, lms.sub_state, lms.X)
    #solve!(lms.sub_problem, lms.sub_state)
    #lms.direction .= -get_solver_result(lms.sub_problem, lms.sub_state)
    if norm(M, lms.p, lms.direction) > max_stepsize(M, lms.p)
        # Vector too long:
        # we reject the step without evaluating the objective
        # and increase damping
        lms.damping_term *= lms.damping_increase_factor
        return lms
    end
    model_improvement = (get_cost(lms.sub_problem, ZeroTangentVector()) - get_cost(lms.sub_problem, lms.direction)) / 2
    if model_improvement < lms.minimum_acceptable_model_improvement
        # Model improvement insufficient, reject step and increase damping term
        lms.damping_term *= lms.damping_increase_factor
        lms.damping_term = min(lms.damping_term, lms.damping_term_max)
        return lms
    end
    # New iterate candidate
    retract!(M, lms.q, lms.p, lms.direction, lms.retraction_method)

    # Evaluate improvement of actual cost divided by predicted cost improvement
    cost_improvement = get_cost(M, nlso, lms.p) - get_cost(M, nlso, lms.q)
    ρ = cost_improvement / model_improvement
    # Update damping term and iterate
    if ρ >= lms.damping_reduction_threshold
        # very good match between model and actual cost: decrease damping term
        lms.damping_term *= lms.damping_reduction_factor
        lms.damping_term = max(lms.damping_term, lms.damping_term_min)
    elseif ρ < lms.damping_increase_threshold
        # poor match between model and actual cost: increase damping term
        lms.damping_term *= lms.damping_increase_factor
        lms.damping_term = min(lms.damping_term, lms.damping_term_max)
    end
    if ρ >= lms.candidate_acceptance_threshold # enough improvement: accept candidate
        copyto!(M, lms.p, lms.q)
        get_residuals!(M, lms.residual_values, nlso, lms.p)
        for (o, jb) in zip(nlso.objective, lms.jacobian_matrices)
            !isnothing(jb) && get_jacobian!(M, jb, o, lms.p)
        end
        get_gradient!(M, lms.X, nlso, lms.p; value_cache = lms.residual_values, jacobian_cache = lms.jacobian_matrices)
    end
    return lms
end

function solve_LM_subproblem!(
        M::AbstractManifold, X, p, problem::P, state::S, grad_Y,
    ) where {P <: AbstractManoptProblem, S <: AbstractManoptSolverState}
    solve!(problem, state)
    copyto!(M, X, p, get_solver_result(problem, state))
    X .*= -1
    return X
end
# We could add “fully” closed form solvers via dispatch here as well

#
#
# Special cases for

function get_last_stepsize(
        dmp::DefaultManoptProblem, lms::LevenbergMarquardtState, k,
    )
    M = get_manifold(dmp)
    return norm(M, lms.p, lms.direction)
end
