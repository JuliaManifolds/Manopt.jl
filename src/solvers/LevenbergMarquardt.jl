# TODO: Update keywords in docs
_doc_LM = """
    LevenbergMarquardt(M, f, jacobian_f, p, num_components=-1; kwargs...)
    LevenbergMarquardt(M, vgf, p; kwargs...)
    LevenbergMarquardt(M, nlso, p; kwargs...)
    LevenbergMarquardt!(M, f, jacobian_f, p, num_components=-1; kwargs...)
    LevenbergMarquardt!(M, vgf, p, num_components=-1; kwargs...)
    LevenbergMarquardt!(M, nlso, p, num_components=-1; kwargs...)

compute the the Riemannian Levenberg-Marquardt algorithm [Peeters:1993, AdachiOkunoTakeda:2022, BaranBergmann:2026](@cite)
to solve

$(_problem(:NonLinearLeastSquares))

The second block of signatures perform the optimization in-place of `p`.

The regularization parameter is updated using a generalized scheme proposed in [Yuan:2015, Fan:2006](@cite)
which offers separate thresholds for the acceptance of new points (`η`) and decreasing the
regularization parameter (`damping_reduction_threshold`).

# Input

$(_args(:M))
* `f`: a cost function ``f: $(_math(:Manifold))→ℝ^m``.
  The cost function can be provided in two different ways
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

You can also provide the cost and its Jacobian already as a[`VectorGradientFunction`](@ref) `vgf`,
Alternatively, passing a [`NonlinearLeastSquaresObjective`](@ref) `nlso`.

# Keyword arguments

$(_kwargs(:evaluation))
* `η=0.2`:                   scaling factor for the sufficient cost decrease threshold required to accept new proposal points. Allowed range: `0 < η < 1`.
* `damping_term_min=0.1`:      initial (and also minimal) value of the damping term
* `β=5.0`:                     parameter by which the damping term is multiplied when the current new point is rejected
* `function_type=`[`FunctionVectorialType`](@ref): an [`AbstractVectorialType`](@ref) specifying the type of cost function provided.
* `initial_jacobian_f`:      the list of initial Jacobians of each block of the cost function `f`.
  By default this is a matrix of size `num_components` times the manifold dimension of similar type as `p`.
* `initial_residual_values`: the initial residual vector of the cost function `f`.
  By default this is a vector of length `num_components` of similar type as `p`.
* `jacobian_type=`[`FunctionVectorialType`](@ref): an [`AbstractVectorialType`](@ref) specifying the type of Jacobian provided.
* `sub_evaluation = `[`InplaceEvaluation`](@ref): an [`AbstractEvaluationType`](@ref) for `linear_subsolver!`.
* `linear_subsolver! = nothing`: (deprecated) short form for specifying a closed form subsolver.
$(_kwargs(:retraction_method))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_LM)"
LevenbergMarquardt(M::AbstractManifold, args...; kwargs...)
function LevenbergMarquardt(
        M::AbstractManifold, f, jacobian_f, num_components::Int = -1;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
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
                ArgumentError(
                    "For mutating evaluation num_components needs to be explicitly specified",
                ),
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
        M::AbstractManifold,
        vgf::VectorGradientFunction,
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        robustifier = IdentityRobustifier(),
        kwargs...,
    )
    # For a single vector gradient function, we always treat robustification componentwise
    nlso = NonlinearLeastSquaresObjective(vgf, ComponentwiseRobustifierFunction(robustifier))
    return LevenbergMarquardt(M, nlso, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt(
        M::AbstractManifold,
        vgf::Vector{<:VectorGradientFunction},
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        robustifier::Vector{<:AbstractRobustifierFunction} = [IdentityRobustifier() for _ in 1:length(vgf)],
        kwargs...,
    )
    nlso = NonlinearLeastSquaresObjective(vgf, robustifier)
    return LevenbergMarquardt(M, nlso, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt(
        M::AbstractManifold, nlso::O, p; kwargs...
    ) where {O <: Union{NonlinearLeastSquaresObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(LevenbergMarquardt; kwargs...)
    q = copy(M, p)
    return LevenbergMarquardt!(M, nlso, q; kwargs...)
end
calls_with_kwargs(::typeof(LevenbergMarquardt)) = (LevenbergMarquardt!,)

function construct_lm_subobjective(use_fast_coordinate_subobjective::Bool, nlso, damping_term_min, ε, α_mode, residuals, jacobian_f)
    if use_fast_coordinate_subobjective
        return NormalEquationsObjective(
            LevenbergMarquardtLinearSurrogateCoordinatesObjective(
                nlso; penalty = damping_term_min, ε = ε, mode = α_mode,
                residuals = residuals,
                jacobian_cache = jacobian_f,
            ),
        )
    else
        return NormalEquationsObjective(
            LevenbergMarquardtLinearSurrogateObjective(
                nlso; penalty = damping_term_min, ε = ε, mode = α_mode,
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
        function_type::AbstractVectorialType = FunctionVectorialType(),
        kwargs...,
    )
    if num_components == -1
        if evaluation === AllocatingEvaluation()
            num_components = length(f(M, p))
        else
            throw(
                ArgumentError(
                    "For mutating evaluation num_components needs to be explicitly specified",
                ),
            )
        end
    end
    nlso = NonlinearLeastSquaresObjective(
        f,
        jacobian_f,
        num_components;
        evaluation = evaluation,
        jacobian_type = jacobian_type,
        function_type = function_type,
    )
    return LevenbergMarquardt!(M, nlso, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt!(
        M::AbstractManifold, vgf::VectorGradientFunction, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        robustifier = IdentityRobustifier(),
        kwargs...,
    )
    nlso = NonlinearLeastSquaresObjective(vgf, robustifier)
    return LevenbergMarquardt!(M, nlso, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt!(
        M::AbstractManifold,
        vgf::Vector{<:VectorGradientFunction},
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        robustifier::Vector{<:AbstractRobustifierFunction} = [IdentityRobustifier() for _ in 1:length(vgf)],
        kwargs...,
    )
    nlso = NonlinearLeastSquaresObjective(vgf, robustifier)
    return LevenbergMarquardt!(M, nlso, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt!(
        M::AbstractManifold, nlso::O, p;
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        stopping_criterion::StoppingCriterion = StopAfterIteration(500) | StopWhenGradientNormLess(1.0e-12) | StopWhenStepsizeLess(1.0e-12),
        debug = [DebugWarnIfCostIncreases()],
        β::Real = 5.0,
        damping_reduction_threshold::Real = Inf,
        β_reduction::Real = 0.5,
        η::Real = 0.2,
        damping_term_min::Real = 0.1,
        X = zero_vector(M, p),
        initial_residual_values = zeros(number_eltype(p), residuals_count(get_objective(nlso))),
        initial_jacobian_f = fill(nothing, length(get_objective(nlso).objective)),
        (linear_subsolver!) = nothing,
        #TODO better names for the next 2?
        ε::Real = 1.0e-6,
        α_mode::Symbol = :Default,
        minimum_acceptable_model_improvement::Real = eps(number_eltype(p)),
        sub_evaluation::AbstractEvaluationType = InplaceEvaluation(),
        use_fast_coordinate_system::Bool = false,
        sub_objective = construct_lm_subobjective(use_fast_coordinate_system, nlso, damping_term_min, ε, α_mode, initial_residual_values, initial_jacobian_f),
        sub_problem = DefaultManoptProblem(TangentSpace(M, p), sub_objective),
        sub_state = if isnothing(linear_subsolver!)
            ConjugateResidualState(TangentSpace(M, p), sub_objective)
        else
            CoordinatesNormalSystemState(M, p; linsolve = linear_subsolver!, evaluation = sub_evaluation)
        end,
        kwargs..., #collect rest
    ) where {O <: Union{NonlinearLeastSquaresObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(LevenbergMarquardt!; kwargs...)
    dnlso = decorate_objective!(M, nlso; kwargs...)
    nlsp = DefaultManoptProblem(M, dnlso)
    sub_state_ = maybe_wrap_evaluation_type(sub_state)
    lms = LevenbergMarquardtState(
        M, initial_residual_values;
        p = p,
        # TODO Rename to have either only math symbols or only speaking names but not both
        β = β,
        damping_reduction_threshold = damping_reduction_threshold,
        β_reduction = β_reduction,
        η = η,
        damping_term_min,
        stopping_criterion = stopping_criterion,
        retraction_method = retraction_method,
        sub_problem = sub_problem,
        sub_state = sub_state_,
        minimum_acceptable_model_improvement = minimum_acceptable_model_improvement,
        initial_jacobian_f = initial_jacobian_f,
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
        dmp::DefaultManoptProblem{mT, <:NonlinearLeastSquaresObjective}, lms::LevenbergMarquardtState,
    ) where {mT <: AbstractManifold}
    M = get_manifold(dmp)
    nlso = get_objective(dmp)
    get_residuals!(M, lms.residual_values, nlso, lms.p)
    for (o, jb) in zip(nlso.objective, lms.jacobian_f)
        if !isnothing(jb)
            get_jacobian!(M, jb, o, lms.p)
        end
    end
    get_gradient!(M, lms.X, nlso, lms.p; value_cache = lms.residual_values, jacobian_cache = lms.jacobian_f)
    return lms
end

function step_solver!(
        dmp::DefaultManoptProblem{mT, <:NonlinearLeastSquaresObjective},
        lms::LevenbergMarquardtState,
        ::Integer,
    ) where {mT <: AbstractManifold}
    # Update damping term in the surrogate
    # should this be with (currently) or without robustifier?
    M = get_manifold(dmp)
    nlso = get_objective(dmp)
    FpSq = get_cost(M, nlso, lms.p)
    set_parameter!(lms.sub_problem, Val(:Objective), Val(:Penalty), lms.damping_term * FpSq)
    # update base point of the tangent space the subproblem works on
    set_parameter!(lms.sub_problem, Val(:Manifold), Val(:Basepoint), lms.p)
    # Subsolver result
    solve_LM_subproblem!(M, lms.direction, lms.p, lms.sub_problem, lms.sub_state)
    #solve!(lms.sub_problem, lms.sub_state)
    #lms.direction .= -get_solver_result(lms.sub_problem, lms.sub_state)
    if norm(M, lms.p, lms.direction) > max_stepsize(M, lms.p)
        # Vector too long; we can reject the step without evaluating the objective
        lms.damping_term *= lms.β
        return lms
    end
    model_improvement = (get_cost(lms.sub_problem, ZeroTangentVector()) - get_cost(lms.sub_problem, lms.direction)) / 2
    if model_improvement < lms.minimum_acceptable_model_improvement
        # Model improvement insufficient, reject step and increase damping term
        lms.damping_term *= lms.β
        return lms
    end
    # New iterate candidate - maybe store in state?

    q = retract(M, lms.p, lms.direction, lms.retraction_method)

    # Evaluate improvement of actual cost divided by predicted cost improvement
    cost_improvement = get_cost(M, nlso, lms.p) - get_cost(M, nlso, q)
    ρ = cost_improvement / model_improvement
    # Update damping term and iterate
    # TODO Abstract this to a generic update for η?
    if ρ >= lms.damping_reduction_threshold
        lms.damping_term *= lms.β_reduction
        lms.damping_term = max(lms.damping_term, lms.damping_term_min)
    end
    if ρ >= lms.η # enough improvement: accept, decrease damping term
        copyto!(M, lms.p, q)
        get_residuals!(M, lms.residual_values, nlso, lms.p)
        for (o, jb) in zip(nlso.objective, lms.jacobian_f)
            if !isnothing(jb)
                get_jacobian!(M, jb, o, lms.p)
            end
        end
        get_gradient!(M, lms.X, nlso, lms.p; value_cache = lms.residual_values, jacobian_cache = lms.jacobian_f)
    else # not enough improvement: reject, increase damping term
        lms.damping_term *= lms.β
        lms.damping_term = min(lms.damping_term, lms.damping_term_max)
    end
    return lms
end

function solve_LM_subproblem!(
        M::AbstractManifold, X, p, problem::P, state::S,
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
        dmp::DefaultManoptProblem{mT, <:NonlinearLeastSquaresObjective},
        lms::LevenbergMarquardtState,
        k,
    ) where {mT <: AbstractManifold}
    M = get_manifold(dmp)
    return norm(M, lms.p, lms.direction)
end
