## The name is not optimal but it is merely something internal with a small safeguard
"""
    default_lm_lin_solve!(sk, JJ::AbstractMatrix, grad_f_c)

Solve the linear system of equations of the normal equations `JJ \\ grad_f_c` where JJ is a symmetric positive
definite matrix and save the result to `sk`. In case of numerical errors the
`PosDefException` is caught and the default symmetric solver `(Symmetric(JJ) \\ grad_f_c)`
is used.

The function is intended to be used with [`LevenbergMarquardt`](@ref).
"""
function default_lm_lin_solve!(sk, JJ::AbstractMatrix, grad_f_c)
    try
        ldiv!(sk, cholesky(Symmetric(JJ)), grad_f_c)
    catch e
        e isa PosDefException ? (sk .= Symmetric(JJ) \ grad_f_c) : rethrow()
    end
    return sk
end

@doc """
    LevenbergMarquardtState{P,T} <: AbstractGradientSolverState

Describes a Gradient based descent algorithm, with

# Fields

* `damping_term`:                         current value of the damping term
* `damping_term_min`:                     lower bound for the damping term
* `damping_term_max`:                     upper bound for the damping term
* `damping_increase_factor`:              factor the damping term is multiplied with when the
  improvement quotient falls below `damping_increase_threshold`.
* `damping_reduction_factor`:             factor the damping term is multiplied with when the
  improvement quotient exceeds `damping_reduction_threshold`.
* `damping_reduction_threshold`:          threshold for the improvement quotient above which
  the damping term is reduced by multiplying it with `damping_reduction_factor`.
* `damping_increase_threshold`:           threshold for the improvement quotient below which
  the damping term is increased by multiplying it with `damping_increase_factor`.
* `direction`:                            the current search direction, which is the solution of
  the linearized subproblem in each iteration.
* `candidate_acceptance_threshold`:       Scaling factor for the sufficient cost decrease threshold required
  to accept new proposal points. Allowed range: `0 < η < 1`.
* `callbacks`:                            the callbacks dictionary
* `jacobian_matrices`:                           the current Jacobian of ``F`` in matrix form per block, hence a vector of matrices.
   This is (by default) set to `nothing` if another representation is used.
* `minimum_acceptable_model_improvement`: the minimum improvement in the model function that
  is required to accept a new point; if this is not met, the new point is rejected and
  the damping term is increased.
$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:retraction_method))
* `residual_values`:                       values of the residuals calculated in the solver setup or the previous iteration
$(_fields(:stopping_criterion; name = "stop"))
$(_fields(:sub_problem))
$(_fields(:sub_state))
$(_fields(:X))

# Constructor

    LevenbergMarquardtState(M, sub_problem, sub_state, initial_residual_values, initial_jacobian_matrices; kwargs...)

Generate the Levenberg-Marquardt solver state.

# Keyword arguments

The following fields are keyword arguments

* `candidate_acceptance_threshold = 0.2`,
* `damping_increase_factor = 5.0`
* `damping_reduction_factor = 0.5`
* `damping_term_min = 0.1`
* `damping_term_max = Inf`
* `damping_term = damping_term_min`
* `damping_reduction_threshold = Inf`
* `damping_increase_threshold = candidate_acceptance_threshold`
* `direction = copy(M, p, X)`
* `p = `$(_link(:rand))
* `X = `$(_link(:zero_vector))
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-12)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1e-12)"))
$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
* `minimum_acceptable_model_improvement::Real = eps(number_eltype(p))`

# See also

[`gradient_descent`](@ref), [`LevenbergMarquardt`](@ref)
"""
mutable struct LevenbergMarquardtState{
        P, T, R <: Real, C <: AbstractDict{Symbol}, Pr, St, TStop <: StoppingCriterion, TRTM <: AbstractRetractionMethod, TRes, TJac,
    } <: AbstractGradientSolverState
    candidate_acceptance_threshold::R
    damping_increase_factor::R
    damping_increase_threshold::R
    damping_reduction_threshold::R
    damping_reduction_factor::R
    damping_term::R
    damping_term_min::R
    damping_term_max::R
    direction::T
    callbacks::C
    jacobian_matrices::TJac
    minimum_acceptable_model_improvement::R
    p::P
    q::P
    residual_values::TRes
    retraction_method::TRTM
    stop::TStop
    sub_problem::Pr
    sub_state::St
    X::T
    function LevenbergMarquardtState(
            sub_problem::Pr, sub_state::St;
            candidate_acceptance_threshold::R, damping_increase_factor::R, damping_increase_threshold::R,
            damping_reduction_threshold::R, damping_reduction_factor::R, damping_term::R,
            damping_term_min::R, damping_term_max::R,
            direction::T, callbacks::C, jacobian_matrices::TJac, minimum_acceptable_model_improvement::R, p::P, q::P,
            residual_values::TRes, retraction_method::TRTM, stopping_criterion::SC, X::T
        ) where {P, T, R <: Real, C <: AbstractDict{Symbol}, Pr, St <: AbstractManoptSolverState, SC <: StoppingCriterion, TRTM <: AbstractRetractionMethod, TRes, TJac}
        return new{P, T, R, C, Pr, St, SC, TRTM, TRes, TJac}(
            candidate_acceptance_threshold, damping_increase_factor, damping_increase_threshold,
            damping_reduction_threshold, damping_reduction_factor, damping_term, damping_term_min, damping_term_max,
            direction, callbacks, jacobian_matrices, minimum_acceptable_model_improvement, p, q, residual_values,
            retraction_method, stopping_criterion, sub_problem, sub_state, X
        )
    end
    function LevenbergMarquardtState(
            M::AbstractManifold, sub_problem, sub_state, initial_residual_values, initial_jacobian_matrices = nothing;
            p = rand(M), X = zero_vector(M, p), direction = copy(M, p, X),
            callbacks = Dict{Symbol, Function}(),
            stopping_criterion::StoppingCriterion = StopAfterIteration(200) | StopWhenGradientNormLess(1.0e-12) | StopWhenStepsizeLess(1.0e-12),
            retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
            candidate_acceptance_threshold::Real = 0.2,
            damping_increase_factor::Real = 5.0,
            damping_increase_threshold::Real = candidate_acceptance_threshold,
            damping_reduction_threshold::Real = Inf,
            damping_reduction_factor::Real = 0.5,
            damping_term_min::Real = 0.1,
            damping_term_max::Real = Inf,
            damping_term::Real = damping_term_min,
            minimum_acceptable_model_improvement::Real = eps(number_eltype(p)),
        )
        (candidate_acceptance_threshold <= 0 || candidate_acceptance_threshold >= 1) && throw(ArgumentError("The value of `candidate_acceptance_threshold` must be strictly between 0 and 1, received $(candidate_acceptance_threshold)"))
        (damping_term_min <= 0) && throw(ArgumentError("The value of damping_term_min must be strictly above 0, received $damping_term_min"))
        (damping_increase_factor <= 1) && throw(ArgumentError("The value of `damping_increase_factor` must be strictly above 1, received $damping_increase_factor"))
        (damping_reduction_factor >= 1) && throw(ArgumentError("The value of `damping_reduction_factor` must be strictly below 1, received $damping_reduction_factor"))
        R = promote_type(
            typeof(candidate_acceptance_threshold), typeof(damping_term_min), typeof(damping_increase_factor), typeof(damping_increase_threshold),
            typeof(damping_reduction_threshold), typeof(damping_reduction_factor), typeof(damping_term_min),
            typeof(damping_term_max), typeof(damping_term), typeof(minimum_acceptable_model_improvement)
        )
        return LevenbergMarquardtState(
            sub_problem, sub_state;
            candidate_acceptance_threshold = convert(R, candidate_acceptance_threshold),
            damping_increase_factor = convert(R, damping_increase_factor), damping_increase_threshold = convert(R, damping_increase_threshold),
            damping_reduction_threshold = convert(R, damping_reduction_threshold), damping_reduction_factor = convert(R, damping_reduction_factor),
            damping_term = convert(R, damping_term), damping_term_min = convert(R, damping_term_min), damping_term_max = convert(R, damping_term_max),
            direction = direction, callbacks = callbacks, jacobian_matrices = initial_jacobian_matrices isa AbstractMatrix ? [initial_jacobian_matrices] : initial_jacobian_matrices, minimum_acceptable_model_improvement = convert(R, minimum_acceptable_model_improvement),
            p = p, q = copy(M, p), residual_values = initial_residual_values, retraction_method = retraction_method, stopping_criterion = stopping_criterion, X = X,
        )
    end
end
additional_callbacks(::Type{<:LevenbergMarquardtState}) = [:Stepsize, :DampingIncreaseStepTooLong, :DampingIncreaseModelInadequate, :DampingDecreaseImprovementTooGood, :DampingIncreaseImprovementTooPoor, :CandidateAccept, :CandidateReject]
get_callbacks(lms::LevenbergMarquardtState) = lms.callbacks
#
function status_summary(lms::LevenbergMarquardtState; context::Symbol = :default)
    (context === :short) && return repr(lms)
    i = get_count(lms, :Iterations)
    (context === :inline) && return "A solver state for the Levenberg–Marquardt algorithm$(_iteration_suffix(lms))"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(lms.stop) ? "Yes" : "No"
    as = _callbacks_summary(lms)
    return """
    # Solver state for `Manopt.jl`s Levenberg Marquardt Algorithm
    $Iter
    ## Parameters$(as)
    * candidate acceptance threshold:$(_MANOPT_INDENT)$(lms.candidate_acceptance_threshold)
    * damping reduction threshold:   $(_MANOPT_INDENT)$(lms.damping_reduction_threshold)
    * damping reduction factor:      $(_MANOPT_INDENT)$(lms.damping_reduction_factor)
    * damping increase threshold:    $(_MANOPT_INDENT)$(lms.damping_increase_threshold)
    * damping increase factor:       $(_MANOPT_INDENT)$(lms.damping_increase_factor)
    * damping term:                  $(_MANOPT_INDENT)$(lms.damping_term) (min: $(lms.damping_term_min) | max: $(lms.damping_term_max))
    * retraction method:             $(_MANOPT_INDENT)$(lms.retraction_method)

    ## Stopping criterion

    $(status_summary(lms.stop; context = context))
    The algorithm converged: $Conv"""
end
function show(io::IO, lms::LevenbergMarquardtState)
    print(io, "LevenbergMarquardtState(", lms.sub_problem, ", ", lms.sub_state, "; ")
    print(io, "candidate_acceptance_threshold = ", lms.candidate_acceptance_threshold)
    print(io, ", damping_increase_factor = ", lms.damping_increase_factor, ", damping_increase_threshold = ", lms.damping_increase_threshold)
    print(io, ", damping_reduction_threshold = ", lms.damping_reduction_threshold, ", damping_reduction_factor = ", lms.damping_reduction_factor)
    print(io, ", damping_term = ", lms.damping_term, ", damping_term_min = ", lms.damping_term_min, ", damping_term_max = ", lms.damping_term_max)
    print(io, ", direction = ", lms.direction, ", callbacks = ", lms.callbacks, ", jacobian_matrices = ", lms.jacobian_matrices, ", minimum_acceptable_model_improvement = ", lms.minimum_acceptable_model_improvement)
    print(io, ", p = ", lms.p, ", q = ", lms.q, ", residual_values = ", lms.residual_values, ", retraction_method = ", lms.retraction_method, ", stopping_criterion = ", lms.stop, ", X = ", lms.X)
    return print(io, ")")
end

_doc_LM = """
    LevenbergMarquardt(M, f, jacobian_f, p, num_components=-1; kwargs...)
    LevenbergMarquardt(M, vgf, p; kwargs...)
    LevenbergMarquardt(M, nlso, p; kwargs...)
    LevenbergMarquardt!(M, f, jacobian_f, p, num_components=-1; kwargs...)
    LevenbergMarquardt!(M, vgf, p; kwargs...)
    LevenbergMarquardt!(M, nlso, p; kwargs...)

compute the Riemannian Levenberg-Marquardt algorithm [Peeters:1993, AdachiOkunoTakeda:2022, BaranBergmann:2026](@cite)
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
* `jacobian_tangent_basis=`[`default_basis`](@extref `ManifoldsBase.default_basis-Union{Tuple{T}, Tuple{AbstractManifold, Type{T}}} where T`)`(M, typeof(p))`: the basis the Jacobian coefficients refer to.
* `jacobian_type=`[`CoefficientVectorialType`](@ref)`(jacobian_tangent_basis)`: an [`AbstractVectorialType`](@ref) specifying the type of Jacobian provided.

as well as then these are already combined in a single [`VectorGradientFunction`](@ref) `vgf`

* `robustifier::`[`AbstractRobustifierFunction`](@ref)` = `[`IdentityRobustifier`](@ref)`()`:
  for the robust variant, specify how the robustification is meant to take place.
  - if you provide a single vectorial function and its Jacobian, a single robustifer is applied
    to every component function of this vectorial function (each component is a block in the sum)
  - if you provide a vector of [`VectorGradientFunction`](@ref)s, each needs a robustifier.

as well as in general using the model improvement parameter ``m_k`` in several places, cf [BaranBergmann:2026](@cite)

* `candidate_acceptance_threshold=0.2`: sufficient model improvement ``η ∈ (0,1)``, i.e. ``m_k > η`` to accept a candidate point
* `damping_increase_factor=5.0`:        factor ``β_{$(_tex(:text, "i"))}`` to increase damping, when the model is inaccurate
* `damping_increase_threshold=candidate_acceptance_threshold`: threshold ``η_{$(_tex(:text, "l"))}`` the value ``m_k`` has to be below to increase damping.
  The default yields, that we increase damping when we reject a candidate.
* `damping_reduction_factor= 1 / damping_increase_factor`: factor ``β_{$(_tex(:text, "d"))}`` to reduce damping, when the model is accurate
* `damping_reduction_threshold=Inf`:    threshold ``η_{$(_tex(:text, "u"))}`` the value ``m_k`` has to exceed to reduce damping
  The default means, that we never reduce damping.
* `damping_term_min = 0.1`:             lower bound ``μ_{$(_tex(:text, "l"))}`` for the damping ``μ_k`` throughout the iterations
* `damping_term_max = Inf`:             upper bound ``μ_{$(_tex(:text, "u"))}`` for the damping ``μ_k`` throughout the iterations
* `initial_damping_term=damping_term_min`: initial damping ``μ_0``
* `initial_residual_values = zeros(m)`: a cache for the vector of residuals, `m` is the total number of residuals, summed over all blocks
* `initial_jacobian_matrices`: a cache for the evaluated Jacobians (currently only used if `use_unified_basis = true`, then initialized to a vector of jacobian matrices, otherwise ignored)
$(_kwargs(:retraction_method))
* `scaling_threshold = 1.0e-6`:         a threshold `ε` to bound the scaling parameter `α` in the robust case away from `1`, see [`get_LevenbergMarquardt_scaling`](@ref)
* `scaling_mode = :Strict`:            specify the scaling stabilization mode, see [`get_LevenbergMarquardt_scaling`](@ref)
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(500)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1.0e-12)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1.0e-12)"))
* `sub_objective`                      : specify the objective for the surrogate sub problem to solver in every iteration.
  This is set depending on the `use_unified_basis`
  - if `true` to the [`LevenbergMarquardtLinearSurrogateCoordinatesObjective`](@ref) which always works in coordinates of one single basis per tangent space and allows to cache Jacobian evaluations
  - if `false` to the [`LevenbergMarquardtLinearSurrogateObjective`](@ref) that can work either with linear operators or in (even different) coordinates.

  This keyword is ignored if you set the `sub_problem` and/or `sub_state` keyword directly
* `sub_problem = `[`DefaultManoptProblem`](@ref)`(`$(_link(:TangentSpace))`(M, p), sub_objective)`: specify the sub problem to be solved. This should usually be phrased on the tangent space at the current iterate
* `sub_state`: the solver for the surrogate, by default a [`ConjugateResidualState`](@ref), see [`conjugate_residual`](@ref),
  or a [`CoordinatesNormalSystemState`](@ref) on a manifold with box constraints, where the sub state is also wrapped to handle the bounds.
* `use_unified_basis = false`:           specify to use a single basis for all Jacobian evaluations at a certain iterate, see `sub_objective`
  this requires that all Jacobians involved are of type [`CoefficientVectorialType`](@ref), since only then a jacobian can be represented as a matrix,
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
        jacobian_tangent_basis::AbstractBasis = default_basis(M, typeof(p)),
        jacobian_type::AbstractVectorialType = CoefficientVectorialType(jacobian_tangent_basis),
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
        evaluation = evaluation, function_type = function_type, jacobian_type = jacobian_type, p = p
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
        function_type::AbstractVectorialType = FunctionVectorialType(),
        robustifier::AbstractRobustifierFunction = IdentityRobustifier(), kwargs...,
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
        f, jacobian_f, num_components, robustifier;
        evaluation = evaluation, jacobian_type = jacobian_type, function_type = function_type, p = p
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
        callbacks = Dict{Symbol, Function}(),
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
        scaling_mode::Symbol = :Strict,
        minimum_acceptable_model_improvement::Real = eps(number_eltype(p)),
        sub_objective = construct_lm_subobjective(use_unified_basis, nlso, damping_term_min, scaling_threshold, scaling_mode, initial_residual_values, initial_jacobian_matrices),
        sub_problem = DefaultManoptProblem(TangentSpace(M, p), sub_objective),
        sub_state = has_anisotropic_max_stepsize(M) ?
            CoordinatesNormalSystemState(M, p) :
            ConjugateResidualState(TangentSpace(M, p), sub_objective; X = zero_vector(M, p)),
        kwargs..., #collect rest
    ) where {O <: Union{ManifoldNonlinearLeastSquaresObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(LevenbergMarquardt!; kwargs...)
    dnlso = decorate_objective!(M, nlso; kwargs...)
    nlsp = DefaultManoptProblem(M, dnlso)
    sub_state_ = sub_state
    if has_anisotropic_max_stepsize(M)
        # This is how to recognize the box constraints as for example Hyperrectangle does
        sub_state_ = LevenbergMarquardtBoxSubsolver(M, sub_state_, p)
    end
    lms = LevenbergMarquardtState(
        M, sub_problem, sub_state_, initial_residual_values, initial_jacobian_matrices;
        callbacks = process_callbacks_arg(callbacks, LevenbergMarquardtState),
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
function initialize_solver!(dmp::DefaultManoptProblem, lms::LevenbergMarquardtState)
    M = get_manifold(dmp)
    nlso = get_objective(dmp, true) # unwarp decorators
    get_residuals!(M, lms.residual_values, nlso, lms.p)
    jms = isnothing(lms.jacobian_matrices) ? fill(nothing, length(nlso.objective)) : lms.jacobian_matrices
    for (o, jb) in zip(nlso.objective, jms)
        !isnothing(jb) && get_jacobian!(M, jb, o, lms.p)
    end
    get_gradient!(M, lms.X, nlso, lms.p; value_cache = lms.residual_values, jacobian_cache = jms)
    return lms
end

function step_solver!(
        dmp::DefaultManoptProblem, lms::LevenbergMarquardtState, k::Integer,
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
    callback(:Stepsize, dmp, lms, k)
    #solve!(lms.sub_problem, lms.sub_state)
    #lms.direction .= -get_solver_result(lms.sub_problem, lms.sub_state)
    if norm(M, lms.p, lms.direction) > max_stepsize(M, lms.p)
        # Vector too long:
        # we reject the step without evaluating the objective
        # and increase damping
        lms.damping_term *= lms.damping_increase_factor
        lms.damping_term = min(lms.damping_term, lms.damping_term_max)
        callback(:DampingIncreaseStepTooLong, dmp, lms, k)
        return lms
    end
    model_improvement = (get_cost(lms.sub_problem, ZeroVector()) - get_cost(lms.sub_problem, lms.direction)) / 2
    if model_improvement < lms.minimum_acceptable_model_improvement
        # Model improvement insufficient, reject step and increase damping term
        lms.damping_term *= lms.damping_increase_factor
        lms.damping_term = min(lms.damping_term, lms.damping_term_max)
        callback(:DampingIncreaseModelInadequate, dmp, lms, k)
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
        callback(:DampingDecreaseImprovementTooGood, dmp, lms, k)
    elseif ρ < lms.damping_increase_threshold
        # poor match between model and actual cost: increase damping term
        lms.damping_term *= lms.damping_increase_factor
        lms.damping_term = min(lms.damping_term, lms.damping_term_max)
        callback(:DampingIncreaseImprovementTooPoor, dmp, lms, k)
    end
    if ρ >= lms.candidate_acceptance_threshold # enough improvement: accept candidate
        callback(:CandidateAccept, dmp, lms, k)
        copyto!(M, lms.p, lms.q)
        get_residuals!(M, lms.residual_values, nlso, lms.p)
        jms = isnothing(lms.jacobian_matrices) ? fill(nothing, length(nlso.objective)) : lms.jacobian_matrices
        for (o, jb) in zip(nlso.objective, jms)
            !isnothing(jb) && get_jacobian!(M, jb, o, lms.p)
        end
        get_gradient!(M, lms.X, nlso, lms.p; value_cache = lms.residual_values, jacobian_cache = jms)
    else
        callback(:CandidateReject, dmp, lms, k)
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
# We could add “fully” closed form solvers via dispatch here as well – for now this is not used yet

# Special Case: With box constraints
function solve_LM_subproblem!(
        M::AbstractManifold, X, p, problem::AbstractManoptProblem,
        state::LevenbergMarquardtBoxSubsolver, grad_Y,
    )
    solve!(problem, state.internal_state)
    copyto!(M, X, p, get_solver_result(problem, state.internal_state))
    X .*= -1
    # trim to box using GCD
    gcd = GeneralizedCauchyDirectionSubsolver(M, p, state)
    state.last_gcd_result, state.last_gcd_stepsize = find_generalized_cauchy_direction!(M, gcd, X, p, X, grad_Y)
    if state.last_gcd_result === :not_found
        # no feasible movement in this direction: return a zero step so that
        # `StopWhenStepsizeLess` can stop the solver instead of looping on NaNs
        zero_vector!(M, X, p)
        return X
    end
    # even if step size larger than 1 is possible, we shouldn't try to go further
    X .*= min(one(state.last_gcd_stepsize), state.last_gcd_stepsize)
    return X
end


function get_last_stepsize(dmp::DefaultManoptProblem, lms::LevenbergMarquardtState, k)
    M = get_manifold(dmp)
    return norm(M, lms.p, lms.direction)
end
