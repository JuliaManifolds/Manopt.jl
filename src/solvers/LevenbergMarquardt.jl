_doc_LM = """
    LevenbergMarquardt(M, f, jacobian_f, p, num_components=-1; kwargs...)
    LevenbergMarquardt(M, vgf, p; kwargs...)
    LevenbergMarquardt(M, nlso, p; kwargs...)
    LevenbergMarquardt!(M, f, jacobian_f, p, num_components=-1; kwargs...)
    LevenbergMarquardt!(M, vgf, p, num_components=-1; kwargs...)
    LevenbergMarquardt!(M, nlso, p, num_components=-1; kwargs...)

compute the the Riemannian Levenberg-Marquardt algorithm [Peeters:1993, AdachiOkunoTakeda:2022](@cite)
to solve

$(_problem(:NonLinearLeastSquares))

The second block of signatures perform the optimization in-place of `p`.

# Input

$(_var(:Argument, :M; type = true))
* `f`: a cost function ``f: $(_math(:M))→ℝ^m``.
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
$(_var(:Argument, :p))
* `num_components`: length ``m`` of the vector returned by the cost function.
  By default its value is -1 which means that it is determined automatically by
  calling `f` one additional time. This is only possible when `evaluation` is [`AllocatingEvaluation`](@ref),
  for mutating evaluation this value must be explicitly specified.

You can also provide the cost and its Jacobian already as a[`VectorGradientFunction`](@ref) `vgf`,
Alternatively, passing a [`NonlinearLeastSquaresObjective`](@ref) `nlso`.

# Keyword arguments

$(_var(:Keyword, :evaluation))
* `η=0.2`:                   scaling factor for the sufficient cost decrease threshold required to accept new proposal points. Allowed range: `0 < η < 1`.
* `expect_zero_residual=false`: whether or not the algorithm might expect that the value of
  residual (objective) at minimum is equal to 0.
* `damping_term_min=0.1`:      initial (and also minimal) value of the damping term
* `β=5.0`:                     parameter by which the damping term is multiplied when the current new point is rejected
* `function_type=`[`FunctionVectorialType`](@ref): an [`AbstractVectorialType`](@ref) specifying the type of cost function provided.
* `initial_jacobian_f`:      the initial Jacobian of the cost function `f`.
  By default this is a matrix of size `num_components` times the manifold dimension of similar type as `p`.
* `initial_residual_values`: the initial residual vector of the cost function `f`.
  By default this is a vector of length `num_components` of similar type as `p`.
* `jacobian_type=`[`FunctionVectorialType`](@ref): an [`AbstractVectorialType`](@ref) specifying the type of Jacobian provided.
* `linear_subsolver!`:    a function with three arguments `sk, JJ, grad_f_c` that solves the
  linear subproblem `sk .= JJ \\ grad_f_c`, where `JJ` is (up to numerical issues) a
  symmetric positive definite matrix. Default value is [`default_lm_lin_solve!`](@ref).
$(_var(:Keyword, :retraction_method))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_LM)"
LevenbergMarquardt(M::AbstractManifold, args...; kwargs...)
function LevenbergMarquardt(
        M::AbstractManifold,
        f,
        jacobian_f,
        num_components::Int = -1;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    return LevenbergMarquardt(
        M, f, jacobian_f, rand(M), num_components; evaluation = evaluation, kwargs...
    )
end
function LevenbergMarquardt(
        M::AbstractManifold,
        f,
        jacobian_f,
        p,
        num_components::Int = -1;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        function_type::AbstractVectorialType = FunctionVectorialType(),
        jacobian_type::AbstractVectorialType = CoordinateVectorialType(DefaultOrthonormalBasis()),
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
        f,
        jacobian_f,
        num_components;
        evaluation = evaluation,
        function_type = function_type,
        jacobian_type = jacobian_type,
    )
    return LevenbergMarquardt(M, vgf, p; evaluation = evaluation, kwargs...)
end
function LevenbergMarquardt(
        M::AbstractManifold,
        vgf::VectorGradientFunction,
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    nlso = NonlinearLeastSquaresObjective(vgf)
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

@doc "$(_doc_LM)"
LevenbergMarquardt!(M::AbstractManifold, args...; kwargs...)
function LevenbergMarquardt!(
        M::AbstractManifold,
        f,
        jacobian_f,
        p,
        num_components::Int = -1;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        jacobian_tangent_basis::AbstractBasis = default_basis(M, typeof(p)),
        jacobian_type = CoordinateVectorialType(jacobian_tangent_basis),
        function_type = FunctionVectorialType(),
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
        M::AbstractManifold,
        nlso::O,
        p;
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        stopping_criterion::StoppingCriterion = StopAfterIteration(200) |
            StopWhenGradientNormLess(1.0e-12) |
            StopWhenStepsizeLess(1.0e-12),
        debug = [DebugWarnIfCostIncreases()],
        expect_zero_residual::Bool = false,
        β::Real = 5.0,
        η::Real = 0.2,
        damping_term_min::Real = 0.1,
        initial_residual_values = similar(p, length(get_objective(nlso).objective)),
        initial_jacobian_f = similar(
            p, length(get_objective(nlso).objective), manifold_dimension(M)
        ),
        (linear_subsolver!) = (default_lm_lin_solve!),
        kwargs..., #collect rest
    ) where {O <: Union{NonlinearLeastSquaresObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(LevenbergMarquardt!; kwargs...)
    dnlso = decorate_objective!(M, nlso; kwargs...)
    nlsp = DefaultManoptProblem(M, dnlso)
    lms = LevenbergMarquardtState(
        M,
        initial_residual_values,
        initial_jacobian_f;
        p = p,
        β,
        η,
        damping_term_min,
        stopping_criterion = stopping_criterion,
        retraction_method = retraction_method,
        expect_zero_residual = expect_zero_residual,
        (linear_subsolver!) = (linear_subsolver!),
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
        dmp::DefaultManoptProblem{mT, <:NonlinearLeastSquaresObjective},
        lms::LevenbergMarquardtState,
    ) where {mT <: AbstractManifold}
    M = get_manifold(dmp)
    nlso = get_objective(dmp)
    get_residuals!(M, lms.residual_values, nlso, lms.p)
    get_jacobian!(M, lms.jacobian, nlso, lms.p)
    get_gradient!(M, lms.X, nlso, lms.p; jacobian_cache = lms.jacobian)
    return lms
end

"""
    default_lm_lin_solve!(sk, JJ, grad_f_c)

Solve the system `JJ \\ grad_f_c` where JJ is (mathematically) a symmetric positive
definite matrix and save the result to `sk`. In case of numerical errors the
`PosDefException` is caught and the default symmetric solver `(Symmetric(JJ) \\ grad_f_c)`
is used.

The function is intended to be used with [`LevenbergMarquardt`](@ref).
"""
function default_lm_lin_solve!(sk, JJ, grad_f_c)
    try
        ldiv!(sk, cholesky(JJ), grad_f_c)
    catch e
        if e isa PosDefException
            sk .= Symmetric(JJ) \ grad_f_c
        else
            rethrow()
        end
    end
    return sk
end

function step_solver!(
        dmp::DefaultManoptProblem{mT, <:NonlinearLeastSquaresObjective},
        lms::LevenbergMarquardtState,
        ::Integer,
    ) where {mT <: AbstractManifold}
    # `o.residual_values` is either initialized by `initialize_solver!` or taken from the previous iteration
    M = get_manifold(dmp)
    nlso = get_objective(dmp)
    # a new Jacobian is only  needed if the last step was successful
    if lms.last_step_successful
        get_jacobian!(M, lms.jacobian, nlso, lms.p)
    end
    λk = lms.damping_term * norm(lms.residual_values)^2
    basis_ox = get_basis(nlso.objective.jacobian_type)
    JJ = transpose(lms.jacobian) * lms.jacobian + λk * I
    # `cholesky` is technically not necessary but it's the fastest method to solve the
    # problem because JJ is symmetric positive definite
    grad_f_c = transpose(lms.jacobian) * lms.residual_values
    sk = similar(grad_f_c)
    lms.linear_subsolver!(sk, JJ, grad_f_c)
    sk .*= -1
    get_vector!(M, lms.X, lms.p, grad_f_c, basis_ox)

    get_vector!(M, lms.step_vector, lms.p, sk, basis_ox)
    lms.last_stepsize = norm(M, lms.p, lms.step_vector)
    temp_x = retract(M, lms.p, lms.step_vector, lms.retraction_method)

    normFk2 = norm(lms.residual_values)^2
    get_value!(M, lms.candidate_residual_values, nlso.objective, temp_x)

    ρk =
        (normFk2 - norm(lms.candidate_residual_values)^2) / (
        -2 * inner(M, lms.p, lms.X, lms.step_vector) - norm(lms.jacobian * sk)^2 -
            λk * norm(sk)^2
    )
    if ρk >= lms.η
        copyto!(M, lms.p, temp_x)
        copyto!(lms.residual_values, lms.candidate_residual_values)
        if lms.expect_zero_residual
            lms.damping_term = max(lms.damping_term_min, lms.damping_term / lms.β)
        end
        lms.last_step_successful = true
    else
        lms.damping_term *= lms.β
        lms.last_step_successful = false
    end
    return lms
end

function get_last_stepsize(::AbstractManoptProblem, lms::LevenbergMarquardtState, ::Any...)
    return lms.last_stepsize
end
