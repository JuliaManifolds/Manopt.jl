_doc_LM_formula = raw"""
```math
\operatorname*{arg\,min}_{p ∈ \mathcal M} \frac{1}{2} \lVert f(p) \rVert^2,
```
"""
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

The second signature performs the optimization in-place of `p`.

# Input

$(_var(:Argument, :M; type=true))
* `f`: a cost function ``f: $(_math(:M))→ℝ^d``.
  The cost function can be provided in two different ways
    * as a single function returning a vector ``f(p)) ∈ ℝ^d``
    * as a vector of functions, where each single function returns a scalar ``f_i(p) ∈ ℝ``
  The type is determined by the `function_type=` keyword argument.
* `jacobian_f`:   the Jacobian of ``f``.
  The Jacobian can be provided in three different ways
  * as a single function returning a vector of gradient vectors ``$(_tex(:grad)) f_i(p)``
  * as a vector of functions, where each single function returns a gradient vector ``$(_tex(:grad)) f_i(p)``
  * as a single function returning a (coefficient) matrix with respect to an [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`)
    of the trangent space at `p`.
  The type is determined by the `jacobian_type=` keyword argument.
$(_var(:Argument, :p))
* `num_components`: length of the vector returned by the cost function (`d`).
  By default its value is -1 which means that it is determined automatically by
  calling `f` one additional time. This is only possible when `evaluation` is [`AllocatingEvaluation`](@ref),
  for mutating evaluation this value must be explicitly specified.

You can also provide the cost and its Jacobian already as a[`VectorGradientFunction`](@ref) `vgf`,
Alternatively, passing a [`NonlinearLeastSquaresObjective`](@ref) `nlso`,
then both the keyword `jacobian_tangent_basis` and the `smoothing` are ignored.

# Keyword arguments

$(_var(:Keyword, :evaluation))
* `η=0.2`:                   scaling factor for the sufficient cost decrease threshold required to accept new proposal points. Allowed range: `0 < η < 1`.
* `expect_zero_residual=false`: whether or not the algorithm might expect that the value of
  residual (objective) at minimum is equal to 0.
* `damping_term_min=0.1`:      initial (and also minimal) value of the damping term
* `β=5.0`:                     parameter by which the damping term is multiplied when the current new point is rejected
* `function_type=`[`FunctionVectorialType`](@ref)`: an [`AbstractVectorialType`](@ref) specifying the type of cost function provided.
* `initial_jacobian_f`:      the initial Jacobian of the cost function `f`.
  By default this is a matrix of size `num_components` times the manifold dimension of similar type as `p`.
* `initial_residual_values`: the initial residual vector of the cost function `f`.
  By default this is a vector of length `num_components` of similar type as `p`.
* `jacobian_type=`[`FunctionVectorialType`](@ref)`: an [`AbstractVectorialType`](@ref) specifying the type of Jacobian provided.
* `smoothing`: specify the function ``ρ`` as an [`ManifoldHessianObjective`](@ref) object.
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :stopping_criterion; default="[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-12)`"))

$(_note(:OtherKeywords))

$(_note(:OutputSection))
"""

@doc "$(_doc_LM)"
LevenbergMarquardt(M::AbstractManifold, args...; kwargs...)
function LevenbergMarquardt(
    M::AbstractManifold,
    f,
    jacobian_f,
    num_components::Int=-1;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs...,
)
    return LevenbergMarquardt(
        M, f, jacobian_f, rand(M), num_components; evaluation=evaluation, kwargs...
    )
end
function LevenbergMarquardt(
    M::AbstractManifold,
    f,
    jacobian_f,
    p,
    num_components::Int=-1;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    function_type::AbstractVectorialType=FunctionVectorialType(),
    jacobian_type::AbstractVectorialType=FunctionVectorialType(),
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
        evaluation=evaluation,
        function_type=function_type,
        jacobian_type=jacobian_type,
    )
    return LevenbergMarquardt(M, vgf, p; evaluation=evaluation, kwargs...)
end
function LevenbergMarquardt(
    M::AbstractManifold,
    vgf::VectorGradientFunction,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    smoothing=:Idenity,
    kwargs...,
)
    _smoothing = smoothing_factory(smoothing)
    nlso = NonlinearLeastSquaresObjective(vgf, _smoothing; evaluation=evaluation)
    return LevenbergMarquardt(M, nlso, p; evaluation=evaluation, kwargs...)
end
function LevenbergMarquardt(
    M::AbstractManifold, nlso::O, p; kwargs...
) where {O<:Union{NonlinearLeastSquaresObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return LevenbergMarquardt!(M, nlso, q; kwargs...)
end

@doc "$(_doc_LM)"
LevenbergMarquardt!(M::AbstractManifold, args...; kwargs...)
function LevenbergMarquardt!(
    M::AbstractManifold,
    f,
    jacobian_f,
    p,
    num_components::Int=-1;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    jacobian_tangent_basis::AbstractBasis=DefaultOrthonormalBasis(),
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
        evaluation=evaluation,
        jacobian_tangent_basis=jacobian_tangent_basis,
    )
    return LevenbergMarquardt!(M, nlso, p; evaluation=evaluation, kwargs...)
end
function LevenbergMarquardt!(
    M::AbstractManifold,
    nlso::O,
    p;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(1e-12) |
                                          StopWhenStepsizeLess(1e-12),
    debug=[DebugWarnIfCostIncreases()],
    expect_zero_residual::Bool=false,
    β::Real=5.0,
    η::Real=0.2,
    damping_term_min::Real=0.1,
    initial_residual_values=similar(p, get_objective(nlso).num_components),
    initial_jacobian_f=similar(
        p, get_objective(nlso).num_components, manifold_dimension(M)
    ),
    kwargs..., #collect rest
) where {O<:Union{NonlinearLeastSquaresObjective,AbstractDecoratedManifoldObjective}}
    i_nlso = get_objective(nlso) # un-decorate for safety
    dnlso = decorate_objective!(M, nlso; kwargs...)
    nlsp = DefaultManoptProblem(M, dnlso)
    lms = LevenbergMarquardtState(
        M,
        initial_residual_values,
        initial_jacobian_f;
        p=p,
        β,
        η,
        damping_term_min,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        expect_zero_residual=expect_zero_residual,
    )
    dlms = decorate_state!(lms; debug=debug, kwargs...)
    solve!(nlsp, dlms)
    return get_solver_return(get_objective(nlsp), dlms)
end
#
# Solver functions
#
function initialize_solver!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    lms::LevenbergMarquardtState,
) where {mT<:AbstractManifold}
    M = get_manifold(dmp)
    lms.residual_values = get_objective(dmp).f(M, lms.p)
    lms.X = get_gradient(dmp, lms.p)
    return lms
end
function initialize_solver!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{InplaceEvaluation}},
    lms::LevenbergMarquardtState,
) where {mT<:AbstractManifold}
    M = get_manifold(dmp)
    get_objective(dmp).f(M, lms.residual_values, lms.p)
    # TODO: Replace with a call to the Jacobian
    get_gradient_from_Jacobian!(M, lms.X, get_objective(dmp), lms.p, lms.jacF)
    return lms
end

function _maybe_get_basis(M::AbstractManifold, p, B::AbstractBasis)
    if requires_caching(B)
        return get_basis(M, p, B)
    else
        return B
    end
end

# TODO: Adapt to vectorial function call instead of .f - maybe even skip or rename?
# TODO: It will just ne “get_value” of the vgf - adapted by smoothing?.
function get_residuals!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{AllocatingEvaluation}},
    residuals,
    p,
) where {mT}
    return copyto!(residuals, get_objective(dmp).f(get_manifold(dmp), p))
end
function get_residuals!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective{InplaceEvaluation}},
    residuals,
    p,
) where {mT}
    return get_objective(dmp).f(get_manifold(dmp), residuals, p)
end

function step_solver!(
    dmp::DefaultManoptProblem{mT,<:NonlinearLeastSquaresObjective},
    lms::LevenbergMarquardtState,
    ::Integer,
) where {mT<:AbstractManifold}
    # `o.residual_values` is either initialized by `initialize_solver!` or taken from the previous iteration
    M = get_manifold(dmp)
    nlso = get_objective(dmp)
    # TODO: Replace with obtaining a basis from the vectorial function
    basis_ox = _maybe_get_basis(M, lms.p, nlso.jacobian_tangent_basis)
    # a new Jacobian is only  needed if the last step was successful
    if lms.last_step_successful
        get_jacobian!(dmp, lms.jacF, lms.p)
    end
    λk = lms.damping_term * norm(lms.residual_values)^2

    JJ = transpose(lms.jacF) * lms.jacF + λk * I
    # `cholesky` is technically not necessary but it's the fastest method to solve the
    # problem because JJ is symmetric positive definite
    grad_f_c = transpose(lms.jacF) * lms.residual_values
    sk = cholesky(JJ) \ -grad_f_c
    get_vector!(M, lms.X, lms.p, grad_f_c, basis_ox)

    get_vector!(M, lms.step_vector, lms.p, sk, basis_ox)
    lms.last_stepsize = norm(M, lms.p, lms.step_vector)
    temp_x = retract(M, lms.p, lms.step_vector, lms.retraction_method)

    normFk2 = norm(lms.residual_values)^2
    get_residuals!(dmp, lms.candidate_residual_values, temp_x)

    ρk =
        (normFk2 - norm(lms.candidate_residual_values)^2) / (
            -2 * inner(M, lms.p, lms.X, lms.step_vector) - norm(lms.jacF * sk)^2 -
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
