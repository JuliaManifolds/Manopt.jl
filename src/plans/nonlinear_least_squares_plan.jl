
@doc raw"""
    NonlinearLeastSquaresObjective{T<:AbstractEvaluationType} <: AbstractManifoldObjective{T}

A type for nonlinear least squares problems.
`T` is a [`AbstractEvaluationType`](@ref) for the `F` and Jacobian functions.

Specify a nonlinear least squares problem

# Fields
* `f`        – a function ``f: \mathcal M → ℝ^d`` to minimize
* `jacobian!!`   – Jacobian of the function ``f``
* `jacobian_tangent_basis`     – the basis of tangent space used for computing the Jacobian.
* `num_components` – number of values returned by `f` (equal to `d`).

Depending on the [`AbstractEvaluationType`](@ref) `T` the function ``F`` has to be provided:

* as a functions `(M::AbstractManifold, p) -> v` that allocates memory for `v` itself for
  an [`AllocatingEvaluation`](@ref),
* as a function `(M::AbstractManifold, v, p) -> v` that works in place of `v` for a
  [`InplaceEvaluation`](@ref).

Also the Jacobian ``jacF!!`` is required:

* as a functions `(M::AbstractManifold, p; basis_domain::AbstractBasis) -> v` that allocates
  memory for `v` itself for an [`AllocatingEvaluation`](@ref),
* as a function `(M::AbstractManifold, v, p; basis_domain::AbstractBasis) -> v` that works
  in place of `v` for an [`InplaceEvaluation`](@ref).

# Constructors

    NonlinearLeastSquaresProblem(M, F, jacF, num_components; evaluation=AllocatingEvaluation(), jacobian_tangent_basis=DefaultOrthonormalBasis())

# See also

[`LevenbergMarquardt`](@ref), [`LevenbergMarquardtState`](@ref)
"""
struct NonlinearLeastSquaresObjective{E<:AbstractEvaluationType,TC,TJ,TB<:AbstractBasis} <:
       AbstractManifoldGradientObjective{E,TC,TJ}
    f::TC
    jacobian!!::TJ
    jacobian_tangent_basis::TB
    num_components::Int
end
function NonlinearLeastSquaresObjective(
    f::TF,
    jacobian_f::TJ,
    num_components::Int;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    jacB=nothing,
    jacobian_tangent_basis::TB=isnothing(jacB) ? DefaultOrthonormalBasis() : jacB,
) where {TF,TJ,TB<:AbstractBasis}
    !isnothing(jacB) &&
        (@warn "The keyword `jacB` is deprecated, use `jacobian_tangent_basis` instead.")
    return NonlinearLeastSquaresObjective{typeof(evaluation),TF,TJ,TB}(
        f, jacobian_f, jacobian_tangent_basis, num_components
    )
end

function get_cost(
    M::AbstractManifold, nlso::NonlinearLeastSquaresObjective{AllocatingEvaluation}, p
)
    return 1//2 * norm(nlso.f(M, p))^2
end
function get_cost(
    M::AbstractManifold, nlso::NonlinearLeastSquaresObjective{InplaceEvaluation}, p
)
    residual_values = zeros(nlso.num_components)
    nlso.f(M, residual_values, p)
    return 1//2 * norm(residual_values)^2
end

function get_gradient(
    M::AbstractManifold, nlso::NonlinearLeastSquaresObjective{AllocatingEvaluation}, p
)
    basis_x = _maybe_get_basis(M, p, nlso.jacobian_tangent_basis)
    Jval = nlso.jacobian!!(M, p; basis_domain=basis_x)
    residual_values = nlso.f(M, p)
    return get_vector(M, p, transpose(Jval) * residual_values, basis_x)
end
function get_gradient(
    M::AbstractManifold, nlso::NonlinearLeastSquaresObjective{InplaceEvaluation}, p
)
    basis_x = _maybe_get_basis(M, p, nlso.jacobian_tangent_basis)
    Jval = zeros(nlso.num_components, manifold_dimension(M))
    nlso.jacobian!!(M, Jval, p; basis_domain=basis_x)
    residual_values = zeros(nlso.num_components)
    nlso.f(M, residual_values, p)
    return get_vector(M, p, transpose(Jval) * residual_values, basis_x)
end

function get_gradient!(
    M::AbstractManifold, X, nlso::NonlinearLeastSquaresObjective{AllocatingEvaluation}, p
)
    basis_x = _maybe_get_basis(M, p, nlso.jacobian_tangent_basis)
    Jval = nlso.jacobian!!(M, p; basis_domain=basis_x)
    residual_values = nlso.f(M, p)
    return get_vector!(M, X, p, transpose(Jval) * residual_values, basis_x)
end

function get_gradient!(
    M::AbstractManifold, X, nlso::NonlinearLeastSquaresObjective{InplaceEvaluation}, p
)
    basis_p = _maybe_get_basis(M, p, nlso.jacobian_tangent_basis)
    Jval = zeros(nlso.num_components, manifold_dimension(M))
    nlso.jacobian!!(M, Jval, p; basis_domain=basis_p)
    residual_values = zeros(nlso.num_components)
    nlso.f(M, residual_values, p)
    get_vector!(M, X, p, transpose(Jval) * residual_values, basis_p)
    return X
end

@doc raw"""
    LevenbergMarquardtState{P,T} <: AbstractGradientSolverState

Describes a Gradient based descent algorithm, with

# Fields

A default value is given in brackets if a parameter can be left out in initialization.

* `x` – a point (of type `P`) on a manifold as starting point
* `stop` – (`StopAfterIteration(200) | StopWhenGradientNormLess(1e-12) | StopWhenStepsizeLess(1e-12)`)
  a [`StoppingCriterion`](@ref)
* `retraction_method` – (`default_retraction_method(M, typeof(p))`) the retraction to use, defaults to
  the default set for your manifold.
* `residual_values` – value of ``F`` calculated in the solver setup or the previous iteration
* `residual_values_temp` – value of ``F`` for the current proposal point
* `jacF` – the current Jacobian of ``F``
* `gradient` – the current gradient of ``F``
* `step_vector` – the tangent vector at `x` that is used to move to the next point
* `last_stepsize` – length of `step_vector`
* `η` – Scaling factor for the sufficient cost decrease threshold required to accept new proposal points. Allowed range: `0 < η < 1`.
* `damping_term` – current value of the damping term
* `damping_term_min` – initial (and also minimal) value of the damping term
* `β` – parameter by which the damping term is multiplied when the current new point is rejected
* `expect_zero_residual` – (`false`) if true, the algorithm expects that the value of
  residual (objective) at mimimum is equal to 0.

# Constructor

    LevenbergMarquardtState(M, initialX, initial_residual_values, initial_jacF; initial_vector), kwargs...)

Generate Levenberg-Marquardt options.

# See also

[`gradient_descent`](@ref), [`LevenbergMarquardt`](@ref)
"""
mutable struct LevenbergMarquardtState{
    P,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    Tresidual_values,
    TJac,
    TGrad,
    Tparams<:Real,
} <: AbstractGradientSolverState
    p::P
    stop::TStop
    retraction_method::TRTM
    residual_values::Tresidual_values
    candidate_residual_values::Tresidual_values
    jacF::TJac
    X::TGrad
    step_vector::TGrad
    last_stepsize::Tparams
    η::Tparams
    damping_term::Tparams
    damping_term_min::Tparams
    β::Tparams
    expect_zero_residual::Bool
    function LevenbergMarquardtState(
        M::AbstractManifold,
        p::P,
        initial_residual_values::Tresidual_values,
        initial_jacF::TJac,
        initial_gradient::TGrad=zero_vector(M, p);
        stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                              StopWhenGradientNormLess(1e-12) |
                                              StopWhenStepsizeLess(1e-12),
        retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
        η::Real=0.2,
        damping_term_min::Real=0.1,
        β::Real=5.0,
        expect_zero_residual::Bool=false,
    ) where {P,Tresidual_values,TJac,TGrad}
        if η <= 0 || η >= 1
            throw(ArgumentError("Value of η must be strictly between 0 and 1, received $η"))
        end
        if damping_term_min <= 0
            throw(
                ArgumentError(
                    "Value of damping_term_min must be strictly above 0, received $damping_term_min",
                ),
            )
        end
        if β <= 1
            throw(ArgumentError("Value of β must be strictly above 1, received $β"))
        end
        Tparams = promote_type(typeof(η), typeof(damping_term_min), typeof(β))
        return new{
            P,
            typeof(stopping_criterion),
            typeof(retraction_method),
            Tresidual_values,
            TJac,
            TGrad,
            Tparams,
        }(
            p,
            stopping_criterion,
            retraction_method,
            initial_residual_values,
            copy(initial_residual_values),
            initial_jacF,
            initial_gradient,
            allocate(M, initial_gradient),
            zero(Tparams),
            η,
            damping_term_min,
            damping_term_min,
            β,
            expect_zero_residual,
        )
    end
end

function show(io::IO, lms::LevenbergMarquardtState)
    i = get_count(lms, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(lms.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Levenberg Marquardt Algorithm
    $Iter
    ## Parameters
    * β: $(lms.β)
    * damping term_ $(lms.damping_term) (min: $(lms.damping_term_min))
    * η: $(lms.η)
    * expect zero residual: $(lms.expect_zero_residual)
    * retraction method: $(lms.retraction_method)

    ## Stopping Criterion
    $(status_summary(lms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
