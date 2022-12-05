
@doc raw"""
    NonlinearLeastSquaresProblem{T<:AbstractEvaluationType} <: Problem{T}

A type for nonlinear least squares problems.
`T` is a [`AbstractEvaluationType`](@ref) for the `F` and Jacobian functions.

Specify a nonlinear least squares problem

# Fields
* `M`        – a manifold ``\mathcal M``
* `F`        – a function ``F: \mathcal M → ℝ^d`` to minimize
* `jacF!!`   – Jacobian of the function ``F``
* `jacB`     – the basis of tangent space used for computing the Jacobian.
* `num_components` – number of values returned by `F` (equal to `d`).

Depending on the [`AbstractEvaluationType`](@ref) `T` the function ``F`` has to be provided:

* as a functions `(M::AbstractManifold, p) -> v` that allocates memory for `v` itself for
  an [`AllocatingEvaluation`](@ref),
* as a function `(M::AbstractManifold, v, p) -> v` that works in place of `v` for a
  [`MutatingEvaluation`](@ref).

Also the Jacobian ``jacF!!`` is required:

* as a functions `(M::AbstractManifold, p; basis_domain::AbstractBasis) -> v` that allocates
  memory for `v` itself for an [`AllocatingEvaluation`](@ref),
* as a function `(M::AbstractManifold, v, p; basis_domain::AbstractBasis) -> v` that works
  in place of `v` for an [`MutatingEvaluation`](@ref).

# Constructors

    NonlinearLeastSquaresProblem(M, F, jacF, num_components; evaluation=AllocatingEvaluation(), jacB=DefaultOrthonormalBasis())

# See also

[`LevenbergMarquardt`](@ref), [`LevenbergMarquardtOptions`](@ref)
"""
struct NonlinearLeastSquaresProblem{
    T<:AbstractEvaluationType,mT<:AbstractManifold,TF,TJ,TB<:AbstractBasis
} <: Problem{T}
    M::mT
    F::TF
    jacobian!!::TJ
    jacB::TB
    num_components::Int
end
function NonlinearLeastSquaresProblem(
    M::mT,
    F::TF,
    jacF::TJ,
    num_components::Int;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    jacB::TB=DefaultOrthonormalBasis(),
) where {mT<:AbstractManifold,TF,TJ,TB<:AbstractBasis}
    return NonlinearLeastSquaresProblem{typeof(evaluation),mT,TF,TJ,TB}(
        M, F, jacF, jacB, num_components
    )
end

function (d::DebugGradient)(::NonlinearLeastSquaresProblem, o::Options, i::Int)
    (i < 1) && return nothing
    Printf.format(d.io, Printf.Format(d.format), get_gradient(o))
    return nothing
end

function get_cost(P::NonlinearLeastSquaresProblem{AllocatingEvaluation}, p)
    return 1//2 * norm(P.F(P.M, p))^2
end
function get_cost(P::NonlinearLeastSquaresProblem{MutatingEvaluation}, p)
    residual_values = zeros(P.num_components)
    P.F(P.M, residual_values, p)
    return 1//2 * norm(residual_values)^2
end

function get_gradient(p::NonlinearLeastSquaresProblem{AllocatingEvaluation}, x)
    basis_x = _maybe_get_basis(p.M, x, p.jacB)
    Jval = p.jacobian!!(p.M, x; basis_domain=basis_x)
    residual_values = p.F(p.M, x)
    return get_vector(p.M, x, transpose(Jval) * residual_values, basis_x)
end
function get_gradient(p::NonlinearLeastSquaresProblem{MutatingEvaluation}, x)
    basis_x = _maybe_get_basis(p.M, x, p.jacB)
    Jval = zeros(p.num_components, manifold_dimension(p.M))
    p.jacobian!!(p.M, Jval, x; basis_domain=basis_x)
    residual_values = zeros(p.num_components)
    p.F(p.M, residual_values, x)
    return get_vector(p.M, x, transpose(Jval) * residual_values, basis_x)
end

function get_gradient!(p::NonlinearLeastSquaresProblem{AllocatingEvaluation}, X, x)
    basis_x = _maybe_get_basis(p.M, x, p.jacB)
    Jval = p.jacobian!!(p.M, x; basis_domain=basis_x)
    residual_values = p.F(p.M, x)
    return get_vector!(p.M, X, x, transpose(Jval) * residual_values, basis_x)
end

function get_gradient!(p::NonlinearLeastSquaresProblem{MutatingEvaluation}, X, x)
    basis_x = _maybe_get_basis(p.M, x, p.jacB)
    Jval = zeros(p.num_components, manifold_dimension(p.M))
    p.jacobian!!(p.M, Jval, x; basis_domain=basis_x)
    residual_values = zeros(p.num_components)
    p.F(p.M, residual_values, x)
    return get_vector!(p.M, X, x, transpose(Jval) * residual_values, basis_x)
end

@doc raw"""
    LevenbergMarquardtOptions{P,T} <: AbstractGradientOptions

Describes a Gradient based descent algorithm, with

# Fields

A default value is given in brackets if a parameter can be left out in initialization.

* `x` – a point (of type `P`) on a manifold as starting point
* `stop` – (`StopAfterIteration(200) | StopWhenGradientNormLess(1e-12) | StopWhenStepsizeLess(1e-12)`)
  a [`StoppingCriterion`](@ref)
* `retraction_method` – (`default_retraction_method(M)`) the retraction to use, defaults to
  the default set for your manifold.
* `residual_values` – value of ``F`` calculated in the solver setup or the previous iteration
* `residual_values_temp` – value of ``F`` for the current proposal point
* `jacF` – the current Jacobian of ``F``
* `gradient` – the current gradient of ``F``
* `step_vector` – the tangent vector at `x` that is used to move to the next point
* `last_stepsize` – length of `step_vector`
* `η` – parameter of the algorithm, the higher it is the more likely the algorithm will be
  to reject new proposal points
* `damping_term` – current value of the damping term
* `damping_term_min` – initial (and also minimal) value of the damping term
* `β` – parameter by which the damping term is multiplied when the current new point is rejected
* `expect_zero_residual` – (`false`) if true, the algorithm expects that the value of
  residual (objective) at mimimum is equal to 0.

# Constructor

    LevenbergMarquardtOptions(M, initialX, initial_residual_values, initial_jacF; initial_vector), kwargs...)

Generate Levenberg-Marquardt options.

# See also
[`gradient_descent`](@ref), [`GradientProblem`](@ref)
"""
mutable struct LevenbergMarquardtOptions{
    P,
    TStop<:StoppingCriterion,
    TRTM<:AbstractRetractionMethod,
    Tresidual_values,
    TJac,
    TGrad,
    Tparams<:Real,
} <: AbstractGradientOptions
    x::P
    stop::TStop
    retraction_method::TRTM
    residual_values::Tresidual_values
    candidate_residual_values::Tresidual_values
    jacF::TJac
    gradient::TGrad
    step_vector::TGrad
    last_stepsize::Tparams
    η::Tparams
    damping_term::Tparams
    damping_term_min::Tparams
    β::Tparams
    expect_zero_residual::Bool
    function LevenbergMarquardtOptions(
        M::AbstractManifold,
        initialX::P,
        initial_residual_values::Tresidual_values,
        initial_jacF::TJac,
        initial_gradient::TGrad=zero_vector(M, initialX);
        stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                              StopWhenGradientNormLess(1e-12) |
                                              StopWhenStepsizeLess(1e-12),
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
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
            initialX,
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
