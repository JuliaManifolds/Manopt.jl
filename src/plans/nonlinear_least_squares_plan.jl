@doc """
    NonlinearLeastSquaresObjective{E<:AbstractEvaluationType} <: AbstractManifoldObjective{T}

An objective to model the nonlinear least squares problem

$(_problem(:NonLinearLeastSquares))

Specify a nonlinear least squares problem

# Fields

* `objective`: a [`AbstractVectorGradientFunction`](@ref)`{E}` containing both the vector of
  cost functions ``f_i`` (or a function returning a vector of costs) as well as their
  gradients ``$(_tex(:grad)) f_i`` (or Jacobian of the vector-valued function).

This `NonlinearLeastSquaresObjective` then has the same [`AbstractEvaluationType`](@ref) `T`
as the (inner) `objective`.

# Constructors

    NonlinearLeastSquaresObjective(f, jacobian, range_dimension::Integer; kwargs...)
    NonlinearLeastSquaresObjective(vf::AbstractVectorGradientFunction)

# Arguments

* `f` the vectorial cost function ``f: $(_math(:Manifold))nifold))) → ℝ^m``
* `jacobian` the Jacobian, might also be a vector of gradients of the component functions of `f`
* `range_dimension::Integer` the number of dimensions `m` the function `f` maps into

These three can also be passed as a [`AbstractVectorGradientFunction`](@ref) `vf` already.

# Keyword arguments

$(_kwargs(:evaluation))
* `function_type::`[`AbstractVectorialType`](@ref)`=`[`FunctionVectorialType`](@ref)`()`: specify
  the format the residuals are given in. By default a function returning a vector.
* `jacobian_tangent_basis::AbstractBasis=DefaultOrthonormalBasis()`; shortcut to specify
  the basis the Jacobian matrix is build with.
* `jacobian_type::`[`AbstractVectorialType`](@ref)`=`[`CoordinateVectorialType`](@ref)`(jacobian_tangent_basis)`:
  specify the format the Jacobian is given in. By default a matrix of the differential with
  respect to a certain basis of the tangent space.

# See also

[`LevenbergMarquardt`](@ref), [`LevenbergMarquardtState`](@ref)
"""
struct NonlinearLeastSquaresObjective{
        E <: AbstractEvaluationType, F <: AbstractVectorGradientFunction{E},
    } <: AbstractManifoldFirstOrderObjective{E, F}
    objective::F
end

function NonlinearLeastSquaresObjective(
        f,
        jacobian,
        range_dimension::Integer;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        jacobian_tangent_basis::AbstractBasis = DefaultOrthonormalBasis(),
        jacobian_type::AbstractVectorialType = CoordinateVectorialType(jacobian_tangent_basis),
        function_type::AbstractVectorialType = FunctionVectorialType(),
        kwargs...,
    )
    vgf = VectorGradientFunction(
        f,
        jacobian,
        range_dimension;
        evaluation = evaluation,
        jacobian_type = jacobian_type,
        function_type = function_type,
    )
    return NonlinearLeastSquaresObjective(vgf; kwargs...)
end

# Cost
function get_cost(
        M::AbstractManifold,
        nlso::NonlinearLeastSquaresObjective{
            E, <:AbstractVectorFunction{E, <:ComponentVectorialType},
        },
        p;
        kwargs...,
    ) where {E <: AbstractEvaluationType}
    v = 0.0
    for i in 1:length(nlso.objective)
        v += abs(get_value(M, nlso.objective, p, i))^2
    end
    v /= 2
    return v
end
function get_cost(
        M::AbstractManifold,
        nlso::NonlinearLeastSquaresObjective{
            E, <:AbstractVectorFunction{E, <:FunctionVectorialType},
        },
        p;
        value_cache = get_value(M, nlso.objective, p),
    ) where {E <: AbstractEvaluationType}
    return sum(abs2, value_cache) / 2
end

function get_jacobian(
        M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...
    )
    J = zeros(length(nlso.objective), manifold_dimension(M))
    get_jacobian!(M, J, nlso, p; kwargs...)
    return J
end
# The jacobian is now just a pass-through
function get_jacobian!(
        M::AbstractManifold, J, nlso::NonlinearLeastSquaresObjective, p; kwargs...
    )
    get_jacobian!(M, J, nlso.objective, p; kwargs...)
    return J
end
function get_gradient(
        M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...
    )
    X = zero_vector(M, p)
    return get_gradient!(M, X, nlso, p; kwargs...)
end
function get_gradient!(
        M::AbstractManifold,
        X,
        nlso::NonlinearLeastSquaresObjective,
        p;
        basis = get_basis(nlso.objective.jacobian_type),
        jacobian_cache = get_jacobian(M, nlso, p; basis = basis),
        value_cache = get_residuals(M, nlso, p),
    )
    return get_vector!(M, X, p, transpose(jacobian_cache) * value_cache, basis)
end

#
#
# --- Residuals
_doc_get_residuals_nlso = """
    get_residuals(M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p)
    get_residuals!(M::AbstractManifold, V, nlso::NonlinearLeastSquaresObjective, p)

Compute the vector of residuals ``f_i(p)``, ``i=1,…,m`` given the manifold `M`,
the [`NonlinearLeastSquaresObjective`](@ref) `nlso` and a current point ``p`` on `M`.
"""

@doc "$(_doc_get_residuals_nlso)"
get_residuals(M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...)

function get_residuals(
        M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...
    )
    V = zeros(length(nlso.objective))
    return get_residuals!(M, V, nlso, p; kwargs...)
end

@doc "$(_doc_get_residuals_nlso)"
get_residuals!(M::AbstractManifold, V, nlso::NonlinearLeastSquaresObjective, p; kwargs...)

function get_residuals!(
        M::AbstractManifold,
        V,
        nlso::NonlinearLeastSquaresObjective{
            E, <:AbstractVectorFunction{E, <:ComponentVectorialType},
        },
        p;
        kwargs...,
    ) where {E <: AbstractEvaluationType}
    for i in 1:length(nlso.objective)
        V[i] = get_value(M, nlso.objective, p, i)
    end
    return V
end
function get_residuals!(
        M::AbstractManifold,
        V,
        nlso::NonlinearLeastSquaresObjective{
            E, <:AbstractVectorFunction{E, <:FunctionVectorialType},
        },
        p,
    ) where {E <: AbstractEvaluationType}
    get_value!(M, V, nlso.objective, p)
    return V
end

@doc """
    LevenbergMarquardtState{P,T} <: AbstractGradientSolverState

Describes a Gradient based descent algorithm, with

# Fields

A default value is given in brackets if a parameter can be left out in initialization.

$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:retraction_method))
* `residual_values`:      value of ``F`` calculated in the solver setup or the previous iteration
* `residual_values_temp`: value of ``F`` for the current proposal point
$(_fields(:stopping_criterion; name = "stop"))
* `jacobian`:                 the current Jacobian of ``F``
* `gradient`:             the current gradient of ``F``
* `step_vector`:          the tangent vector at `x` that is used to move to the next point
* `last_stepsize`:        length of `step_vector`
* `η`:                    Scaling factor for the sufficient cost decrease threshold required
  to accept new proposal points. Allowed range: `0 < η < 1`.
* `damping_term`:         current value of the damping term
* `damping_term_min`:     initial (and also minimal) value of the damping term
* `β`:                    parameter by which the damping term is multiplied when the current
  new point is rejected
* `expect_zero_residual`: if true, the algorithm expects that the value of
  the residual (objective) at minimum is equal to 0.
* `linear_subsolver!`:    a function with three arguments `sk, JJ, grad_f_c`` that solves the
  linear subproblem `sk .= JJ \\ grad_f_c`, where `JJ` is (up to numerical issues) a
  symmetric positive definite matrix. Default value is [`default_lm_lin_solve!`](@ref).

# Constructor

    LevenbergMarquardtState(M, initial_residual_values, initial_jacobian; kwargs...)

Generate the Levenberg-Marquardt solver state.

# Keyword arguments

The following fields are keyword arguments

* `β=5.0`
* `damping_term_min=0.1`
* `η=0.2`,
* `expect_zero_residual=false`
* `initial_gradient=`$(_link(:zero_vector))
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-12)`$(_sc(:Any))[`StopWhenStepsizeLess`](@ref)`(1e-12)"))

# See also

[`gradient_descent`](@ref), [`LevenbergMarquardt`](@ref)
"""
mutable struct LevenbergMarquardtState{
        P,
        TStop <: StoppingCriterion,
        TRTM <: AbstractRetractionMethod,
        Tresidual_values,
        TJac,
        TGrad,
        Tparams <: Real,
        TLS,
    } <: AbstractGradientSolverState
    p::P
    stop::TStop
    retraction_method::TRTM
    residual_values::Tresidual_values
    candidate_residual_values::Tresidual_values
    jacobian::TJac
    X::TGrad
    step_vector::TGrad
    last_stepsize::Tparams
    η::Tparams
    damping_term::Tparams
    damping_term_min::Tparams
    β::Tparams
    expect_zero_residual::Bool
    last_step_successful::Bool
    linear_subsolver!::TLS
    function LevenbergMarquardtState(
            M::AbstractManifold,
            initial_residual_values::Tresidual_values,
            initial_jacobian::TJac;
            p::P = rand(M),
            X::TGrad = zero_vector(M, p),
            stopping_criterion::StoppingCriterion = StopAfterIteration(200) |
                StopWhenGradientNormLess(1.0e-12) |
                StopWhenStepsizeLess(1.0e-12),
            retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
            η::Real = 0.2,
            damping_term_min::Real = 0.1,
            β::Real = 5.0,
            expect_zero_residual::Bool = false,
            linear_subsolver!::TLS = (default_lm_lin_solve!),
        ) where {P, Tresidual_values, TJac, TGrad, TLS}
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
            TLS,
        }(
            p,
            stopping_criterion,
            retraction_method,
            initial_residual_values,
            copy(initial_residual_values),
            initial_jacobian,
            X,
            allocate(M, X),
            zero(Tparams),
            η,
            damping_term_min,
            damping_term_min,
            β,
            expect_zero_residual,
            true,
            linear_subsolver!,
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

    ## Stopping criterion

    $(status_summary(lms.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
