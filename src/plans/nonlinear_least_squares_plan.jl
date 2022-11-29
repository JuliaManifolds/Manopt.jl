
@doc raw"""
    NonlinearLeastSquaresProblem{T} <: Problem{T}

A type for nonlinear least squares problems.
`T` is a [`AbstractEvaluationType`](@ref) for the gradient function.


specify a problem for gradient based algorithms.

# Fields
* `M`        – a manifold ``\mathcal M``
* `F`        – a function ``F: \mathcal M → ℝ^d`` to minimize
* `jacF!!`   – Jacobian of the function ``F``
* `jacB`     – the basis of tangent space used for computing the Jacobian.
* `num_components` – number of values returned by `F` (equal to `d`).

Depending on the [`AbstractEvaluationType`](@ref) `T` the gradient has to be provided

* as a function `x -> X` that allocates memory for `X` itself for an [`AllocatingEvaluation`](@ref)
* as a function `(X,x) -> X` that work in place of `X` for an [`MutatingEvaluation`](@ref)

# Constructors
    GradientProblem(M, cost, gradient; evaluation=AllocatingEvaluation(), jacB=DefaultOrthonormalBasis())

# See also
[`LevenbergMarquardt`](@ref), [`GradientDescentOptions`](@ref)
"""
struct NonlinearLeastSquaresProblem{T,mT<:AbstractManifold,TF,TJ,TB<:AbstractBasis} <:
       Problem{T}
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
    Fval = zeros(P.num_components)
    P.F(P.M, Fval, p)
    return 1//2 * norm(Fval)^2
end

function get_gradient(p::NonlinearLeastSquaresProblem{AllocatingEvaluation}, x)
    Jval = p.jacobian!!(p.M, x; B_dom=p.jacB)
    Fval = p.F(p.M, x)
    return get_vector(p.M, x, transpose(Jval) * Fval, p.jacB)
end
function get_gradient(p::NonlinearLeastSquaresProblem{MutatingEvaluation}, x)
    Jval = zeros(p.num_components, manifold_dimension(p.M))
    p.jacobian!!(p.M, Jval, x; B_dom=p.jacB)
    Fval = zeros(p.num_components)
    p.F(p.M, Fval, x)
    return get_vector(p.M, x, transpose(Jval) * Fval, p.jacB)
end

function get_gradient!(p::NonlinearLeastSquaresProblem{AllocatingEvaluation}, X, x)
    Jval = p.jacobian!!(p.M, x; B_dom=p.jacB)
    Fval = p.F(p.M, x)
    return get_vector!(p.M, X, x, transpose(Jval) * Fval, p.jacB)
end

function get_gradient!(p::NonlinearLeastSquaresProblem{MutatingEvaluation}, X, x)
    Jval = zeros(p.num_components, manifold_dimension(p.M))
    p.jacobian!!(p.M, Jval, x; B_dom=p.jacB)
    Fval = zeros(p.num_components)
    p.F(p.M, Fval, x)
    return get_vector!(p.M, X, x, transpose(Jval) * Fval, p.jacB)
end

@doc raw"""
    LevenbergMarquardtOptions{P,T} <: AbstractGradientOptions

Describes a Gradient based descent algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` – a point (of type `P`) on a manifold as starting point
* `Fval` -- value of ``F`` calculated in the solver setup or the previous iteration
* `Fval_temp` -- value of ``F`` for the current proposal point
* `jacF` – the current Jacobian of ``F``
* `stopping_criterion` – ([`StopAfterIteration`](@ref)`(200)`) a [`StoppingCriterion`](@ref)
* `retraction_method` – (`default_retraction_method(M)`) the retraction to use, defaults to
  the default set for your manifold.
* `flagnz` -- if false, the algorithm expects that the value of residual at mimimum is equal
  to 0.

# Constructor

    LevenbergMarquardtOptions(M, x; initial_vector=zero_vector(M, x), kwargs...)

Generate gradient descent options, where `initial_vector` can be used to set the tangent vector to store the gradient to a certain type.
All following fields are keyword arguments.

# See also
[`gradient_descent`](@ref), [`GradientProblem`](@ref)
"""
mutable struct LevenbergMarquardtOptions{
    P,TStop<:StoppingCriterion,TRTM<:AbstractRetractionMethod,TFval,TJac,TGrad,Tparams<:Real
} <: AbstractGradientOptions
    x::P
    stop::TStop
    retraction_method::TRTM
    Fval::TFval
    Fval_temp::TFval
    jacF::TJac
    gradient::TGrad
    step_vector::TGrad
    last_stepsize::Tparams
    η::Tparams
    damping_term::Tparams
    damping_term_min::Tparams
    β::Tparams
    flagnz::Bool
    function LevenbergMarquardtOptions{P}(
        M::AbstractManifold,
        initialX::P,
        initial_Fval::TFval,
        initial_jacF::TJac,
        initial_gradient::TGrad;
        stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                              StopWhenGradientNormLess(1e-12) |
                                              StopWhenStepsizeLess(1e-12),
        retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
        η::Real=0.2,
        damping_term_min::Real=0.1,
        β::Real=5.0,
        flagnz::Bool=true,
    ) where {P,TFval,TJac,TGrad}
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
            P,typeof(stopping_criterion),typeof(retraction_method),TFval,TJac,TGrad,Tparams
        }(
            initialX,
            stopping_criterion,
            retraction_method,
            initial_Fval,
            copy(initial_Fval),
            initial_jacF,
            initial_gradient,
            allocate(M, initial_gradient),
            zero(Tparams),
            η,
            damping_term_min,
            damping_term_min,
            β,
            flagnz,
        )
    end
end

function LevenbergMarquardtOptions(
    M::AbstractManifold,
    x::P,
    initial_Fval,
    initial_jacF,
    initial_gradient=zero_vector(M, x);
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(1e-12) |
                                          StopWhenStepsizeLess(1e-12),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    kwargs...,
) where {P}
    return LevenbergMarquardtOptions{P}(
        M,
        x,
        initial_Fval,
        initial_jacF,
        initial_gradient;
        stopping_criterion,
        retraction_method,
        kwargs...,
    )
end
