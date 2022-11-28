@doc raw"""
    LevenbergMarquardt(M, F, jacF, x)

Solve an optimization problem of the form

```math
\operatorname{minimize}_{x ∈ \mathcal M} \frac{1}{2} \lVert F(x) \rVert^2,
```

where ``F: \mathcal M \to ℝ^d`` is a continuously differentiable function,
using the Riemannian Levenberg-Marquardt algorithm [^Peeters1993].
The implementation follows [^Adachi2022].

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F: \mathcal M→ℝ^d``
* `jacF` – the Jacobian of ``F``
* `x` – an initial value ``x ∈ \mathcal M``
* `num_components` -- length of the vector returned by the cost function (`d`).

# Optional
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient works by allocation (default) form `gradF(M, x)`
  or [`MutatingEvaluation`](@ref) in place, i.e. is of the form `gradF!(M, X, x)`.
* `retraction_method` – (`default_retraction_method(M)`) a `retraction(M,x,ξ)` to use.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(200), `[`StopWhenGradientNormLess`](@ref)`(1e-12))`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details

# References

[^Adachi2022]:
    > S. Adachi, T. Okuno, and A. Takeda, “Riemannian Levenberg-Marquardt Method with Global
    > and Local Convergence Properties.” arXiv, Oct. 01, 2022.
    > doi: [10.48550/arXiv.2210.00253](https://doi.org/10.48550/arXiv.2210.00253).
[^Peeters1993]:
    > R. L. M. Peeters, “On a Riemannian version of the Levenberg-Marquardt algorithm,”
    > VU University Amsterdam, Faculty of Economics, Business Administration and Econometrics,
    > Serie Research Memoranda 0011, 1993.
    > link: [https://econpapers.repec.org/paper/vuawpaper/1993-11.htm](https://econpapers.repec.org/paper/vuawpaper/1993-11.htm).
"""
function LevenbergMarquardt(
    M::AbstractManifold, F::TF, gradF::TDF, x, num_components::Int; kwargs...
) where {TF,TDF}
    x_res = copy(M, x)
    return LevenbergMarquardt!(M, F, gradF, x_res, num_components; kwargs...)
end

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
    GradientDescentOptions{P,T} <: AbstractGradientOptions

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
) where {P}
    return LevenbergMarquardtOptions{P}(
        M,
        x,
        initial_Fval,
        initial_jacF,
        initial_gradient;
        stopping_criterion,
        retraction_method,
    )
end

@doc raw"""
    LevenbergMarquardt!(M, F, jacF, x)


For more options see [`LevenbergMarquardt`](@ref).
"""
function LevenbergMarquardt!(
    M::AbstractManifold,
    F::TF,
    jacF::TDF,
    x,
    num_components::Int;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(200) |
                                          StopWhenGradientNormLess(1e-12) |
                                          StopWhenStepsizeLess(1e-12),
    debug=[DebugWarnIfCostIncreases()],
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    kwargs..., #collect rest
) where {TF,TDF}
    p = NonlinearLeastSquaresProblem(M, F, jacF, num_components; evaluation=evaluation)
    o = LevenbergMarquardtOptions(
        M,
        x,
        similar(x, num_components), # TODO: rethink this?
        similar(x, num_components, manifold_dimension(M)); # TODO: rethink this?
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
    )
    o = decorate_options(o; debug=debug, kwargs...)
    return get_solver_return(solve(p, o))
end
#
# Solver functions
#
function initialize_solver!(
    p::NonlinearLeastSquaresProblem{AllocatingEvaluation}, o::LevenbergMarquardtOptions
)
    o.Fval = p.F(p.M, o.x)
    o.gradient = get_gradient(p, o.x)
    return o
end
function initialize_solver!(
    p::NonlinearLeastSquaresProblem{MutatingEvaluation}, o::LevenbergMarquardtOptions
)
    p.F(p.M, o.Fval, o.x)
    o.gradient = get_gradient(p, o.x)
    return o
end
function step_solver!(
    p::NonlinearLeastSquaresProblem{Teval}, o::LevenbergMarquardtOptions, iter::Integer
) where {Teval<:AbstractEvaluationType}
    # o.Fval is either initialized by initialize_solver! or taken from the previous iteraion
    if Teval === AllocatingEvaluation
        o.jacF = p.jacobian!!(p.M, o.x; B_dom=p.jacB)
    else
        p.jacobian!!(p.M, o.jacF, o.x; B_dom=p.jacB)
    end
    λk = o.damping_term * norm(o.Fval)

    JJ = transpose(o.jacF) * o.jacF + λk * I
    # `cholesky` is technically not necessary but it's the fastest method to solve the
    # problem because JJ is symmetric positive definite
    grad_f_c = transpose(o.jacF) * o.Fval
    sk = cholesky(JJ) \ -grad_f_c
    get_vector!(p.M, o.gradient, o.x, grad_f_c, p.jacB)

    get_vector!(p.M, o.step_vector, o.x, sk, p.jacB)
    o.last_stepsize = norm(p.M, o.x, o.step_vector)
    temp_x = retract(p.M, o.x, o.step_vector, o.retraction_method)

    normFk2 = norm(o.Fval)^2
    if Teval === AllocatingEvaluation
        o.Fval_temp = p.F(p.M, temp_x)
    else
        p.F(p.M, o.Fval_temp, temp_x)
    end

    ρk =
        2 * (normFk2 - norm(o.Fval_temp)^2) / (
            -2 * inner(p.M, o.x, o.gradient, o.step_vector) - norm(o.jacF * sk)^2 -
            λk * norm(sk)
        )
    if ρk >= o.η
        copyto!(p.M, o.x, temp_x)
        copyto!(o.Fval, o.Fval_temp)
        if !o.flagnz
            o.damping_term = max(o.damping_term_min, o.damping_term / o.β)
        end
    else
        o.damping_term *= o.β
    end
    return o
end

function _get_last_stepsize(::Problem, o::LevenbergMarquardtOptions, ::Val{false}, vars...)
    return o.last_stepsize
end
