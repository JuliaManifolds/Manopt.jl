"""
    AbstractRobustifierFunction <: Function

An abstract type to represent robustifiers, i.e., functions ``ρ: ℝ → ℝ``,
currently mainly used within [Levenberg-Marquardt](@ref).

Usually these should be twice continuously differentiable functions with

* ``ρ(0) = 0``
* ``ρ'(0) = 1``

to mimic the classical least squares behaviour around zero residuals.
and
* ``ρ'(x) < 1`` in outlier regions
* ``ρ''(x) < 0`` in outlier regions

Note that the robustifier is applioed to the squared residuals within the
nonlinear least squares framework, i.e., ``ρ(f_i(p)^2)``.
"""
abstract type AbstractRobustifierFunction <: Function end

@doc """
    NonlinearLeastSquaresObjective{E<:AbstractEvaluationType} <: AbstractManifoldObjective{T}

An objective to model the robustified nonlinear least squares problem

$(_problem(:NonLinearLeastSquares))

# Fields

* `objective`: a vector of [`AbstractVectorGradientFunction`](@ref)`{E}`s, one for each
  block component cost function ``F_i``, which might internally also be a vector of component costs ``(F_i)_j``,
  as well as their Jacobian ``J_{F_i}`` or a vector of gradients ``$(_tex(:grad)) (F_i)_j``
  depending on the specified [`AbstractVectorialType`](@ref)s.
* `robustifier`: a vector of [`AbstractRobustifierFunction`](@ref)`s`, one for each
  block component cost function ``F_i``.

# Constructors

    NonlinearLeastSquaresObjective(f, jacobian, range_dimension::Integer, robustifier=IdentityRobustifier(); kwargs...)
    NonlinearLeastSquaresObjective(vf::AbstractVectorGradientFunction, robustifier::AbstractRobustifierFunction=IdentityRobustifier())
    NonlinearLeastSquaresObjective(fs::Vector{<:AbstractVectorGradientFunction}, robustifiers::Vector{<:AbstractRobustifierFunction}=fill(IdentityRobustifier(), length(fs)))

# Arguments

* `f` the vectorial cost function ``f: $(_math(:Manifold)) → ℝ^m``
* `jacobian` the Jacobian, might also be a vector of gradients of the component functions of `f`
* `range_dimension::Integer` the number of dimensions `m` the function `f` maps into

These three can also be passed as a [`AbstractVectorGradientFunction`](@ref) `vf` already.

* `robustifier` the robustifier function(s) to use, by default the [`IdentityRobustifier`](@ref) (for each component),
    which corresponds to the classical nonlinear least squares problem.

# Keyword arguments

$(_kwargs(:evaluation))

As well as for the first variant of having a single block

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
        E <: AbstractEvaluationType,
        VF <: Vector{AbstractVectorGradientFunction{E}},
        RF <: Vector{<:AbstractRobustifierFunction},
    } <: AbstractManifoldFirstOrderObjective{E, VF}
    objective::VF
    robustifier::RF
    # block components case constructor
    function NonlinearLeastSquaresObjective(
            fs::VF,
            robustifiers::RV = fill(IdentityRobustifier(), length(fs)),
        ) where {E <: AbstractEvaluationType, VF <: Vector{AbstractVectorGradientFunction{E}}, RV <: Vector{<:AbstractRobustifierFunction}}
        # we need to check that the lengths match
        (length(fs) != length(robustifiers)) && throw(
            ArgumentError(
                "Number of functions ($(length(fs))) does not match number of robustifiers ($(length(robustifiers)))",
            ),
        )
        return new{E, VF, RV}(fs, robustifiers)
    end
    # single component case constructor
    function NonlinearLeastSquaresObjective(
            f::F,
            robustifier::R = IdentityRobustifier(),
        ) where {E <: AbstractEvaluationType, F <: AbstractVectorGradientFunction{E}, R <: AbstractRobustifierFunction}
        return new{E, Vector{F}, Vector{R}}([f,], [robustifier,])
    end
end

# the old single function constructor – TODO: remove? No but carefully document that this is just one special case that has a nicer shortcut and calls the vectorial one
function NonlinearLeastSquaresObjective(
        f,
        jacobian,
        range_dimension::Integer,
        robustifier::AbstractRobustifierFunction = IdentityRobustifier();
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        jacobian_tangent_basis::AbstractBasis = DefaultOrthonormalBasis(),
        jacobian_type::AbstractVectorialType = CoordinateVectorialType(jacobian_tangent_basis),
        function_type::AbstractVectorialType = FunctionVectorialType(),
    )
    vgf = VectorGradientFunction(
        f,
        jacobian,
        range_dimension;
        evaluation = evaluation,
        jacobian_type = jacobian_type,
        function_type = function_type,
    )
    return NonlinearLeastSquaresObjective(vgf, robustifier)
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

#
#
# Robustifiers

function get_robustifier_values end

"""
    (a, b, c) = get_robustifier_values(ρ::AbstractRobustifierFunction, x::Real)

Evaluate the robustifier ``ρ`` and its first two derivatives at `x`.

# Returns
A tuple `(a, b, c)` with
* `a = ρ(x)`
* `b = ρ'(x)`
* `c = ρ''(x)` (might be `missing` if not defined)
"""
get_robustifier_values(ρ::AbstractRobustifierFunction, x::Real)

"""
    RobustifierFunction{F, G, H} <: AbstractRobustifierFunction

A struct to represent a robustifier function ``ρ: ℝ → ℝ`` along with its first
and second derivative ``ρ'`` and ``ρ''``, respectively.

# Fields

* `ρ::F` : the robustifier function
* `ρ_prime::G` : the first derivative of the robustifier function
* `ρ_double_prime::H` : the second derivative of the robustifier function

# Constructor

    RobustifierFunction(ρ, ρ_prime, ρ_double_prime)

Generate a `RobustifierFunction` given the function `ρ` and its first and second derivative.
"""
struct RobustifierFunction{F <: Function, G <: Function, H <: Union{Function, Missing}} <: AbstractRobustifierFunction
    ρ::F
    ρ_prime::G
    ρ_double_prime::H
end

function get_robustifier_values(ρf::RobustifierFunction, x::Real)
    a = ρf.ρ(x)
    b = ρf.ρ_prime(x)
    c = ismissing(ρf.ρ_double_prime) ? missing : ρf.ρ_double_prime(x)
    return (a, b, c)
end

#
#
# Concrete cases

"""
    ArctanRobustifier <: AbstractRobustifierFunction

A robustifier that is based on the arctangent function.

The formula for the robustifier is given as
```math
ρ(x) = $(_tex(:rm, "atan"))(x)
```
and its first and second derivatives read as
```math
ρ'(x) = $(_tex(:frac, "1", "1 + x^2"))
```
and
```math
ρ''(x) = -$(_tex(:frac, "2x", "(1 + x^2)^2"))
```
"""
struct ArctanRobustifier <: AbstractRobustifierFunction end

function get_robustifier_values(::ArctanRobustifier, x::Real)
    (x == 0) && (return (0.0, 1.0, 0.0))
    a = atan(x)
    b = 1 / (1 + x^2)
    c = -2 * x / (1 + x^2)^2
    return (a, b, c)
end

"""
    CauchyRobustifier <: AbstractRobustifierFunction

A robustifier that is based on the Cauchy function. Note that robustifiers act on the
squared residuals within the nonlinear least squares framework, i.e., ``ρ(f_i(p)^2)``.

The formula for the Cauchy robustifier is given as
```math
ρ(x) = $(_tex(:log, "1 + x"))
```

and its first and second derivatives read as

```math
ρ'(x) = $(_tex(:frac, "1", "1 + x"))
```

and

```math
ρ''(x) = -$(_tex(:frac, "1", "(1 + x)^2"))
```
"""
struct CauchyRobustifier <: AbstractRobustifierFunction end

function get_robustifier_values(::CauchyRobustifier, x::Real)
    (x == 0) && (return (0.0, 1.0, -1.0))
    a = log(1 + x)
    b = 1 / (1 + x)
    c = -1 / (1 + x)^2
    return (a, b, c)

end

"""
ComposedRobustifierFunction{F<:AbstractRobustifierFunction, G<:AbstractRobustifierFunction} <: AbstractRobustifierFunction

A robustifier that is the composition of two robustifier functions ``ρ = ρ_1 ∘ ρ_2``.

For formulae for the first and second derivatives are
```math
ρ'(x) = ρ_1'(ρ_2(x)) ⋅ ρ_2'(x)
```
and
```math
ρ''(x) = ρ_1''(ρ_2(x)) ⋅ (ρ_2'(x))^2 + ρ_1'(ρ_2(x)) ⋅ ρ_2''(x)
```

# Fields
* `ρ1::F` : the first robustifier function
* `ρ2::G` : the second robustifier function

# Constructor

    ComposedRobustifierFunction(ρ1::F, ρ2::G) where {F<:AbstractRobustifierFunction, G<:AbstractRobustifierFunction}
    ρ1 ∘ ρ2

"""
struct ComposedRobustifierFunction{
        F <: AbstractRobustifierFunction,
        G <: AbstractRobustifierFunction,
    } <: AbstractRobustifierFunction
    ρ1::F
    ρ2::G
end

Base.:∘(rf1::AbstractRobustifierFunction, rf2::AbstractRobustifierFunction) =
    ComposedRobustifierFunction(rf1, rf2)

function get_robustifier_values(
        crf::ComposedRobustifierFunction, x::Real
    )
    (a2, b2, c2) = get_robustifier_values(crf.ρ2, x)
    (a1, b1, c1) = get_robustifier_values(crf.ρ1, a2)
    a = a1
    b = b1 * b2
    c = (ismissing(c1) || ismissing(c2)) ? missing : c1 * b2^2 + b1 * c2
    return (a, b, c)
end

"""
    HuberRobustifier <: AbstractRobustifierFunction

A robustifier that is based on the Huber function. Note that robustifiers act on the
squared residuals within the nonlinear least squares framework, i.e., ``ρ(f_i(p)^2)``.

The formula for the Huber robustifier is given as

```math
ρ(x) = $(
    _tex(
        :cases,
        "x & $(_tex(:text, "if ")) x ≤ 1",
        "2$(_tex(:sqrt, "x")) - 1 & $(_tex(:text, "if ")) x > 1"
    )
)
```

that is, its first and second derivatives read as

```math
ρ'(x) = $(
    _tex(
        :cases,
        "1 & $(_tex(:text, "if ")) x ≤ 1",
        "$(_tex(:frac, "1", raw"\sqrt{x}")) & $(_tex(:text, "if ")) x > 1"
    )
)
```

and

```math
ρ''(x) =  $(
    _tex(
        :cases,
        "0 & $(_tex(:text, "if ")) x ≤ 1",
        "-$(_tex(:frac, "1", "2 x^{3/2}")) & $(_tex(:text, "if ")) x > 1"
    )
)
```

If you want to use a different threshold `δ > 0`, use a
[`ScaledRobustifierFunction`](@ref) to scale the residuals accordingly, or even use the
shorthand `δ ∘ HuberRobustifier()`.
"""
struct HuberRobustifier <: AbstractRobustifierFunction end

function get_robustifier_values(::HuberRobustifier, x::Real)
    (x == 0) && (return (0.0, 1.0, 0.0))
    a = x <= 1 ? x : 2 * sqrt(x) - 1
    b = x <= 1 ? 1.0 : 1 / sqrt(x)
    c = x <= 1 ? 0.0 : -1 / (2 * x^(3 / 2))
    return (a, b, c)
end

"""
    IdentityRobustifier <: AbstractRobustifierFunction

A robustifier that is the identity function, i.e., ``ρ(x) = x``.

Its first and second derivatives read as ``ρ'(x) = 1`` and ``ρ''(x) = 0``.
"""
struct IdentityRobustifier <: AbstractRobustifierFunction end
get_robustifier_values(::IdentityRobustifier, x::Real) = (x, 1.0, 0.0)


"""
    ScaledRobustifierFunction{F<:AbstractRobustifierFunction, R <: Real} <: AbstractRobustifierFunction

A given robustifier function to scale the residuals a real value `scale` ``s``,
i.e. we consider ``ρ_s(f(p)^2) = ρ(s^2⋅f(p)^2)`` for some [`AbstractRobustifierFunction`](@ref) ``ρ``.
The function and its derivatives hence read as
* ``ρ_s(x) = s^2 ρ(x / s^2)``
* ``ρ_s'(x) = ρ'(x / s^2)``
* ``ρ_s''(x) = $(_tex(:frac, "1", "s^2")) ρ''(x / s^2)``

# Fields

* `robustifier::F` : the underlying robustifier function
* `scale::R` : the scaling factor `s`

# Constructor

    ScaledRobustifierFunction(robustifier::F, scale::R) where {F<:AbstractRobustifierFunction, R <: Real}
    scale ∘ robustifier

Generate a `ScaledRobustifierFunction` given a robustifier function and a scaling factor.
"""
struct ScaledRobustifierFunction{
        F <: AbstractRobustifierFunction,
        R <: Real,
    } <: AbstractRobustifierFunction
    robustifier::F
    scale::R
end

Base.:∘(s::Real, rf::AbstractRobustifierFunction) = ScaledRobustifierFunction(rf, s)
Base.:∘(s::Real, rf::ScaledRobustifierFunction) = ScaledRobustifierFunction(rf.robustifier, s * rf.scale)

function get_robustifier_values(srf::ScaledRobustifierFunction, x::Real)
    s2 = srf.scale^2
    (a, b, c) = get_robustifier_values(srf.robustifier, x / s2)
    a_scaled = s2 * a
    b_scaled = b
    c_scaled = ismissing(c) ? missing : c / s2
    return (a_scaled, b_scaled, c_scaled)
end

"""
    SoftL1Robustifier <: AbstractRobustifierFunction

A robustifier that is based on Soft ``ℓ_1`` norm.
Note that robustifiers act on the
squared residuals within the nonlinear least squares framework, i.e., ``ρ(f_i(p)^2)``.

The formula for the robustifier is given as

```math
ρ(x) = 2($(_tex(:sqrt, "1 + x")) - 1)
```
and its first and second derivatives read as
```math
ρ'(x) = $(_tex(:frac, "1", "$(_tex(:sqrt, "1 + x"))"))
```
and
```math
ρ''(x) = -$(_tex(:frac, "1", "2 (1 + x)^{3/2}"))
```
"""
struct SoftL1Robustifier <: AbstractRobustifierFunction end

function get_robustifier_values(::SoftL1Robustifier, x::Real)
    (x == 0) && (return (0.0, 1.0, -0.5))
    s = sqrt(1 + x)
    a = 2 * (s - 1)
    b = 1 / s
    c = -1 / (2 * s * (1 + x))
    return (a, b, c)
end

"""
    TolerantRobustifier <: AbstractRobustifierFunction

A robustifier that is based on the tolerant function.

The formula for the robustifier is given as
```math
ρ_{a,b}(x) = b$(_tex(:log, "1+ $(_tex(:rm, "e")){(s-a)/b}")) - b$(_tex(:log, "1 + $(_tex(:rm, "e")){-a/b}"))
```
and its first and second derivatives read as
```math
ρ'_{a,b}(x) = $(_tex(:frac, "1", "1 + $(_tex(:rm, "e")){(a - x)/b}"))
```

and

```math
ρ''_{a,b}(x) = $(_tex(:frac, "1", "4b$(_tex(:rm, "cosh"))^2$(_tex(:Bigl))($(_tex(:frac, "(a - x)", "2b"))$(_tex(:Bigr)))"))
```
"""
struct TolerantRobustifier <: AbstractRobustifierFunction
    a::Real
    b::Real
    function TolerantRobustifier(a::Real, b::Real)
        (b <= 0) && throw(ArgumentError("Parameter b must be strictly positive, received $b"))
        return new(a, b)
    end
end

function get_robustifier_values(trf::TolerantRobustifier, x::Real)
    a = trf.a
    b = trf.b
    exp_term1 = exp((x - a) / b)
    exp_term2 = exp(-a / b)
    s1 = log(1 + exp_term1)
    s2 = log(1 + exp_term2)
    val = b * (s1 - s2)
    deriv1 = 1 / (1 + exp((a - x) / b))
    deriv2 = 1 / (4 * b * cosh((a - x) / (2b))^2)
    return (val, deriv1, deriv2)
end

"""
    TukeyRobustifier <: AbstractRobustifierFunction

A robustifier that is based on the Tukey function. Note that robustifiers act on the
squared residuals within the nonlinear least squares framework, i.e., ``ρ(f_i(p)^2)``.

The formula for the Tukey robustifier is given as
```math
ρ(x) = $(
    _tex(
        :cases,
        "$(_tex(:frac, "1", "3"))(1-(1-x)^3) & $(_tex(:text, "if ")) x ≤ 1",
        "$(_tex(:frac, "1", "3")) & $(_tex(:text, "if ")) x > 1"
    )
)
```
that is, its first and second derivatives read as
```math
ρ'(x) = $(
    _tex(
        :cases,
        "(1 - x)^2 & $(_tex(:text, "if ")) x ≤ 1",
        "0 & $(_tex(:text, "if ")) x > 1"
    )
)
```
and
```math
ρ''(x) =  $(
    _tex(
        :cases,
        "-2(1 - x) & $(_tex(:text, "if ")) x ≤ 1",
        "0 & $(_tex(:text, "if ")) x > 1"
    )
)
```
"""
struct TukeyRobustifier <: AbstractRobustifierFunction end

function get_robustifier_values(::TukeyRobustifier, x::Real)
    (x == 0) && (return (0.0, 1.0, -2.0))
    if x <= 1
        a = (1 / 3) * (1 - (1 - x)^3)
        b = (1 - x)^2
        c = 2 * (x - 1)
        return (a, b, c)
    else
        return (1 / 3, 0.0, 0.0)
    end
end

@doc """
    LevenbergMarquardtState{P,T} <: AbstractGradientSolverState

Describes a Gradient based descent algorithm, with

# Fields

A default value is given in brackets if a parameter can be left out in initialization.

$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:retraction_method))
* `residual_values`:      value of ``F`` calculated in the solver setup or the previous iteration
* `candidate_residual_values`: value of ``F`` for the current proposal point
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
        Pr,
        St,
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
    sub_problem::Pr
    sub_state::St
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
            linear_subsolver! = nothing, #remove on next breaking release
            sub_problem::Pr = linear_subsolver!, # todo: change default?
            sub_state::St = InplaceEvaluation(),
        ) where {P, Tresidual_values, TJac, TGrad, Pr, St}
        if η <= 0 || η >= 1
            throw(ArgumentError("Value of η must be strictly between 0 and 1, received $η"))
        end
        if linear_subsolver! !== nothing
            @warn "The keyword argument `linear_subsolver!` is deprecated and will be removed in future releases. Please use `sub_problem` and `sub_state` instead."
            sub_problem = linear_subsolver!
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
        SC = typeof(stopping_criterion)
        RM = typeof(retraction_method)
        return new{
            P, SC, RM, Tresidual_values, TJac, TGrad, Tparams, Pr, St,
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
            sub_problem,
            sub_state,
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

#
#
# --- Subproblems ----

#
# ----- A cost/grad objective to e.g. do CG, or Gauß-Newton ----


abstract type AbstractLevenbergMarquardtSurrogateObjective{E<:AbstractEvaluationType, NLSO <: NonlinearLeastSquaresObjective{E}} <: AbstractManifoldFirstOrderObjective{E, NLSO} end

@doc """
    LevenbergMarquardtSurrogateObjective{E<:AbstractEvaluationType, VF<:AbstractManifoldFirstOrderObjective{E}, R} <: AbstractManifoldFirstOrderObjective{E, VF}

Given an [`NonlinearLeastSquaresObjective`](@ref) `objective` and a damping term `damping_term`,
this objective represents the penalized objective for the sub-problem to solve within every step
of the Levenberg-Marquardt algorithm given by

```math
μ(X) = $(_tex(:frac, "1", "2"))$(_tex(:norm, "y + $(_tex(:Cal, "L"))(X)"; size = "big"))^2
  + $(_tex(:frac, "λ", "2"))$(_tex(:norm, "X"))^2
```

where ``X ∈ $(_math(:TangentSpace))`` is the new step to compute.
Set ``α = 1 - $(_tex(:sqrt, "1 + 2 $(_tex(:frac, "ρ''(p)", "ρ'(p)"))$(_tex(:norm, "F(p)"; index = "2"))^2"))``.

Then we have  ``y = $(_tex(:frac, _tex(:sqrt, "ρ(p)"), "1-α"))F(p)``.

and ``$(_tex(:Cal, "L")) = CJ_F^*(p)[X]`` is a linear operator using the adjoint of the Jacobian and
```math
C = $(_tex(:sqrt, "ρ'(p)"))(I-αP), $(_tex(:qquad)) P = $(_tex(:frac, "F(p)F(p)^" * _tex(:rm, "T"), _tex(:norm, "F(p)"; index = "2") * "^2")),
```

## Fields

* `objective`: the [`NonlinearLeastSquaresObjective`](@ref) to penalize
* `penalty`: the damping term ``λ``
"""
mutable struct LevenbergMarquardtSurrogatePenaltyObjective{
        E <: AbstractEvaluationType, NLSO <: NonlinearLeastSquaresObjective{E}, R,
    } <: AbstractLevenbergMarquardtSurrogateObjective{E, NLSO}
    objective::NLSO
    penalty::R
end

get_objective(lmsco::LevenbergMarquardtSurrogatePenaltyObjective) = lmsco.objective

# TODO: Implement a function get_linear_operator that returns the linear operator A
# as a matric with respect to a certain basis of the tangent space

# TODO: Find a better name ?
"""
    evaluate_linear_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtSurrogatePenaltyObjective, p, X; penalty = 0,
    )
    evaluate_linear_operator(
        M::AbstractManifold, lmsco::o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X; penalty = 0,
    )

Compute the linear operator ``$(_tex(:Cal, "A"))` corresponding to the optimality conditions of the
Levenberg-Marquardt surrogate objective, i.e.,
```math
$(_tex(:Cal, "A"))(X) = $(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))(X) + λX
= J_F^*(p)$(_tex(:bigl))[ C^T C J_F(p)[X] $(_tex(:bigr))] + λX,
```
where ``λ = `penalty` is a damping parameter.

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also [`evaluate_tangent_vector`](@ref) for evaluating the corresponding vector field
"""
function evaluate_linear_operator(
        M::AbstractManifold, lmsco::AbstractLevenbergMarquardtSurrogateObjective, p, X;
        penalty = 0,
    )
    Y = zero_vector(M, p)
    return evaluate_linear_operator!(M, Y, lmsco, p, X; penalty = penalty)
end
function evaluate_linear_operator!(
        M::AbstractManifold, Y, lmsco::AbstractLevenbergMarquardtSurrogateObjective, p, X;
        penalty = 0,
    )
    nlso = get_objective(lmsco)
    # For every block
    zero_vector!(M, Y, p)
    Z = copy(M, p, Y)
    for (o,r) in zip(nlso.objective, nlso.robustifier)
        evaluate_linear_operator!(M, Z, o, r, p, X; penalty = 0)
        Y .+= Z
    end
    # Finally add the damping term
    (penalty != 0) && (Y .+= penalty * X)
    return Y
end
# for a single block – the actual formula
function evaluate_linear_operator!(
        M::AbstractManifold, Y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X;
        penalty = 0,
    )
    a = get_value(M, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    α = 1 - sqrt(1 + 2 * (ismissing(ρ_double_prime) ? 0.0 : ρ_double_prime / ρ_prime) * F_p_norm2)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
    get_jacobian!(M, Y, o, p, X)
    # Compute C^TCa = C^2 a (inplace of a)
    a .= ρ_prime * (1 - α * (a*a') ./ F_p_norm2)^2 * a
    # Now apply the adjoint
    get_adjoint_jacobian!(M, Y, o, p, a)
    # Finally add the damping term
    (penalty != 0) && (Y .+= penalty * X)
    return Y
end

# TODO: implement a function get_tangent_vector that returns the computed tangent vector
# in coefficients of a certain basis of the tangent space
# TODO: Find a better name ?
"""
    evaluate_tangent_vector(
        M::AbstractManifold, lmsco::LevenbergMarquardtSurrogatePenaltyObjective, p
    )
    evaluate_tangent_vector!(
        M::AbstractManifold, X, lmsco::LevenbergMarquardtSurrogatePenaltyObjective, p
    )
    evaluate_tangent_vector!(
        M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p
    )

Compute the linear operator tangent vector ``X`` corresponding to the optimality conditions of the
Levenberg-Marquardt surrogate objective, i.e.,
```math
X = - J_F^*(p)[ C^T y],
```

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also [`evaluate_linear_operator`](@ref) for evaluating the corresponding linear operator of the linear system
"""
function evaluate_tangent_vector(
        M::AbstractManifold, lmsco::AbstractLevenbergMarquardtSurrogateObjective, p
    )
    X = zero_vector(M, p)
    return evaluate_tangent_vector!(M, X, lmsco, p)
end
function evaluate_tangent_vector!(
        M::AbstractManifold, X, lmsco::AbstractLevenbergMarquardtSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    # For every block
    zero_vector!(M, X, p)
    Z = copy(M, p, X)
    for (o,r) in zip(nlso.objective, nlso.robustifier)
        evaluate_tangent_vector!(M, Z, o, r, p)
        X .+= Z
    end
    return X
end
# for a single block – the actual formula
function evaluate_tangent_vector!(
        M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p
    )
    y = get_value(M, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    α = 1 - sqrt(1 + 2 * (ismissing(ρ_double_prime) ? 0.0 : ρ_double_prime / ρ_prime) * F_p_norm2)
    # Compute y = (sqrt(ρ(p)) / (1-α)) F(p)
    y .= (sqrt(ρ_value) / (1 - α)) * y
    # Now compute J_F^*(p)[C^T y] (inplace of y)
    y .= ρ_prime * (1 - α * (y*y') ./ F_p_norm2) * y
    # Now apply the adjoint and negate
    get_adjoint_jacobian!(M, X, o, p, y)
    X .*= -1
    return X
end


@doc """
    LevenbergMarquardtSurrogateObjective{E<:AbstractEvaluationType, VF<:AbstractManifoldFirstOrderObjective{E}, R} <: AbstractManifoldFirstOrderObjective{E, VF}

Given an [`NonlinearLeastSquaresObjective`](@ref) `objective` and a damping term `damping_term`,
this objective represents the penalized objective for the sub-problem to solve within every step
of the Levenberg-Marquardt algorithm given by

```math
σ(X) = $(_tex(:frac, "1", "2"))$(_tex(:norm, "y + $(_tex(:Cal, "L"))(X)"; size = "big"))^2
$(_tex(:quad))$(_tex(:text, " such that "))$(_tex(:quad))$(_tex(:norm, "X")) ≤ Δ
```

where ``X ∈ $(_math(:TangentSpace))`` is the new step to compute.
Set ``α = 1 - $(_tex(:sqrt, "1 + 2 $(_tex(:frac, "ρ''(p)", "ρ'(p)"))$(_tex(:norm, "F(p)"; index = "2"))^2"))``.

Then we have  ``y = $(_tex(:frac, _tex(:sqrt, "ρ(p)"), "1-α"))F(p)``.

and ``$(_tex(:Cal, "L")) = CJ_F^*(p)[X]`` is a linear operator using the adjoint of the Jacobian and
```math
C = $(_tex(:sqrt, "ρ'(p)"))(I-αP), $(_tex(:qquad)) P = $(_tex(:frac, "F(p)F(p)^" * _tex(:rm, "T"), _tex(:norm, "F(p)"; index = "2") * "^2")),
```

## Fields

* `objective`: the [`NonlinearLeastSquaresObjective`](@ref) to penalize
* `radius`: the trust region radius ``Δ``
"""
mutable struct LevenbergMarquardtSurrogateConstrainedObjective{
        E <: AbstractEvaluationType, NLSO <: NonlinearLeastSquaresObjective{E}, R,
    } <: AbstractLevenbergMarquardtSurrogateObjective{E, NLSO}
    objective::NLSO
    radius::R
end

get_objective(lmsco::LevenbergMarquardtSurrogateConstrainedObjective) = lmsco.objective
