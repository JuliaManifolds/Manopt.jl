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

Note that the robustifier is applied to the squared residuals within the
nonlinear least squares framework, i.e., ``ρ(f_i(p)^2)``.
"""
abstract type AbstractRobustifierFunction <: Function end

@doc """
    NonlinearLeastSquaresObjective{E<:AbstractEvaluationType} <: AbstractManifoldObjective{E}

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
* `jacobian_type::`[`AbstractVectorialType`](@ref)`=`[`CoefficientVectorialType`](@ref)`(jacobian_tangent_basis)`:
  specify the format the Jacobian is given in. By default a matrix of the differential with
  respect to a certain basis of the tangent space.

# See also

[`LevenbergMarquardt`](@ref), [`LevenbergMarquardtState`](@ref)
"""
struct NonlinearLeastSquaresObjective{
        E <: AbstractEvaluationType,
        VF <: Vector{<:AbstractVectorGradientFunction{E}},
        RF <: Vector{<:AbstractRobustifierFunction},
    } <: AbstractManifoldFirstOrderObjective{E, VF}
    objective::VF
    robustifier::RF
    # block components case constructor
    function NonlinearLeastSquaresObjective(
            fs::VF,
            robustifiers::RV = fill(IdentityRobustifier(), length(fs)),
        ) where {E <: AbstractEvaluationType, VF <: Vector{<:AbstractVectorGradientFunction{E}}, RV <: Vector{<:AbstractRobustifierFunction}}
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
        return new{E, Vector{F}, Vector{R}}([f], [robustifier])
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
        jacobian_type::AbstractVectorialType = CoefficientVectorialType(jacobian_tangent_basis),
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
        nlso::NonlinearLeastSquaresObjective,
        p;
        kwargs...,
    )
    v = 0.0
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        vi = sum(abs2, get_value(M, o, p))
        (a, _, _) = get_robustifier_values(r, vi)
        v += a
    end
    v /= 2
    return v
end
#

_doc_get_gradient_nlso = """
    get_gradient(M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...)
    get_gradient!(M::AbstractManifold, X, nlso::NonlinearLeastSquaresObjective, p; kwargs...)

Compute the gradient for the [`NonlinearLeastSquaresObjective`](@ref) `nlso` at the point ``p ∈ M``,
i.e.

```math
$(_tex(:grad)) F(p) = $(_tex(:sum, "i=1", "m")) ρ'$(_tex(:bigl))($(_tex(:norm, "F_i(p)"; index = "2"))^2$(_tex(:bigr)))
$(_tex(:sum, "j=1", "n_i")) f_{i,j}(p) $(_tex(:grad)) f_{i,j}(p)
```

where ``F_i(p) ∈ ℝ^{n_i}`` is the vector of residuals for the `i`-th block component cost function
and ``f_{i,j}(p)`` its `j`-th component function.

# Keyword arguments
* `value_cache=nothing` : if provided, this vector is used to store the residuals ``F(p)``
  internally to avoid recomputations.
"""
#

@doc "$(_doc_get_gradient_nlso)"
function get_gradient(
        M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p
    )
    X = zero_vector(M, p)
    return get_gradient!(M, X, nlso, p; kwargs...)
end

# @doc "$(_doc_get_gradient_nlso)"
function get_gradient!(
        M::AbstractManifold, X, nlso::NonlinearLeastSquaresObjective, p; value_cache = nothing,
    )
    zero_vector!(M, X, p)
    start = 0
    Y = copy(M, p, X)
    for (o, r) in zip(nlso.objective, nlso.robustifier) # for every block
        len = length(o)
        Fi_p = isnothing(value_cache) ? get_value(M, o, p) : view(value_cache, (start + 1):(start + len))
        # get gradients for every component
        for j in 1:len
            get_gradient!(M, Y, o, p, j) # gradient of f_{i,j}
            Y .+= Fi_p[j] .* Y
        end
        # compute robustifier derivative
        (_, b, _) = get_robustifier_values(r, sum(abs2, Fi_p))
        X .+= b .* Y
        start += len
    end
    return X
end


# --- Residuals
_doc_get_residuals_nlso = """
    get_residuals(M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p)
    get_residuals!(M::AbstractManifold, v, nlso::NonlinearLeastSquaresObjective, p)

Compute the vector of residuals ``F(p) ∈ ℝ^n``, ``n = $(_tex(:sum, "1", "m")) n_i``.
In other words this is the concatenation of the residual vectors ``F_i(p)``, ``i=1,…,m``
of the components of the the [`NonlinearLeastSquaresObjective`](@ref) `nlso`
at the current point ``p`` on `M`.

This can be computed in-place of `v`.
"""

@doc "$(_doc_get_residuals_nlso)"
function get_residuals(
        M::AbstractManifold, nlso::NonlinearLeastSquaresObjective, p; kwargs...
    )
    v = zeros(sum(length(o) for o in nlso.objective))
    return get_residuals!(M, v, nlso, p; kwargs...)
end

@doc "$(_doc_get_residuals_nlso)"
function get_residuals!(
        M::AbstractManifold, v, nlso::NonlinearLeastSquaresObjective, p; kwargs...,
    )
    start = 0
    for o in nlso.objective # for every block
        len = length(o)
        view_v = view(v, (start + 1):(start + len))
        get_value!(M, view_v, o, p)
        start += len
    end
    return v
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
ρ''(x) = -$(_tex(:frac, "1", "2 (1 + x)^{3/2}")).
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
ρ_{a,b}(x) = b$(_tex(:log))(1+ $(_tex(:rm, "e"))^{(s-a)/b}) - b$(_tex(:log))(1 + $(_tex(:rm, "e"))^{-a/b})
```
and its first and second derivatives read as
```math
ρ'_{a,b}(x) = $(_tex(:frac, "1", "1 + $(_tex(:rm, "e")){(a - x)/b}"))
```
and
```math
ρ''_{a,b}(x) = $(_tex(:frac, "1", "4b$(_tex(:rm, "cosh"))^2$(_tex(:Bigl))($(_tex(:frac, "(a - x)", "2b"))$(_tex(:Bigr)))")).
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
).
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
$(_fields(:stopping_criterion; name = "stop"))
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
* `minimum_acceptable_model_improvement`: the minimum improvement in the model function that
  is required to accept a new point; if this is not met, the new point is rejected and
  the damping term is increased.
* `model_worsening_warning_threshold`: if the subproblem worsens by more than this
  threshold, a warning is issued. This is useful to detect numerical issues in the linear
  subproblem solver.
* `sub_problem`: the linear subproblem solver to use to solve the linearized
  subproblem in each iteration.
* `sub_state`: the state to use for the linear subproblem solver.

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
* `minimum_acceptable_model_improvement::Real=eps(number_eltype(p))`
* `model_worsening_warning_threshold::Real=-sqrt(eps(number_eltype(p)))`

# See also

[`gradient_descent`](@ref), [`LevenbergMarquardt`](@ref)
"""
mutable struct LevenbergMarquardtState{
        P,
        TStop <: StoppingCriterion,
        TRTM <: AbstractRetractionMethod,
        Tresidual_values,
        TGrad,
        Tparams <: Real,
        Pr,
        St,
    } <: AbstractGradientSolverState
    p::P
    stop::TStop
    retraction_method::TRTM
    residual_values::Tresidual_values
    X::TGrad
    η::Tparams
    damping_term::Tparams
    damping_term_min::Tparams
    β::Tparams
    expect_zero_residual::Bool
    minimum_acceptable_model_improvement::Tparams
    model_worsening_warning_threshold::Tparams
    sub_problem::Pr
    sub_state::St
    function LevenbergMarquardtState(
            M::AbstractManifold,
            initial_residual_values::Tresidual_values;
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
            minimum_acceptable_model_improvement::Real = eps(number_eltype(p)),
            model_worsening_warning_threshold::Real = -sqrt(eps(number_eltype(p))),
            linear_subsolver! = nothing, #remove on next breaking release
            sub_problem::Pr = linear_subsolver!, # TODO change default on next breaking release
            sub_state::St = InplaceEvaluation(),
        ) where {P, Tresidual_values, TGrad, Pr, St}
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
        (β <= 1) && throw(ArgumentError("Value of β must be strictly above 1, received $β"))
        Tparams = promote_type(typeof(η), typeof(damping_term_min), typeof(β))
        SC = typeof(stopping_criterion)
        RM = typeof(retraction_method)
        return new{
            P, SC, RM, Tresidual_values, TGrad, Tparams, Pr, St,
        }(
            p,
            stopping_criterion, retraction_method,
            initial_residual_values,
            X, η,
            damping_term_min, damping_term_min,
            β,
            expect_zero_residual,
            # TODO: Both are for now just for debug
            minimum_acceptable_model_improvement, model_worsening_warning_threshold,
            sub_problem, sub_state,
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

@doc """
    LevenbergMarquardtLinearSurrogateObjective{E<:AbstractEvaluationType, VF<:AbstractManifoldFirstOrderObjective{E}, R} <: AbstractManifoldFirstOrderObjective{E, VF}

Given an [`NonlinearLeastSquaresObjective`](@ref) `objective` and a damping term `damping_term`,
this objective represents the penalized objective for the sub-problem to solve within every step
of the Levenberg-Marquardt algorithm given by

```math
μ_p(X) = $(_tex(:frac, "1", "2"))$(_tex(:norm, _tex(:Cal, "L") * "(X) + y"; index = "2"))^2
  + $(_tex(:frac, "λ", "2"))$(_tex(:norm, "X"; index = "p"))^2,
  $(_tex(:qquad))$(_tex(:text, " for "))X ∈ $(_math(:TangentSpace)), λ ≥ 0,
```

where ``X ∈ $(_math(:TangentSpace))`` and ``λ ≥ 0`` is the damping or penalty term.

In order to build a surrogate also for the robutsified Levenberg-Marquardt, introduce
``α = 1 - $(_tex(:sqrt, "1 + 2 $(_tex(:frac, "ρ''(p)", "ρ'(p)"))$(_tex(:norm, "F(p)"; index = "2"))^2"))``
and set ``y = $(_tex(:frac, _tex(:sqrt, "ρ'(p)"), "1-α"))F(p)`` and ``$(_tex(:Cal, "L"))(X) = CJ_F^*(p)[F(p)]``
with

```math
C = $(_tex(:sqrt, "ρ'(p)"))(I-αP), $(_tex(:qquad)) P = $(_tex(:frac, "F(p)F(p)^" * _tex(:rm, "T"), _tex(:norm, "F(p)"; index = "2") * "^2")),
```

where ``F(p) ∈ ℝ^n`` is the vector of residuals at point ``p ∈ M`` and ``J_F^*(p): ℝ^n → $(_math(:TangentSpace))```
is the adjoint Jacobian.

## Fields

* `objective`:     the [`NonlinearLeastSquaresObjective`](@ref) to penalize
* `penalty::Real`: the damping term ``λ``
* `ε::Real`:       stabilization for ``α ≤ 1-ε`` in the rescaling of the Jacobian, that
* `mode::Symbol`:  which ode to use to stabilize α, see the internal helper [`get_LevenbergMarquardt_scaling`](@ref)

## Constructor

    LevenbergMarquardtLinearSurrogateObjective(objective; penalty::Real = 1e-6, ε::Real = 1e-4, mode::Symbol = :Default )

"""
mutable struct LevenbergMarquardtLinearSurrogateObjective{E <: AbstractEvaluationType, R} <: AbstractLinearSurrogateObjective{E, NonlinearLeastSquaresObjective{E}}
    objective::NonlinearLeastSquaresObjective{E}
    penalty::R
    ε::R
    mode::Symbol
    function LevenbergMarquardtLinearSurrogateObjective(objective::NonlinearLeastSquaresObjective{E}; penalty::R = 1.0e-6, ε::R = 1.0e-4, mode::Symbol = :Default) where {E, R <: Real}
        return new{E, R}(objective, penalty, ε, mode)
    end
end

function show(io::IO, o::LevenbergMarquardtLinearSurrogateObjective{E}) where {E}
    return print(io, "LevenbergMarquardtLinearSurrogateObjective{$E}($(o.objective); penalty=$(o.penalty), ε=$(o.ε), mode=$(o.mode))")
end

"""
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime::Real, ρ_double_prime::Real, FSq::Real, ε::Real, mode::Symbol)

Compute the scaling ``$(_tex(:frac, _tex(:sqrt, "ρ'"), "1 - α"))`` for the residula ``y`` and
the scaling ``$(_tex(:frac, "α", _tex(:norm, "F"; index = "2") * "^2"))`` that are required for the robust
rescaling within [`LevenbergMarquardt`](@ref)s [`vector_field`](@ref) and [`linear_operator`](@ref),
respectively. The value for ``α`` is given by

```math
    α = 1-$(_tex(:sqrt, "1 + 2$(_tex(:frac, "ρ_k''", "ρ_k'"))$(_tex(:norm, "F_k"; index = "2"))"))
```

where
* ``ρ'`` is the first derivative of the [`AbstractRobustifierFunction`](@ref) at ``$(_tex(:norm, "F"; index = "2"))``
* ``ρ''`` is the second derivative of the [`AbstractRobustifierFunction`](@ref) at ``$(_tex(:norm, "k"; index = "2"))``
* `FSq` is the value ``$(_tex(:norm, "F"; index = "2"))``

## Numerical stability

For a unique solution that is a minimizer in a Levenberg-Marquardt step,
we require `α < 1` and [TriggsMcLauchlanHartleyFitzgibbon:2000](@cite) recommends to bound this even by ``1-ε``.

Furthermore if ``ρ´_k + 2ρ''_k $(_tex(:norm, "F"; index = "2")) ≤ 0`` the Hessian is also indefinite.
This can be caught by making sure the argument of the ``√`` is ensured to be nonnegative.

The [Ceres solver](http://ceres-solver.org/nnls_modeling.html#theory) even omits the second term
in the square root already if ``ρ_k'' < 0`` for stability reason, which means setting ``α = 0``.
In the case ``$(_tex(:norm, "F"; index = "2"))`` we also set the operator scaling ``α / FkSq = 0``.

## Additional arguments

* `ε::Real`: the stability for ``α`` to not be too close to one.
* `mode::Symbol` specify the mode of calculation
  - `:Default` keeps negative ``ρ''_k < 0`` but makes sure the square root is well-defined.
  - `:Strict` set ``α = 0`` when ``ρ''_k < 0`` like Ceres does
"""
function get_LevenbergMarquardt_scaling(
        ρ_prime::Real, ρ_double_prime::Real, FkSq::Real,
        ε::Real, mode::Symbol
    )
    # second derivative existent and negative: In strict mode (motivated by ceres) -> return sqrt(ρ_prime), 0
    (ismissing(ρ_double_prime) || (ρ_double_prime < 0 && mode == :Strict)) && return (sqrt(ρ_prime), 0.0)
    (iszero(FkSq) && mode == :Strict) && return (sqrt(ρ_prime), 0.0)
    α = 1 - sqrt(max(1 + 2 * (ρ_double_prime / ρ_prime) * FkSq, 0.0))
    α = min(α, 1 - ε)
    residual_scaling = sqrt(ρ_prime) / (1 - α)
    operator_scaling = ifelse(iszero(FkSq), 0.0, α / FkSq)
    return residual_scaling, operator_scaling
end
function set_parameter!(lmlso::LevenbergMarquardtLinearSurrogateObjective, ::Val{:Penalty}, penalty::Real)
    lmlso.penalty = penalty
    return lmlso
end

get_objective(lmsco::LevenbergMarquardtLinearSurrogateObjective) = lmsco.objective
"""
    get_cost(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )

Compute the surrogate cost. Let ``F`` denote the vector of residuals (of a block),
``ρ, ρ'``, ``ρ''`` the value, first, and second derivative of the [`AbstractRobustifierFunction`](@ref)
of the inner [`NonlinearLeastSquaresObjective`](@ref)

```math
σ_k(X) = $(_tex(:frac, "1", "2"))$(_tex(:norm, "y + $(_tex(:Cal, "L"))(X)"; index = "2"))^2, $(_tex(:qquad)) X ∈ $(_math(:TangentSpace))
```

where
* ``$(_tex(:Cal, "L"))(X) = CJ[X]`` see [`linear_operator`](@ref) with a `penalty` of zero.
* ``y`` the rescaled vector field, see [`vector_field`](@ref)
"""
function get_cost(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    cost = norm(linear_operator(M, lmsco, p, X) + vector_field(M, lmsco, p))^2 / 2
    # add the damping term
    cost += (lmsco.penalty / 2) * norm(M, p, X)^2
    return cost
end
"""
    get_cost(TpM::TangentSpace, slso::SymmetricLInearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, X)

Compute the surrogate cost when solving its normal equation, see also
[`get_cost(::AbstractManifold, ::LevenbergMarquardtLinearSurrogateObjective, p, X)`](@ref),
[`linear_operator`](@ref), and [`vector_field`](@ref) for more details.
"""
function get_cost(
        TpM::TangentSpace, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, X
    ) where {E <: AbstractEvaluationType}
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_cost(M, slso.objective, p, X)
end

@doc """

"""
function get_gradient(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X;
        value_cache = get_residuals(M, lmsco.objective, p)
    )
    Y = zero_vector(M, p)
    return get_gradient!(M, Y, lmsco, p, X; value_cache = value_cache)
end
function get_gradient!(
        M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X;
        value_cache = get_residuals(M, lmsco.objective, p),
    )
    nlso = lmsco.objective
    # For every block
    zero_vector!(M, Y, p)
    Z = copy(M, p, Y)
    start = 0
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len_o = length(o)
        get_gradient!(
            M, Z, o, r, p, X;
            value_cache = value_cache[(start + 1):(start + len_o)], ε = lmsco.ε, mode = lmsco.mode,
        )
        Y .+= Z
        start += len_o
    end
    # add penalty term
    Y .+= lmsco.penalty .* X
    return Y
end
function get_gradient!(
        M::AbstractManifold, Y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X;
        value_cache = get_value(M, o, p), ε::Real, mode::Symbol,
    )
    a = value_cache # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_p_norm2, ε, mode)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
    b = zero(a)
    get_jacobian!(M, b, o, p, X)
    # Compute C^TCa = C^2 a (inplace of a)
    b .= ρ_prime .* (I - operator_scaling * (a * a'))^2 * b
    # add C^T y = C^T (sqrt(ρ(p)) / (1 - α) F(p)) (which overall has a ρ_prime upfront)
    b .+= residual_scaling .* (I - operator_scaling * (a * a')) * a
    # apply the adjoint
    get_adjoint_jacobian!(M, Y, o, p, b)
    return Y
end

#
#
# We can do a closed form solution of the Surrogate using \ as soon as we have a basis
# We can model that as a state of a solver for ease of use
"""
    CoordinatesNormalSystemState <: AbstractManoptSolverState

A solver state indicating that we solve the [`LevenbergMarquardtLinearSurrogateObjective`](@ref)
using a linear system in coordinates of the tangent space at the current iterate

# Fields
* `A` an ``n×n`` matrix to store the normal operator in coordinates, where `n` is the number of coordinates
* `b` a ``n`` vector storing the right hand side of the system of normal equations
* `basis::`[`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`)
* `linsolve` a functor `(A,b) -> c` to solve the linear system or `(c, A, b) -> c` depending on the evaluation type specified in `solve!`
"""
mutable struct CoordinatesNormalSystemState{E <: AbstractEvaluationType, F, TA <: AbstractMatrix, TB <: AbstractVector, TBA <: AbstractBasis} <: AbstractManoptSolverState
    A::TA
    b::TB
    basis::TBA
    c::TB
    linsolve!!::F
end
function CoordinatesNormalSystemState(
        M::AbstractManifold, p = rand(M);
        evaluation::E = AllocatingEvaluation(), linsolve::F = \, basis::B = DefaultOrthonormalBasis()
    ) where {E <: AbstractEvaluationType, F, B <: AbstractBasis}
    c = get_coordinates(M, p, zero_vector(M, p))
    n = length(c)
    A = zeros(eltype(c), n, n)
    b = zeros(eltype(c), n)
    return CoordinatesNormalSystemState{E, F, typeof(A), typeof(b), B}(A, b, basis, c, linsolve)
end

# The objective here should be a LevenbergMarquardtLinearSurrogateObjective, but might be decorated as well, so for now lets not type it (yet?)
function solve!(dmp::DefaultManoptProblem{<:TangentSpace}, cnss::CoordinatesNormalSystemState{AllocatingEvaluation})
    # Update A and b
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    o = get_objective(dmp)
    linear_operator!(M, cnss.A, o, p, cnss.basis)
    vector_field!(M, cnss.b, o, p, cnss.basis)
    cnss.c = cnss.linsolve!!(cnss.A, -cnss.b)
    X = get_vector(M, p, cnss.c, cnss.basis)
    # TODO: Remove ?
    if get_cost(dmp, 0 * X) < get_cost(dmp, -X) - sqrt(eps())
        @show get_cost(dmp, 0 * X)
        @show get_cost(dmp, -X)
        error("model cost is much worse than zero step, something is wrong")
    end
    return cnss
end
function solve!(dmp::DefaultManoptProblem{<:TangentSpace}, cnss::CoordinatesNormalSystemState{InplaceEvaluation})
    # Update A and b
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    o = get_objective(dmp) # implicit: SymmetricSystem ...
    linear_operator!(M, cnss.A, o, p, cnss.basis)
    vector_field!(M, cnss.b, o, p, cnss.basis)
    cnss.linsolve!!(cnss.c, cnss.A, -cnss.b)
    return cnss
end
# Maybe a bit too precise, but in this case we get a coefficient vector and we want a tangent vector
function get_solver_result(
        dmp::DefaultManoptProblem{<:TangentSpace, <:SymmetricLinearSystem{<:AbstractEvaluationType, <:LevenbergMarquardtLinearSurrogateObjective}},
        cnss::CoordinatesNormalSystemState
    )
    TpM = get_manifold(dmp)
    M = base_manifold(TpM)
    p = base_point(TpM)
    return get_vector(M, p, cnss.c, cnss.basis)
end
#
#
#
"""
    linear_normal_operator(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    linear_normal_operator(M::AbstractManifold, lmsco::o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    linear_normal_operator!(M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    linear_normal_operator!(M::AbstractManifold, Y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X; ε = lmsco.ε, mode = lmsco.mod)

    linear_normal_operator(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p[, c], B::AbstractBasis; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    linear_normal_operator(M::AbstractManifold, lmsco::o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p[, c], B::AbstractBasis; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    linear_normal_operator!(M::AbstractManifold, [A | b], lmsco::LevenbergMarquardtLinearSurrogateObjective, p[, c], B::AbstractBasis; ε = lmsco.ε, mode = lmsco.mode, penalty = lmsco.penalty)
    linear_normal_operator!(M::AbstractManifold, [A | b], o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p[, c], B::AbstractBasis; ε = lmsco.ε, mode = lmsco.mode)

Compute the linear operator ``$(_tex(:Cal, "A"))`` corresponding to the optimality conditions of the
modified Levenberg-Marquardt surrogate objective, i.e. the normal conditions

```math
$(_tex(:Cal, "A"))(X) = $(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))(X) + λX
= J_F^*(p)$(_tex(:bigl))[ C^T C J_F(p)[X] $(_tex(:bigr))] + λX,
```

where ``λ = ```penalty` is a damping parameter.
While this can be set as a keyword, e.g. when computing the cost, its default is the internally
stored `penalty` from the surrogate objective ``μ_k``.

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

If you provide an [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) `B` ``Z_1,…,Z_d`` instead, the operator is returned in Matrix form `A`,
with respect to these coordinates, i.e. if you provide furthermore `X`
in coordinates `c` as ``X = $(_tex(:displaystyle))$(_tex(:sum, "i=0", "d")) c_iZ_i``,
then the result ``Y = $(_tex(:displaystyle))$(_tex(:sum, "i=0", "d")) b_iZ_i`` with ``b = Ac`` is computed.

See also [`normal_vector_field`](@ref) for evaluating the corresponding vector field.
"""
function linear_normal_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X;
        penalty = lmsco.penalty,
    )
    Y = zero_vector(M, p)
    return linear_normal_operator!(M, Y, lmsco, p, X; penalty = penalty)
end
function linear_normal_operator!(
        M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X;
        penalty = lmsco.penalty,
    )
    nlso = get_objective(lmsco)
    # For every block
    zero_vector!(M, Y, p)
    Z = copy(M, p, Y)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        linear_normal_operator!(M, Z, o, r, p, X; ε = lmsco.ε, mode = lmsco.mode)
        Y .+= Z
    end
    # Finally add the damping term
    (penalty != 0) && (Y .+= penalty * X)
    return Y
end
# for a single block – the actual formula - but never with penalty
function linear_normal_operator!(
        M::AbstractManifold, Y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X;
        ε::Real, mode::Symbol
    )
    a = get_value(M, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_p_norm2, ε, mode)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
    b = zero(a)
    get_jacobian!(M, b, o, p, X)
    # Compute C^TCb = C^2 b (inplace of a)
    a .= ρ_prime .* (I - operator_scaling * (a * a'))^2 * b
    # Now apply the adjoint
    get_adjoint_jacobian!(M, Y, o, p, a)
    # penalty is added once after summing up all blocks, so we do not add it here
    return Y
end

#
# Basis case: (a) including coordinates
function linear_normal_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, c, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    d = zero(c)
    return linear_normal_operator!(M, d, lmsco, p, c, B; penalty = penalty)
end
function linear_normal_operator!(
        M::AbstractManifold, d, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, c, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(d, 0)
    e = copy(d)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        linear_normal_operator!(M, e, o, r, p, c, B; ε = lmsco.ε, mode = lmsco.mode)
        d .+= e
    end
    # Finally add the damping term
    (penalty != 0) && (d .+= penalty * c)
    return d
end
function linear_normal_operator!(M::AbstractManifold, d, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, c, B::AbstractBasis; kwargs...)
    A = linear_normal_operator(M, o, r, p, B; kwargs...)
    d .= A * c
    return d
end

#
# Basis case: (b) no coordinates -> compute a matrix representation
function linear_normal_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    d = number_of_coordinates(M, B)
    A = zeros(eltype(p), d, d)
    return linear_normal_operator!(M, A, lmsco, p, B; penalty = penalty)
end
function linear_normal_operator!(
        M::AbstractManifold, A, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis;
        penalty = lmsco.penalty
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(A, 0)
    C = copy(A)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        linear_normal_operator!(M, C, o, r, p, B; ε = lmsco.ε, mode = lmsco.mode)
        A .+= C
    end
    # Finally add the damping term
    (penalty != 0) && (A .= A + penalty * I)
    return A
end
function linear_normal_operator!(
        M::AbstractManifold, A, o::AbstractVectorGradientFunction,
        r::AbstractRobustifierFunction, p, basis::AbstractBasis;
        ε::Real, mode::Symbol
    )
    a = get_value(M, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    _, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_p_norm2, ε, mode)
    # to Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
    # (a) J_F is n-by-d so we have to allocate – where could we maybe store something like that and pass it down?
    JF = get_jacobian(M, o, p; basis = basis)
    # compute A' C^TC A (C^TC = C^2 here) inplace of A
    A .= JF' * (ρ_prime .* (I - operator_scaling * (a * a'))^2) * JF
    # damping term is added once after summing up all blocks, so we do not add it here
    return A
end
"""
    linear_operator(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X)
    linear_operator(M::AbstractManifold, lmsco::o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X)
    linear_operator!(M::AbstractManifold, y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X)
    linear_operator!(M::AbstractManifold, y, lmsco::o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X)

Compute the linear operator ``$(_tex(:Cal, "L"))`` corresponding to the optimality conditions of the
Levenberg-Marquardt surrogate objective, i.e. the normal conditions

```math
$(_tex(:Cal, "L"))(X) = C J_F(p)[X] $(_tex(:bigr))],
```

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.
This can be computed in-place of `y`.

See also [`vector_field`](@ref) for evaluating the corresponding vector field
"""
function linear_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    nlso = get_objective(lmsco)
    n = sum(length(o) for o in nlso.objective)
    y = zeros(eltype(p), n)
    return linear_operator!(M, y, lmsco, p, X)
end
function linear_operator!(
        M::AbstractManifold, y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    nlso = get_objective(lmsco)
    # Init to zero
    fill!(y, 0)
    start = 0
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len = length(o)
        linear_operator!(M, view(y, (start + 1):(start + len)), o, r, p, X; ε = lmsco.ε, mode = lmsco.mode)
        start += len
    end
    return y
end
# for a single block – the actual formula
function linear_operator!(
        M::AbstractManifold, y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X,
        value_cache = get_value(M, o, p); ε::Real, mode::Symbol,
    )
    F_p = get_value(M, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, F_p)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_p_norm2, ε, mode)
    get_jacobian!(M, y, o, p, X)
    # Compute C y
    y .= residual_scaling .* (I - operator_scaling * (F_p * F_p'))^2 * y
    return y
end

_doc_normal_vector_field = """
    normal_vector_field(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    normal_vector_field!(M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    normal_vector_field!(M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p)

    normal_vector_field(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis)
    normal_vector_field!(M::AbstractManifold, c, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis)
    normal_vector_field!(M::AbstractManifold, c, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, B::AbstractBasis)

Compute the normal linear operator tangent vector ``X`` corresponding to the optimality conditions of the
Levenberg-Marquardt surrogate objective, i.e.,

```math
X = J_F^*(p)[ C^T y], $(_tex(:quad)) y = $(_tex(:frac, _tex(:sqrt, "ρ'(p)"), "1-α"))F(p).
```

If you provide an [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) `B` ``=$(_tex(:set, "Z_1,…,Z_d"))`` additionally,
the result will be given in coordinates `c`, i.e. such that ``X = $(_tex(:sum, "i=1", "d")) c_iZ_i``.

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also [`linear_normal_operator`](@ref) for evaluating the corresponding linear operator of the (normal) linear system,
and [`get_LevenbergMarquardt_scaling`](@ref) for details on the scaling and computation of ``C``.
"""

@doc "$(_doc_normal_vector_field)"
function normal_vector_field(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    X = zero_vector(M, p)
    return normal_vector_field!(M, X, lmsco, p)
end

@doc "$(_doc_normal_vector_field)"
function normal_vector_field!(
        M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    # For every block
    zero_vector!(M, X, p)
    Z = copy(M, p, X)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        normal_vector_field!(M, Z, o, r, p; ε = lmsco.ε, mode = lmsco.mode)
        X .+= Z
    end
    return X
end
# for a single block – the actual formula
@doc "$(_doc_normal_vector_field)"
function normal_vector_field!(
        M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p;
        ε::Real, mode::Symbol,
    )
    y = get_value(M, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_p_norm2, ε, mode)
    # Compute y = (sqrt(ρ'(p)) / (1-α)) F(p) and
    # Now compute J_F^*(p)[C^T y] (inplace of y)
    y .= residual_scaling .* sqrt(ρ_prime) * (I - operator_scaling * (y * y')) * y
    # Now apply the adjoint and negate
    get_adjoint_jacobian!(M, X, o, p, y)
    return X
end

#
# (b) in a basis
@doc "$(_doc_normal_vector_field)"
function normal_vector_field(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis,
    )
    c = get_coordinates(M, p, zero_vector(M, p), B)
    return normal_vector_field!(M, c, lmsco, p, B)
end

@doc "$(_doc_normal_vector_field)"
function normal_vector_field!(
        M::AbstractManifold, c, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, B::AbstractBasis,
    )
    nlso = get_objective(lmsco)
    # For every block
    fill!(c, 0)
    d = copy(c)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        normal_vector_field!(M, d, o, r, p, B; ε = lmsco.ε, mode = lmsco.mode)
        c .+= d
    end
    return c
end
# for a single block – the actual formula
@doc "$(_doc_normal_vector_field)"
function normal_vector_field!(
        M::AbstractManifold, c, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, B::AbstractBasis;
        ε::Real, mode::Symbol,
    )
    y = get_value(M, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    residual_scaling, operator_scaling = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_p_norm2, ε, mode)
    # Compute y = (sqrt(ρ'(p)) / (1-α)) F(p) and
    # Now compute J_F^*(p)[C^T y] (inplace of y)
    y .= residual_scaling .* sqrt(ρ_prime) * (I - operator_scaling * (y * y')) * y
    # Now apply the adjoint
    get_adjoint_jacobian!(M, c, o, p, y, B)
    return c
end

"""
    vector_field(M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    vector_field!(M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p)
    vector_field!(M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p)

Compute the vector field ``y`` corresponding to the Levenberg-Marquardt surrogate objective, i.e.,

```math
y = $(_tex(:frac, _tex(:sqrt, "ρ'(p)"), "1-α"))F(p).
```

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also
* [`get_LevenbergMarquardt_scaling`](@ref) for details on the scaling
* [`linear_operator`](@ref) for evaluating the corresponding linear operator of the linear system
"""
function vector_field(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    n = sum(length(o) for o in nlso.objective)
    y = zeros(number_eltype(p), n)
    return vector_field!(M, y, lmsco, p)
end
function vector_field!(
        M::AbstractManifold, y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    # Init to zero
    fill!(y, 0)
    start = 0
    # For every block
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        vector_field!(M, view(y, (start + 1):(start + length(o))), o, r, p; ε = lmsco.ε, mode = lmsco.mode)
        start += length(o)
    end
    return y
end
# for a single block – the actual formula
function vector_field!(
        M::AbstractManifold, y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p;
        ε::Real, mode::Symbol,
    )
    get_value!(M, y, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    residual_scaling, _ = get_LevenbergMarquardt_scaling(ρ_prime, ρ_double_prime, F_p_norm2, ε, mode)
    # Compute y = (sqrt(ρ(p)) / (1-α)) F(p)
    y .*= residual_scaling
    return y
end

#
# The Symmetric Linear System (e.g. in CGRes) for the LM Surrogate is its normal equations and vector.
# (a) a vector X or a basis B
function linear_operator(
        M::AbstractManifold, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, XB
    ) where {E <: AbstractEvaluationType}
    return linear_normal_operator(M, slso.objective, p, XB)
end
function linear_operator!(
        M::AbstractManifold, Y, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, XB
    ) where {E <: AbstractEvaluationType}
    return linear_normal_operator!(M, Y, slso.objective, p, XB)
end
# (b) coefficients in a basis
function linear_operator(
        M::AbstractManifold, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, c, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    return linear_normal_operator(M, slso.objective, p, c, B)
end
function linear_operator!(
        M::AbstractManifold, Y, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, c, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    return linear_normal_operator!(M, Y, slso.objective, p, c, B)
end
# RHS as a tangent vector
function vector_field(
        M::AbstractManifold, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p
    ) where {E <: AbstractEvaluationType}
    return -normal_vector_field(M, slso.objective, p)
end
function vector_field!(
        M::AbstractManifold, Y, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p
    ) where {E <: AbstractEvaluationType}
    normal_vector_field!(M, Y, slso.objective, p)
    Y .*= -1
    return Y
end
# RHS in coordinates
function vector_field(
        M::AbstractManifold, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    return -normal_vector_field(M, slso.objective, p, B)
end
function vector_field!(
        M::AbstractManifold, c, slso::SymmetricLinearSystem{E, <:LevenbergMarquardtLinearSurrogateObjective}, p, B::AbstractBasis
    ) where {E <: AbstractEvaluationType}
    normal_vector_field!(M, c, slso.objective, p, B)
    c .*= -1
    return c
end
