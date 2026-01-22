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
* `jacobian_type::`[`AbstractVectorialType`](@ref)`=`[`CoordinateVectorialType`](@ref)`(jacobian_tangent_basis)`:
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

@doc "$(_doc_get_gradient_nlso)"
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
    @info v
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
            linear_subsolver! = nothing, #remove on next breaking release
            sub_problem::Pr = linear_subsolver!, # todo: change default?
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
        if β <= 1
            throw(ArgumentError("Value of β must be strictly above 1, received $β"))
        end
        Tparams = promote_type(typeof(η), typeof(damping_term_min), typeof(β))
        SC = typeof(stopping_criterion)
        RM = typeof(retraction_method)
        return new{
            P, SC, RM, Tresidual_values, TGrad, Tparams, Pr, St,
        }(
            p,
            stopping_criterion,
            retraction_method,
            initial_residual_values,
            X,
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
mutable struct LevenbergMarquardtLinearSurrogateObjective{E <: AbstractEvaluationType, R} <: AbstractLinearSurrogateObjective{E, NonlinearLeastSquaresObjective{E}}
    objective::NonlinearLeastSquaresObjective{E}
    penalty::R
end

function set_parameter!(lmlso::LevenbergMarquardtLinearSurrogateObjective, ::Val{:Penalty}, penalty::Real)
    lmlso.penalty = penalty
    return lmlso
end

get_objective(lmsco::LevenbergMarquardtLinearSurrogateObjective) = lmsco.objective

function get_cost(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X;
        value_cache = get_residuals(M, lmsco.objective, p),
        penalty = lmsco.penalty,
    )
    nlso = lmsco.objective
    cost = 0.0
    # For every block
    start = 0
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        cost += get_cost(M, o, r, p; value_cache = value_cache[(start + 1):(start + length(o))])
    end
    # Finally add the damping term
    cost += (penalty / 2) * norm(M, p, X)^2
    return cost
end
function get_cost(
        M::AbstractManifold, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, C;
        value_cache = get_value(M, o, p)
    )
    F_p_norm2 = sum(abs2, value_cache)
    (ρ_value, _, _) = get_robustifier_values(r, F_p_norm2)
    return 0.5 * ρ_value
end

function get_gradient(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X;
        value_cache = get_residuals(M, lmsco.objective, p),
        penalty = lmsco.penalty,
    )
    Y = zero_vector(M, p)
    return get_gradient!(
        M, Y, lmsco, p, X;
        value_cache = value_cache,
        penalty = penalty,
    )
end
function get_gradient!(
        M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X;
        value_cache = get_residuals(M, lmsco.objective, p),
        penalty = lmsco.penalty,
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
            value_cache = value_cache[(start + 1):(start + len_o)],
        )
        Y .+= Z
        start += len_o
    end
    # Finally add the damping term
    (penalty != 0) && (Y .+= penalty * X)
    return Y
end
function get_gradient!(
        M::AbstractManifold, Y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X;
        value_cache = get_value(M, o, p),
    )
    a = value_cache # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    α = 1 - sqrt(1 + 2 * (ismissing(ρ_double_prime) ? 0.0 : ρ_double_prime / ρ_prime) * F_p_norm2)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
    b = zero(a)
    get_jacobian!(M, b, o, p, X)
    # Compute C^TCa = C^2 a (inplace of a)
    b .= ρ_prime .* (I - α * (a * a') ./ F_p_norm2)^2 * b
    # add C^T y = C^T (sqrt(ρ(p)) / (1 - α) F(p)) (which overall has a ρ_prime upfront)
    b .+= (ρ_prime / (1 - α)) .* (I - α * (a * a') ./ F_p_norm2) * a
    # apply the adjoint
    get_adjoint_jacobian!(M, Y, o, p, b)
    return Y
end

"""
    linear_normal_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X; penalty = lmsco.penalty,
    )
    linear_normal_operator(
        M::AbstractManifold, lmsco::o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X; penalty = lmsco.penalty,
    )
    linear_normal_operator!(
        M::AbstractManifold, Y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X; penalty = lmsco.penalty,
    )
    linear_normal_operator!(
        M::AbstractManifold, Y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X; penalty = lmsco.penalty,
    )

Compute the linear operator ``$(_tex(:Cal, "A"))` corresponding to the optimality conditions of the
modified Levenberg-Marquardt surrogate objective, i.e. the normal conditions

```math
$(_tex(:Cal, "A"))(X) = $(_tex(:Cal, "L"))^* $(_tex(:Cal, "L"))(X) + λX
= J_F^*(p)$(_tex(:bigl))[ C^T C J_F(p)[X] $(_tex(:bigr))] + λX,
```

where ``λ = `penalty` is a damping parameter.

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also [`normal_vector_field`](@ref) for evaluating the corresponding vector field
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
        linear_normal_operator!(M, Z, o, r, p, X; penalty = 0)
        Y .+= Z
    end
    # Finally add the damping term
    (penalty != 0) && (Y .+= penalty * X)
    return Y
end
# for a single block – the actual formula
function linear_normal_operator!(
        M::AbstractManifold, Y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X;
        penalty = 0,
    )
    a = get_value(M, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, a)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    α = 1 - sqrt(1 + 2 * (ismissing(ρ_double_prime) ? 0.0 : ρ_double_prime / ρ_prime) * F_p_norm2)
    # Compute J_F^*(p)[C^T C J_F(p)[X]], but since C is symmetric, we can do that squared idrectly
    b = zero(a)
    get_jacobian!(M, b, o, p, X)
    # Compute C^TCb = C^2 b (inplace of a)
    a .= ρ_prime .* (I - α * (a * a') ./ F_p_norm2)^2 * b
    # Now apply the adjoint
    get_adjoint_jacobian!(M, Y, o, p, a)
    # Finally add the damping term
    (penalty != 0) && (Y .+= penalty * X)
    return Y
end

"""
    linear_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X
    )
    linear_operator(
        M::AbstractManifold, lmsco::o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X,
    )

Compute the linear operator ``$(_tex(:Cal, "L"))` corresponding to the optimality conditions of the
Levenberg-Marquardt surrogate objective, i.e. the normal conditions

```math
$(_tex(:Cal, "L"))(X) = C J_F^*(p)[X] $(_tex(:bigr))],
```

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also [`evaluate_tangent_vector`](@ref) for evaluating the corresponding vector field
"""
function linear_operator(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X,
    )
    nlso = get_objective(lmsco)
    n = sum(length(o) for o in nlso.objective)
    y = zeros(eltype(p), n)
    return linear_operator!(M, y, lmsco, p, X)
end
function linear_operator!(
        M::AbstractManifold, y, lmsco::LevenbergMarquardtLinearSurrogateObjective, p, X,
    )
    nlso = get_objective(lmsco)
    # Init to zero
    fill!(y, 0)
    start = 0
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        len = length(o)
        linear_operator!(M, y[(start + 1):(start + len)], o, r, p, X)
        start += len
    end
    return y
end
# for a single block – the actual formula
function linear_operator!(
        M::AbstractManifold, y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p, X,
    )
    F_p = get_value(M, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, F_p)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    α = 1 - sqrt(1 + 2 * (ismissing(ρ_double_prime) ? 0.0 : ρ_double_prime / ρ_prime) * F_p_norm2)
    get_jacobian!(M, y, o, p, X)
    # Compute C y
    y .= sqrt(ρ_prime) .* (I - α * (F_p * F_p') ./ F_p_norm2) * y
    return y
end

"""
    normal_vector_field(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    normal_vector_field!(
        M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    normal_vector_field!(
        M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p
    )

Compute the normal linear operator tangent vector ``X`` corresponding to the optimality conditions of the
Levenberg-Marquardt surrogate objective, i.e.,
```math
X = - J_F^*(p)[ C^T y], $(_tex(:quad)) y = $(_tex(:frac, _tex(:sqrt, "ρ(p)"), "1-α"))F(p).
```

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also [`evaluate_linear_operator`](@ref) for evaluating the corresponding linear operator of the linear system
"""
function normal_vector_field(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    X = zero_vector(M, p)
    return normal_vector_field!(M, X, lmsco, p)
end
function normal_vector_field!(
        M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    # For every block
    zero_vector!(M, X, p)
    Z = copy(M, p, X)
    for (o, r) in zip(nlso.objective, nlso.robustifier)
        normal_vector_field!(M, Z, o, r, p)
        X .+= Z
    end
    return X
end
# for a single block – the actual formula
function normal_vector_field!(
        M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p
    )
    y = get_value(M, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    α = 1 - sqrt(1 + 2 * (ismissing(ρ_double_prime) ? 0.0 : ρ_double_prime / ρ_prime) * F_p_norm2)
    # Compute y = (sqrt(ρ'(p)) / (1-α)) F(p) and
    # Now compute J_F^*(p)[C^T y] (inplace of y)
    y .= (ρ_prime / (1 - α)) * (I - α * (y * y') ./ F_p_norm2) * y
    # Now apply the adjoint and negate
    get_adjoint_jacobian!(M, X, o, p, y)
    X .*= -1
    return X
end

"""
    vector_field(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    vector_field!(
        M::AbstractManifold, X, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    vector_field!(
        M::AbstractManifold, X, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p
    )

Compute the vector field ``y`` corresponding to the Levenberg-Marquardt surrogate objective, i.e.,

```math
y = $(_tex(:frac, _tex(:sqrt, "ρ(p)"), "1-α"))F(p).
```

Note that this is done per every block (vectorial function with its robustifier) of the underlying
[`NonlinearLeastSquaresObjective`](@ref) and summed up.

See also [`evaluate_linear_operator`](@ref) for evaluating the corresponding linear operator of the linear system
"""
function vector_field(
        M::AbstractManifold, lmsco::LevenbergMarquardtLinearSurrogateObjective, p
    )
    nlso = get_objective(lmsco)
    n = sum(length(o) for o in nlso.objective)
    y = zeros(eltype(p), n)
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
        vector_field!(M, y[(start + 1):(start + length(o))], o, r, p)
        start += length(o)
    end
    return y
end
# for a single block – the actual formula
function vector_field!(
        M::AbstractManifold, y, o::AbstractVectorGradientFunction, r::AbstractRobustifierFunction, p
    )
    get_value!(M, y, o, p) # evaluate residuals F(p)
    F_p_norm2 = sum(abs2, y)
    (_, ρ_prime, ρ_double_prime) = get_robustifier_values(r, F_p_norm2)
    α = 1 - sqrt(1 + 2 * (ismissing(ρ_double_prime) ? 0.0 : ρ_double_prime / ρ_prime) * F_p_norm2)
    # Compute y = (sqrt(ρ(p)) / (1-α)) F(p)
    y .= (sqrt(ρ_value) / (1 - α)) * y
    return y
end
