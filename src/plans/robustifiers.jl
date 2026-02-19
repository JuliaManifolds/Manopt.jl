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
    ComponentwiseRobustifierFunction{F<:AbstractRobustifierFunction} <: AbstractRobustifierFunction

A robustifier to indicate that for a certain [`AbstractVectorGradientFunction`](@ref) a
robustifier should be applied component wise.

## Fields
* `robustifier::R` the robustifer to be applied componentwise

## Constructor

    ComponentwiseRobustifierFunction(robustifier::AbstractRobustifierFunction)

Create a new componentwise robustifier function, where this wrapper avoids to ”double wrap”,
i.e. calling the constructor with a componentwise robustifier creates a new componentwise robustifier
with the only taking the internal robustifier.
"""
struct ComponentwiseRobustifierFunction{R <: AbstractRobustifierFunction} <: AbstractRobustifierFunction
    robustifier::R
    ComponentwiseRobustifierFunction(robustifier::R) where {R <: AbstractRobustifierFunction} = new{R}(robustifier)
    ComponentwiseRobustifierFunction(cr::ComponentwiseRobustifierFunction{R}) where {R <: AbstractRobustifierFunction} = new{R}(cr.robustifier)
end

# Default case: for a number, this is just a passthrough
function get_robustifier_values(crf::ComponentwiseRobustifierFunction, x::Real)
    return get_robustifier_values(crf.robustifier, x)
end
function get_robustifier_values(crf::ComponentwiseRobustifierFunction, x::AbstractArray)
    # TODO turn form a vector of tuples into a tuple of vectors
    return collect(zip([get_robustifier_values(crf, xi) for xi in x]...))
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
    if x <= 1
        return (x, 1.0, 0.0)
    else
        sx = sqrt(x)
        return (2 * sx - 1, 1 / sx, 1 / (2 * sx * x))
    end
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
