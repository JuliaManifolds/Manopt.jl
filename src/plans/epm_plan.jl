@doc raw"""
    ExactPenaltyMethodOptions{P,T} <: Options

Describes the exact penalty method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` – a set point on a manifold as starting point
* `sub_problem` – problem for the subsolver
* `sub_options` – options of the subproblem
* `ϵ` – (`1e–3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `u` – (`1e–1`) the smoothing parameter and threshold for violation of the constraints
* `u_min` – (`1e-6`) the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ` – (`1.0`) the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(300), `[`StopWhenAll`](@ref)`(`[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min), `[`StopWhenChangeLess`](@ref)`(min_stepsize)))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.


# Constructor

    ExactPenaltyMethodOptions(M::AbstractManifold, P::ConstrainedProblem, x; kwargs...)

construct an exact penalty options with the fields and defaults as above, where the
manifold `M` and the [`ConstrainedProblem`](@ref) `P` are used for defaults in the keyword
arguments.

# See also
[`exact_penalty_method`](@ref)
"""
mutable struct ExactPenaltyMethodOptions{
    P,Pr<:Problem,Op<:Options,TStopping<:StoppingCriterion
} <: Options
    x::P
    sub_problem::Pr
    sub_options::Op
    ϵ::Real
    ϵ_min::Real
    u::Real
    u_min::Real
    ρ::Real
    θ_ρ::Real
    θ_u::Real
    θ_ϵ::Real
    stop::TStopping
    function ExactPenaltyMethodOptions(
        ::AbstractManifold,
        x0::P,
        sub_problem::Pr,
        sub_options::Op;
        ϵ::Real=1e-3,
        ϵ_min::Real=1e-6,
        ϵ_exponent=1 / 100,
        θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent),
        u::Real=1e-1,
        u_min::Real=1e-6,
        u_exponent=1 / 100,
        θ_u=(u_min / u)^(u_exponent),
        ρ::Real=1.0,
        θ_ρ::Real=0.3,
        stopping_criterion::StoppingCriterion=StopWhenAny(
            StopAfterIteration(300),
            StopWhenAll(StopWhenSmallerOrEqual(:ϵ, ϵ_min), StopWhenChangeLess(1e-10)),
        ),
    ) where {P,Pr<:Problem,Op<:Options}
        o = new{P,Pr,Op,typeof(stopping_criterion)}()
        o.x = x0
        o.sub_problem = sub_problem
        o.sub_options = sub_options
        o.ϵ = ϵ
        o.ϵ_min = ϵ_min
        o.u = u
        o.u_min = u_min
        o.ρ = ρ
        o.θ_ρ = θ_ρ
        o.θ_u = θ_u
        o.θ_ϵ = θ_ϵ
        o.stop = stopping_criterion
        return o
    end
end
get_iterate(O::ExactPenaltyMethodOptions) = O.x
function set_iterate!(O::ExactPenaltyMethodOptions, p)
    O.x = p
    return O
end
"""
    abstract type SmoothingTechnique

Specify a smoothing technique, e.g. for the [`ExactPenaltyCost`](@ref) and [`ExactPenaltyGrad`](@ref).
"""
abstract type SmoothingTechnique end

@doc raw"""
    LogarithmicSumOfExponentials <: SmoothingTechnique

Specify a smoothing based on ``\max\{a,b\} ≈ u \log(\mathrm{e}^{\frac{a}{u}}+\mathrm{e}^{\frac{b}{u}})``
for some ``u``.
"""
struct LogarithmicSumOfExponentials <: SmoothingTechnique end

@doc raw"""
    LinearQuadraticHuber <: SmoothingTechnique

Specify a smoothing based on ``\max\{0,x\} ≈ \mathcal P(x,u)`` for some ``u``, where

```math
\mathcal P(x, u) = \begin{cases}
  0 & \text{ if } x \leq 0,\\
  \frac{x^2}{2u} & \text{ if } 0 \leq x \leq u,\\
  x-\frac{u}{2} & \text{ if } x \geq u.
\end{cases}
```
"""
struct LinearQuadraticHuber <: SmoothingTechnique end

@doc raw"""
    ExactPenaltyCost{S, Pr, R}

Represent the cost of the exact penalty method based on a [`ConstrainedProblem`](@ref) `P`
and a parameter ``ρ`` given by

```math
f(p) + ρ\Bigl(
    \sum_{i=0}^m \max\{0,g_i(p)\} + \sum_{j=0}^n \lvert h_j(p)\rvert
\Bigr),
```
where we use an additional parameter ``u`` and a smoothing technique, e.g.
[`LogarithmicSumOfExponentials`](@ref) or [`LinearQuadraticHuber`](@ref)
to obtain a smooth cost function. This struct is also a functor `(M,p) -> v` of the cost ``v``.

## Fields

* `P`, `ρ`, `u` as mentioned above.

## Constructor

    ExactPenaltyCost(P::ConstrainedProblem, ρ, u; smoothing=LinearQuadraticHuber())
"""
mutable struct ExactPenaltyCost{S,Pr,R}
    P::Pr
    ρ::R
    u::R
end
function ExactPenaltyCost(
    P::Pr, ρ::R, u::R; smoothing=LinearQuadraticHuber()
) where {Pr<:ConstrainedProblem,R}
    return ExactPenaltyCost{typeof(smoothing),Pr,R}(P, ρ, u)
end
function (L::ExactPenaltyCost{<:LogarithmicSumOfExponentials})(::AbstractManifold, p)
    gp = get_inequality_constraints(L.P, p)
    hp = get_equality_constraints(L.P, p)
    m = length(gp)
    n = length(hp)
    cost_ineq = (m > 0) ? sum(L.u .* log.(1 .+ exp.(gp ./ L.u))) : 0.0
    cost_eq = (n > 0) ? sum(L.u .* log.(exp.(hp ./ L.u) .+ exp.(-hp ./ L.u))) : 0.0
    return get_cost(L.P, p) + (L.ρ) * (cost_ineq + cost_eq)
end
function (L::ExactPenaltyCost{<:LinearQuadraticHuber})(::AbstractManifold, p)
    gp = get_inequality_constraints(L.P, p)
    hp = get_equality_constraints(L.P, p)
    m = length(gp)
    n = length(hp)
    cost_eq_greater_u = (m > 0) ? sum((gp .- L.u / 2) .* (gp .> L.u)) : 0.0
    cost_eq_pos_smaller_u = (m > 0) ? sum((gp .^ 2 ./ (2 * L.u)) .* (0 .< gp .<= L.u)) : 0.0
    cost_ineq = cost_eq_greater_u + cost_eq_pos_smaller_u
    cost_eq = (n > 0) ? sum(sqrt.(hp .^ 2 .+ L.u^2)) : 0.0
    return get_cost(L.P, p) + (L.ρ) * (cost_ineq + cost_eq)
end
@doc raw"""
    ExactPenaltyGrad{S<:SmoothingTechnique, Pr<:ConstrainedProblem, R}

Represent the gradient of the [`ExactPenaltyCost`](@ref) based on a [`ConstrainedProblem`](@ref) `P`
and a parameter ``ρ`` and a smoothing technique, which uses an additional parameter ``u``.

This struct is also a functor in both formats
* `(M, p) -> X` to compute the gradient in allocating fashion.
* `(M, X, p)` to compute the gradient in in-place fashion.

## Fields

* `P`, `ρ`, `u` as mentioned above.

## Constructor

    ExactPenaltyGradient(P::ConstrainedProblem, ρ, u; smoothing=LinearQuadraticHuber())
"""
mutable struct ExactPenaltyGrad{S,Pr,R}
    P::Pr
    ρ::R
    u::R
end
function ExactPenaltyGrad(
    P::Pr, ρ::R, u::R; smoothing=LinearQuadraticHuber()
) where {Pr<:ConstrainedProblem,R}
    return ExactPenaltyGrad{typeof(smoothing),Pr,R}(P, ρ, u)
end
# Default (e.g. functions constraints) - we have to evaluate all gradients
# Since for LogExp the prefactor c seems to not be zero, this might be the best way to go here
function (EG::ExactPenaltyGrad)(M::AbstractManifold, p)
    X = zero_vector(M, p)
    return EG(M, X, p)
end
function (EG::ExactPenaltyGrad{<:LogarithmicSumOfExponentials})(::AbstractManifold, X, p)
    gp = get_inequality_constraints(EG.P, p)
    hp = get_equality_constraints(EG.P, p)
    m = length(gp)
    n = length(hp)
    # start with gradf
    get_gradient!(EG.P, X, p)
    c = 0
    # add grad gs
    (m > 0) && (c = EG.ρ .* exp.(gp ./ EG.u) ./ (1 .+ exp.(gp ./ EG.u)))
    (m > 0) && (X .+= sum(get_grad_inequality_constraints(EG.P, p) .* c))
    # add grad hs
    (n > 0) && (
        c =
            EG.ρ .* (exp.(hp ./ EG.u) .- exp.(-hp ./ EG.u)) ./
            (exp.(hp ./ EG.u) .+ exp.(-hp ./ EG.u))
    )
    (n > 0) && (X .+= sum(get_grad_equality_constraints(EG.P, p) .* c))
    return X
end

# Default (e.g. functions constraints) - we have to evaluate all gradients
function (EG::ExactPenaltyGrad{<:LinearQuadraticHuber})(
    ::AbstractManifold, X, p::P
) where {P}
    gp = get_inequality_constraints(EG.P, p)
    hp = get_equality_constraints(EG.P, p)
    m = length(gp)
    n = length(hp)
    get_gradient!(EG.P, X, p)
    if m > 0
        gradgp = get_grad_inequality_constraints(EG.P, p)
        X .+= sum(gradgp .* (gp .>= EG.u) .* EG.ρ) # add the ones >= u
        X .+= sum(gradgp .* (gp ./ EG.u .* (0 .<= gp .< EG.u)) .* EG.ρ) # add < u
    end
    if n > 0
        c = (hp ./ sqrt.(hp .^ 2 .+ EG.u^2)) .* EG.ρ
        X .+= sum(get_grad_equality_constraints(EG.P, p) .* c)
    end
    return X
end
# Variant 2: Vectors of allocating gradients - we can spare a few gradient evaluations
function (
    EG::ExactPenaltyGrad{
        <:LinearQuadraticHuber,<:ConstrainedProblem{AllocatingEvaluation,<:VectorConstraint}
    }
)(
    ::AbstractManifold, X, p::P
) where {P}
    m = length(EG.P.G)
    n = length(EG.P.H)
    get_gradient!(EG.P, X, p)
    for i in 1:m
        gpi = get_inequality_constraint(EG.P, p, i)
        if (gpi >= 0) # the cases where we have to evaluate the gradient
            # we can just add the gradient scaled by ρ
            (gpi .>= EG.u) && (X .+= gpi .* EG.ρ)
            (0 < gpi < EG.u) && (X .+= gpi .* (gpi / EG.u) * EG.ρ)
        end
    end
    for j in 1:n
        hpj = get_equality_constraint(EG.P, p, j)
        if hpj > 0
            c = hpj / sqrt(hpj^2 + EG.u^2)
            X .+= get_grad_equality_constraint(EG.P, p, j) .* (c * EG.ρ)
        end
    end
    return X
end

# Variant 3: Vectors of mutating gradients - we can spare a few gradient evaluations and allocations
function (
    EG::ExactPenaltyGrad{
        <:LinearQuadraticHuber,<:ConstrainedProblem{MutatingEvaluation,<:VectorConstraint}
    }
)(
    M::AbstractManifold, X, p::P
) where {P}
    m = length(EG.P.G)
    n = length(EG.P.H)
    get_gradient!(EG.P, X, p)
    Y = zero_vector(M, p)
    for i in 1:m
        gpi = get_inequality_constraint(EG.P, p, i)
        if (gpi >= 0) # the cases where we have to evaluate the gradient
            # we only have to evaluate the gradient if gpi > 0
            get_grad_inequality_constraint!(EG.P, Y, p, i)
            # we can just add the gradient scaled by ρ
            (gpi >= EG.u) && (X .+= EG.ρ .* Y)
            # we have to use a different factor, but can exclude the case g = 0 as well
            (0 < gpi < EG.u) && (X .+= ((gpi / EG.u) * EG.ρ) .* Y)
        end
    end
    for j in 1:n
        hpj = get_equality_constraint(EG.P, p, j)
        if hpj > 0
            get_grad_equality_constraint!(EG.P, Y, p, j)
            X .+= ((hpj / sqrt(hpj^2 + EG.u^2)) * EG.ρ) .* Y
        end
    end
    return X
end
