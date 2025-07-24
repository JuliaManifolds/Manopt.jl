"""
    abstract type SmoothingTechnique

Specify a smoothing technique, see for example [`ExactPenaltyCost`](@ref) and [`ExactPenaltyGrad`](@ref).
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

Represent the cost of the exact penalty method based on a [`ConstrainedManifoldObjective`](@ref) `P`
and a parameter ``ρ`` given by

```math
f(p) + ρ\Bigl(
    \sum_{i=0}^m \max\{0,g_i(p)\} + \sum_{j=0}^n \lvert h_j(p)\rvert
\Bigr),
```
where an additional parameter ``u`` is used as well as a smoothing technique,
for example [`LogarithmicSumOfExponentials`](@ref) or [`LinearQuadraticHuber`](@ref)
to obtain a smooth cost function. This struct is also a functor `(M,p) -> v` of the cost ``v``.

## Fields

* `ρ`, `u`: as described in the mathematical formula, .
* `co`:     the original cost

## Constructor

    ExactPenaltyCost(co::ConstrainedManifoldObjective, ρ, u; smoothing=LinearQuadraticHuber())
"""
mutable struct ExactPenaltyCost{S, CO, R}
    co::CO
    ρ::R
    u::R
end
function ExactPenaltyCost(
        co::ConstrainedManifoldObjective, ρ::R, u::R; smoothing = LinearQuadraticHuber()
    ) where {R}
    return ExactPenaltyCost{typeof(smoothing), typeof(co), R}(co, ρ, u)
end
function set_parameter!(epc::ExactPenaltyCost, ::Val{:ρ}, ρ)
    epc.ρ = ρ
    return epc
end
function set_parameter!(epc::ExactPenaltyCost, ::Val{:u}, u)
    epc.u = u
    return epc
end
function (L::ExactPenaltyCost{<:LogarithmicSumOfExponentials})(M::AbstractManifold, p)
    gp = get_inequality_constraint(M, L.co, p, :)
    hp = get_equality_constraint(M, L.co, p, :)
    m = length(gp)
    n = length(hp)
    cost_ineq = (m > 0) ? sum(L.u .* log.(1 .+ exp.(gp ./ L.u))) : 0.0
    cost_eq = (n > 0) ? sum(L.u .* log.(exp.(hp ./ L.u) .+ exp.(-hp ./ L.u))) : 0.0
    return get_cost(M, L.co, p) + (L.ρ) * (cost_ineq + cost_eq)
end
function (L::ExactPenaltyCost{<:LinearQuadraticHuber})(M::AbstractManifold, p)
    gp = get_inequality_constraint(M, L.co, p, :)
    hp = get_equality_constraint(M, L.co, p, :)
    m = length(gp)
    n = length(hp)
    cost_eq_greater_u = (m > 0) ? sum((gp .- L.u / 2) .* (gp .> L.u)) : 0.0
    cost_eq_pos_smaller_u = (m > 0) ? sum((gp .^ 2 ./ (2 * L.u)) .* (0 .< gp .<= L.u)) : 0.0
    cost_ineq = cost_eq_greater_u + cost_eq_pos_smaller_u
    cost_eq = (n > 0) ? sum(sqrt.(hp .^ 2 .+ L.u^2)) : 0.0
    return get_cost(M, L.co, p) + (L.ρ) * (cost_ineq + cost_eq)
end

@doc raw"""
    ExactPenaltyGrad{S, CO, R}

Represent the gradient of the [`ExactPenaltyCost`](@ref) based on a [`ConstrainedManifoldObjective`](@ref) `co`
and a parameter ``ρ`` and a smoothing technique, which uses an additional parameter ``u``.

This struct is also a functor in both formats
* `(M, p) -> X` to compute the gradient in allocating fashion.
* `(M, X, p)` to compute the gradient in in-place fashion.

## Fields

* `ρ`, `u` as stated before
* `co` the nonsmooth objective

## Constructor

    ExactPenaltyGradient(co::ConstrainedManifoldObjective, ρ, u; smoothing=LinearQuadraticHuber())
"""
mutable struct ExactPenaltyGrad{S, CO, R}
    co::CO
    ρ::R
    u::R
end
function set_parameter!(epg::ExactPenaltyGrad, ::Val{:ρ}, ρ)
    epg.ρ = ρ
    return epg
end
function set_parameter!(epg::ExactPenaltyGrad, ::Val{:u}, u)
    epg.u = u
    return epg
end
function ExactPenaltyGrad(
        co::ConstrainedManifoldObjective, ρ::R, u::R; smoothing = LinearQuadraticHuber()
    ) where {R}
    return ExactPenaltyGrad{typeof(smoothing), typeof(co), R}(co, ρ, u)
end
# Default (functions constraints): evaluate all gradients
# Since for LogExp the pre-factor c seems to not be zero, this might be the best way to go here
function (EG::ExactPenaltyGrad)(M::AbstractManifold, p)
    X = zero_vector(M, p)
    return EG(M, X, p)
end
function (EG::ExactPenaltyGrad{<:LogarithmicSumOfExponentials})(M::AbstractManifold, X, p)
    gp = get_inequality_constraint(M, EG.co, p, :)
    hp = get_equality_constraint(M, EG.co, p, :)
    m = length(gp)
    n = length(hp)
    # start with `gradf`
    get_gradient!(M, X, EG.co, p)
    c = 0
    # add gradient of the components of g
    (m > 0) && (c = EG.ρ .* exp.(gp ./ EG.u) ./ (1 .+ exp.(gp ./ EG.u)))
    (m > 0) && (X .+= sum(get_grad_inequality_constraint(M, EG.co, p, :) .* c))
    # add gradient of the components of h
    (n > 0) && (
        c =
            EG.ρ .* (exp.(hp ./ EG.u) .- exp.(-hp ./ EG.u)) ./
            (exp.(hp ./ EG.u) .+ exp.(-hp ./ EG.u))
    )
    (n > 0) && (X .+= sum(get_grad_equality_constraint(M, EG.co, p, :) .* c))
    return X
end

# Default (functions constraints): evaluate all gradients
function (EG::ExactPenaltyGrad{<:LinearQuadraticHuber})(
        M::AbstractManifold, X, p::P
    ) where {P}
    gp = get_inequality_constraint(M, EG.co, p, :)
    hp = get_equality_constraint(M, EG.co, p, :)
    m = length(gp)
    n = length(hp)
    get_gradient!(M, X, EG.co, p)
    if m > 0
        gradgp = get_grad_inequality_constraint(M, EG.co, p, :)
        X .+= sum(gradgp .* (gp .>= EG.u) .* EG.ρ) # add the ones >= u
        X .+= sum(gradgp .* (gp ./ EG.u .* (0 .<= gp .< EG.u)) .* EG.ρ) # add < u
    end
    if n > 0
        c = (hp ./ sqrt.(hp .^ 2 .+ EG.u^2)) .* EG.ρ
        X .+= sum(get_grad_equality_constraint(M, EG.co, p, :) .* c)
    end
    return X
end
