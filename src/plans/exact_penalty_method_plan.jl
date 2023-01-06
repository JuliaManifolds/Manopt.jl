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

Represent the cost of the exact penalty method based on a [`ConstrainedManifoldObjective`](@ref) `P`
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

    ExactPenaltyCost(co::ConstrainedManifoldObjective, ρ, u; smoothing=LinearQuadraticHuber())
"""
mutable struct ExactPenaltyCost{S,CO,R}
    co::CO
    ρ::R
    u::R
end
function ExactPenaltyCost(
    co::ConstrainedManifoldObjective, ρ::R, u::R; smoothing=LinearQuadraticHuber()
) where {R}
    return ExactPenaltyCost{typeof(smoothing),typeof(co),R}(co, ρ, u)
end
function set_manopt_parameter!(epc::ExactPenaltyCost, ::Val{:ρ}, ρ)
    epc.ρ = ρ
    return epc
end
function set_manopt_parameter!(epc::ExactPenaltyCost, ::Val{:u}, u)
    epc.u = u
    return epc
end
function (L::ExactPenaltyCost{<:LogarithmicSumOfExponentials})(M::AbstractManifold, p)
    gp = get_inequality_constraints(M, L.co, p)
    hp = get_equality_constraints(M, L.co, p)
    m = length(gp)
    n = length(hp)
    cost_ineq = (m > 0) ? sum(L.u .* log.(1 .+ exp.(gp ./ L.u))) : 0.0
    cost_eq = (n > 0) ? sum(L.u .* log.(exp.(hp ./ L.u) .+ exp.(-hp ./ L.u))) : 0.0
    return get_cost(M, L.co, p) + (L.ρ) * (cost_ineq + cost_eq)
end
function (L::ExactPenaltyCost{<:LinearQuadraticHuber})(M::AbstractManifold, p)
    gp = get_inequality_constraints(M, L.co, p)
    hp = get_equality_constraints(M, L.co, p)
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

* `P`, `ρ`, `u` as mentioned above.

## Constructor

    ExactPenaltyGradient(co::ConstrainedManifoldObjective, ρ, u; smoothing=LinearQuadraticHuber())
"""
mutable struct ExactPenaltyGrad{S,CO,R}
    co::CO
    ρ::R
    u::R
end
function set_manopt_parameter!(epg::ExactPenaltyGrad, ::Val{:ρ}, ρ)
    epg.ρ = ρ
    return epg
end
function set_manopt_parameter!(epg::ExactPenaltyGrad, ::Val{:u}, u)
    epg.u = u
    return epg
end
function ExactPenaltyGrad(
    co::ConstrainedManifoldObjective, ρ::R, u::R; smoothing=LinearQuadraticHuber()
) where {R}
    return ExactPenaltyGrad{typeof(smoothing),typeof(co),R}(co, ρ, u)
end
# Default (e.g. functions constraints) - we have to evaluate all gradients
# Since for LogExp the prefactor c seems to not be zero, this might be the best way to go here
function (EG::ExactPenaltyGrad)(M::AbstractManifold, p)
    X = zero_vector(M, p)
    return EG(M, X, p)
end
function (EG::ExactPenaltyGrad{<:LogarithmicSumOfExponentials})(M::AbstractManifold, X, p)
    gp = get_inequality_constraints(M, EG.co, p)
    hp = get_equality_constraints(M, EG.co, p)
    m = length(gp)
    n = length(hp)
    # start with gradf
    get_gradient!(M, X, EG.co, p)
    c = 0
    # add grad gs
    (m > 0) && (c = EG.ρ .* exp.(gp ./ EG.u) ./ (1 .+ exp.(gp ./ EG.u)))
    (m > 0) && (X .+= sum(get_grad_inequality_constraints(M, EG.co, p) .* c))
    # add grad hs
    (n > 0) && (
        c =
            EG.ρ .* (exp.(hp ./ EG.u) .- exp.(-hp ./ EG.u)) ./
            (exp.(hp ./ EG.u) .+ exp.(-hp ./ EG.u))
    )
    (n > 0) && (X .+= sum(get_grad_equality_constraints(M, EG.co, p) .* c))
    return X
end

# Default (e.g. functions constraints) - we have to evaluate all gradients
function (EG::ExactPenaltyGrad{<:LinearQuadraticHuber})(
    M::AbstractManifold, X, p::P
) where {P}
    gp = get_inequality_constraints(M, EG.co, p)
    hp = get_equality_constraints(M, EG.co, p)
    m = length(gp)
    n = length(hp)
    get_gradient!(M, X, EG.co, p)
    if m > 0
        gradgp = get_grad_inequality_constraints(M, EG.co, p)
        X .+= sum(gradgp .* (gp .>= EG.u) .* EG.ρ) # add the ones >= u
        X .+= sum(gradgp .* (gp ./ EG.u .* (0 .<= gp .< EG.u)) .* EG.ρ) # add < u
    end
    if n > 0
        c = (hp ./ sqrt.(hp .^ 2 .+ EG.u^2)) .* EG.ρ
        X .+= sum(get_grad_equality_constraints(M, EG.co, p) .* c)
    end
    return X
end
# Variant 2: Vectors of allocating gradients - we can spare a few gradient evaluations
function (
    EG::ExactPenaltyGrad{
        <:LinearQuadraticHuber,
        <:ConstrainedManifoldObjective{AllocatingEvaluation,<:VectorConstraint},
    }
)(
    M::AbstractManifold, X, p::P
) where {P}
    m = length(EG.co.g)
    n = length(EG.co.h)
    get_gradient!(M, X, EG.co, p)
    for i in 1:m
        gpi = get_inequality_constraint(M, EG.co, p, i)
        if (gpi >= 0) # the cases where we have to evaluate the gradient
            # we can just add the gradient scaled by ρ
            (gpi .>= EG.u) && (X .+= gpi .* EG.ρ)
            (0 < gpi < EG.u) && (X .+= gpi .* (gpi / EG.u) * EG.ρ)
        end
    end
    for j in 1:n
        hpj = get_equality_constraint(M, EG.co, p, j)
        if hpj > 0
            c = hpj / sqrt(hpj^2 + EG.u^2)
            X .+= get_grad_equality_constraint(M, EG.co, p, j) .* (c * EG.ρ)
        end
    end
    return X
end

# Variant 3: Vectors of mutating gradients - we can spare a few gradient evaluations and allocations
function (
    EG::ExactPenaltyGrad{
        <:LinearQuadraticHuber,
        <:ConstrainedManifoldObjective{InplaceEvaluation,<:VectorConstraint},
    }
)(
    M::AbstractManifold, X, p::P
) where {P}
    m = length(EG.co.g)
    n = length(EG.co.h)
    get_gradient!(M, X, EG.co, p)
    Y = zero_vector(M, p)
    for i in 1:m
        gpi = get_inequality_constraint(M, EG.co, p, i)
        if (gpi >= 0) # the cases where we have to evaluate the gradient
            # we only have to evaluate the gradient if gpi > 0
            get_grad_inequality_constraint!(M, Y, EG.co, p, i)
            # we can just add the gradient scaled by ρ
            (gpi >= EG.u) && (X .+= EG.ρ .* Y)
            # we have to use a different factor, but can exclude the case g = 0 as well
            (0 < gpi < EG.u) && (X .+= ((gpi / EG.u) * EG.ρ) .* Y)
        end
    end
    for j in 1:n
        hpj = get_equality_constraint(M, EG.co, p, j)
        if hpj > 0
            get_grad_equality_constraint!(M, Y, EG.co, p, j)
            X .+= ((hpj / sqrt(hpj^2 + EG.u^2)) * EG.ρ) .* Y
        end
    end
    return X
end
