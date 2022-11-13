@doc raw"""
    EPMOptions{P,T} <: Options

Describes the exact penalty method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` – a set point on a manifold as starting point
* `sub_problem` – problem for the subsolver
* `sub_options` – options of the subproblem
* `max_inner_iter` – (`200`) the maximum number of iterations the subsolver should perform in each iteration
* `num_outer_itertgn` – (`30`)
* `ϵ` – (`1e–3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `u` – (`1e–1`) the smoothing parameter and threshold for violation of the constraints
* `u_min` – (`1e-6`) the lower bound for the smoothing parameter and threshold for violation of the constraints
* `ρ` – (`1.0`) the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `min_stepsize` – (`1e-10`) minimal step size
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(300), `[`StopWhenAll`](@ref)`(`[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min), `[`StopWhenChangeLess`](@ref)`(min_stepsize)))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.


# Constructor

    EPMOptions(x)

construct an exact penalty Option with the fields and defaults as above.

# See also
[`exact_penalty_method`](@ref)
"""
mutable struct EPMOptions{P,Pr<:Problem,Op<:Options,TStopping<:StoppingCriterion} <: Options
    x::P
    sub_problem::Pr
    sub_options::Op
    max_inner_iter::Int
    num_outer_itertgn::Int
    ϵ::Real
    ϵ_min::Real
    u::Real
    u_min::Real
    ρ::Real
    θ_ρ::Real
    θ_u::Real
    θ_ϵ::Real
    min_stepsize::Real
    stop::TStopping
    function EPMOptions(
        M::AbstractManifold,
        x0::P,
        sub_problem::Pr,
        sub_options::Op;
        max_inner_iter::Int=200,
        num_outer_itertgn::Int=30,
        ϵ::Real=1e-3,
        ϵ_min::Real=1e-6,
        u::Real=1e-1,
        u_min::Real=1e-6,
        ρ::Real=1.0,
        θ_ρ::Real=0.3,
        min_stepsize::Real=1e-10,
        stopping_criterion::StoppingCriterion=StopWhenAny(
            StopAfterIteration(300),
            StopWhenAll(
                StopWhenSmallerOrEqual(:ϵ, ϵ_min), StopWhenChangeLess(min_stepsize)
            ),
        ),
    ) where {P,Pr<:Problem,Op<:Options}
        o = new{P,Pr,Op,typeof(stopping_criterion)}()
        o.x = x0
        o.sub_problem = sub_problem
        o.sub_options = sub_options
        o.max_inner_iter = max_inner_iter
        o.num_outer_itertgn = num_outer_itertgn
        o.ϵ = ϵ
        o.ϵ_min = ϵ_min
        o.u = u
        o.u_min = u_min
        o.ρ = ρ
        o.θ_ρ = θ_ρ
        o.θ_u = 0.0
        o.θ_ϵ = 0.0
        o.min_stepsize = min_stepsize
        o.stop = stopping_criterion
        return o
    end
end
get_iterate(O::EPMOptions) = O.x

"""
    abstract type SmoothingTechnique

    Specify a smoothing technique, e.g. for the [`ExactPenaltyCost`](@ref) and [`ExactPenaltyGrad`](@ref).
"""
abstract type SmoothingTechnique end

@doc raw"""
    LogarithmicSumOfExponentials <: SmoothingTechnique

Spefiy a smoothing based on ``\max\{a,b\} ≈ u \log(\mathrm{e}^{\frac{a}{u}}+\mathrm{e}^{\frac{b}{u}})``
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
```
"""
struct LinearQuadraticHuber <: SmoothingTechnique end

@doc raw"""
    ExactPenaltyCost{S, Pr, R}

Represent the cost of the exact penalty method based on a [`ConstrainedProblem`](@ref) `P`
and a parameter ``ρ`` given by

```math
f(p) + ρ\Bigl(
    \sum_{i=0}^m \max\{0,g_i(p)} + \sum_{j=0}^n \lvert h_j(p)\rvert
\Bigr),
```
where we use an additional parameter ``u`` and a smoothing technique, e.g.
[`LogarithmicSumOfExponentials`](@ref) or [`LinearQuadraticHuber`](@ref)
to obtain a smooth cost function.

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
    cost_eq_pos_smaller_u =
        (m > 0) ? sum((gp .^ 2 ./ (2 * L.u)) .* ((gp .> 0) .& (gp .<= L.u))) : 0.0
    cost_ineq = cost_eq_greater_u + cost_eq_pos_smaller_u
    cost_eq = (n > 0) ? sum(sqrt.(hp .^ 2 .+ L.u^2)) : 0.0
    return get_cost(L.P, p) + (L.ρ) * (cost_ineq + cost_eq)
end

@doc raw"""
    ExactPenaltyGrad{S<:SmoothingTechnique, Pr<:ConstrainedProblem, R}

Represent the gradient of the [`ExactPenaltyCost`](@ref) based on a [`ConstrainedProblem`](@ref) `P`
and a parameter ``ρ`` and a smoothing parameyterwhere we use an additional parameter ``u``.

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
function (EG::ExactPenaltyGrad{<:LogarithmicSumOfExponentials})(M::AbstractManifold, p)
    gp = get_inequality_constraints(EG.P, p)
    hp = get_equality_constraints(EG.P, p)
    m = length(gp)
    n = length(hp)
    grad_ineq = zero_vector(M, p)
    c = 0
    (m > 0) && (c = EG.ρ .* exp.(gp ./ EG.u) ./ (1 .+ exp.(gp ./ EG.u)))
    (m > 0) && (grad_ineq = sum(get_grad_inequality_constraints(EG.P, p) .* c))
    grad_eq = zero_vector(M, p)
    (n > 0) && (
        c =
            EG.ρ .* (exp.(hp ./ EG.u) .- exp.(-hp ./ EG.u)) ./
            (exp.(hp ./ EG.u) .+ exp.(-hp ./ EG.u))
    )
    (n > 0) && (grad_eq = sum(get_grad_equality_constraints(EG.P, p) .* c))
    return get_gradient(EG.P, p) .+ grad_ineq .+ grad_eq
end

function (EG::ExactPenaltyGrad{<:LinearQuadraticHuber})(M::AbstractManifold, p::P) where {P}
    gp = get_inequality_constraints(EG.P, p)
    hp = get_equality_constraints(EG.P, p)
    m = length(gp)
    n = length(hp)

    grad_ineq = zero_vector(M, p)
    if m > 0
        gradgp = get_grad_inequality_constraints(EG.P, p)
        grad_ineq_cost_greater_u = sum(gradgp .* ((gp .>= 0) .& (gp .>= EG.u)) .* EG.ρ)
        grad_ineq_cost_smaller_u = sum(
            gradgp .* (gp ./ EG.u .* ((gp .>= 0) .& (gp .< EG.u))) .* EG.ρ
        )
        grad_ineq = grad_ineq_cost_greater_u + grad_ineq_cost_smaller_u
    end
    grad_eq = zero_vector(M, p)
    (n > 0) && (
        grad_eq = sum(
            get_grad_inequality_constraints(EG.P, p) .* (hp ./ sqrt.(hp .^ 2 .+ EG.u^2)) .* EG.ρ,
        )
    )
    return get_gradient(EG.P, p) + grad_ineq + grad_eq
end
