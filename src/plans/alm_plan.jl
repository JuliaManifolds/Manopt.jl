#
# Options
#
@doc raw"""
    AugmentedLagrangianMethodOptions{P,T} <: Options

Describes the augmented Lagrangian method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` – a set point on a manifold as starting point
* `sub_problem` – problem for the subsolver
* `sub_options` – options of the subproblem
* `max_inner_iter` – (`200`) the maximum number of iterations the subsolver should perform in each iteration
* `num_outer_itertgn` – (`30`)
* `ϵ` – (`1e–3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `λ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the equality constraints
* `λ_min` – (`- λ_max`) a lower bound for the Lagrange multiplier belonging to the equality constraints
* `μ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the inequality constraints
* `μ` – (`ones(len(`[`get_inequality_constraints`](@ref)`(p,x))`) the Lagrange multiplier with respect to the inequality constraints
* `λ` – (`ones(len(`[`get_equality_constraints`](@ref)`(p,x))`) the Lagrange multiplier with respect to the equality constraints
* `ρ` – (`1.0`) the penalty parameter
* `τ` – (`0.8`) factor for the improvement of the evaluation of the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `θ_ϵ` – (`(ϵ_min/ϵ)^(1/num_outer_itertgn)`) the scaling factor of the accuracy tolerance
* `penalty` – (`Inf`) evaluation of the current penalty term
* `min_stepsize` – (`1e-10`) minimal step size
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(300), `[`StopWhenAll`](@ref)`(`[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min), `[`StopWhenChangeLess`](@ref)`(min_stepsize)))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.


# Constructor

    AugmentedLagrangianMethodOptions(x)

construct an augmented Lagrangian Option with the fields and defaults as above.

# See also
[`augmented_Lagrangian_method`](@ref)
"""
mutable struct AugmentedLagrangianMethodOptions{
    P,Pr<:Problem,Op<:Options,TStopping<:StoppingCriterion
} <: Options
    x::P
    sub_problem::Pr
    sub_options::Op
    ϵ::Real
    ϵ_min::Real
    λ_max::Real
    λ_min::Real
    μ_max::Real
    μ::Vector
    λ::Vector
    ρ::Real
    τ::Real
    θ_ρ::Real
    θ_ϵ::Real
    penalty::Real
    stop::TStopping
    function AugmentedLagrangianMethodOptions(
        M::AbstractManifold,
        p::ConstrainedProblem,
        x0::P,
        sub_problem::Pr,
        sub_options::Op;
        ϵ::Real=1e-3,
        ϵ_min::Real=1e-6,
        λ_max::Real=20.0,
        λ_min::Real=-λ_max,
        μ_max::Real=20.0,
        μ::Vector=ones(length(get_inequality_constraints(p, x0))),
        λ::Vector=ones(length(get_equality_constraints(p, x0))),
        ρ::Real=1.0,
        τ::Real=0.8,
        θ_ρ::Real=0.3,
        maximum_iteration=300,
        ϵ_exponent=1 / 100,
        θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent),
        stopping_criterion::StoppingCriterion=StopAfterIteration(maximum_iteration) | (
            StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(min_stepsize)
        ),
        min_stepsize::Real=1e-10,
    ) where {P,Pr<:Problem,Op<:Options}
        o = new{P,Pr,Op,typeof(stopping_criterion)}()
        o.x = x0
        o.sub_problem = sub_problem
        o.sub_options = sub_options
        o.ϵ = ϵ
        o.ϵ_min = ϵ_min
        o.λ_max = λ_max
        o.λ_min = λ_min
        o.μ_max = μ_max
        o.μ = μ
        o.λ = λ
        o.ρ = ρ
        o.τ = τ
        o.θ_ρ = θ_ρ
        o.θ_ϵ = θ_ϵ
        o.penalty = Inf
        o.stop = stopping_criterion
        return o
    end
end
get_iterate(o::AugmentedLagrangianMethodOptions) = o.x

@doc raw"""
    AugmentedLagrangianCost{Pr,R,T}

Store the parameters ``ρ ∈ \mathbb R``, ``μ ∈ \mathbb R^m``, ``λ ∈ \mathbb R^n``
of the augmented Lagrangian associated to the [`ConstrainedProblem`](@ref) `P`.

This struct is also a functor that can be used as a cost function within a solver,
based on the internal [`ConstrainedProblem`](@ref) we can compute

```math
\mathcal L_\rho(p, μ, λ)
= f(x) + \frac{ρ}{2} \biggl(
    \sum_{j=1}^n \Bigl( h_j(p) + \frac{λ_j}{ρ} \Bigr)^2
    +
    \sum_{i=1}^m \max\Bigl\{ 0, \frac{μ_i}{ρ} + g_i(p) \Bigr\}^2
\Bigr)
```

## Fields

* `P::Pr`, `ρ::R`, `μ::T`, `λ::T` as mentioned above
"""
mutable struct AugmentedLagrangianCost{Pr,R,T}
    P::Pr
    ρ::R
    μ::T
    λ::T
end
function (L::AugmentedLagrangianCost)(::AbstractManifold, p)
    gp = get_inequality_constraints(L.P, p)
    hp = get_equality_constraints(L.P, p)
    m = length(gp)
    n = length(hp)
    c = get_cost(L.P, p)
    d = 0.0
    (m > 0) && (d += sum(max.(zeros(m), L.μ ./ L.ρ .+ gp) .^ 2))
    (n > 0) && (d += sum((hp .+ L.λ ./ L.ρ) .^ 2))
    return c + (L.ρ / 2) * d
end

@doc raw"""
    AugmentedLagrangianGrad{Pr,R,T}

Store the parameters ``ρ ∈ \mathbb R``, ``μ ∈ \mathbb R^m``, ``λ ∈ \mathbb R^n``
of the augmented Lagrangian associated to the [`ConstrainedProblem`](@ref) `P`.

This struct is also a functor that can be used as a cost function within a solver,
based on the internal [`ConstrainedProblem`](@ref) and computes the gradient
``\operatorname{grad} \mathcal L_{ρ}(q, μ, λ)``, see also [`AugmentedLagrangianCost`](@ref).
"""
mutable struct AugmentedLagrangianGrad{Pr,R,T}
    P::Pr
    ρ::R
    μ::T
    λ::T
end
function (LG::AugmentedLagrangianGrad)(M::AbstractManifold, p)
    gp = get_inequality_constraints(LG.P, p)
    hp = get_equality_constraints(LG.P, p)
    m = length(gp)
    n = length(hp)
    grad_ineq = zero_vector(M, p)
    (m > 0) && (
        grad_ineq = sum(
            ((gp .* LG.ρ .+ LG.μ) .* get_grad_inequality_constraints(LG.P, p)) .*
            ((gp .+ LG.μ ./ LG.ρ) .> 0),
        )
    )
    grad_eq = zero_vector(M, p)
    (n > 0) && (grad_eq = sum((hp .* LG.ρ .+ LG.λ) .* get_grad_eqality_constraint(LG.P, p)))
    return get_gradient(LG.P, p) + grad_ineq + grad_eq
end
