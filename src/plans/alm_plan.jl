#
# State
#
@doc raw"""
    AugmentedLagrangianMethodState{P,T} <: AbstractManoptSolverState

Describes the augmented Lagrangian method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` – a set point on a manifold as starting point
* `sub_problem` – problem for the subsolver
* `sub_options` – options of the subproblem
* `ϵ` – (`1e–3`) the accuracy tolerance
* `ϵ_min` – (`1e-6`) the lower bound for the accuracy tolerance
* `λ` – (`ones(len(`[`get_equality_constraints`](@ref)`(p,x))`) the Lagrange multiplier with respect to the equality constraints
* `λ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the equality constraints
* `λ_min` – (`- λ_max`) a lower bound for the Lagrange multiplier belonging to the equality constraints
* `μ` – (`ones(len(`[`get_inequality_constraints`](@ref)`(p,x))`) the Lagrange multiplier with respect to the inequality constraints
* `μ_max` – (`20.0`) an upper bound for the Lagrange multiplier belonging to the inequality constraints
* `ρ` – (`1.0`) the penalty parameter
* `τ` – (`0.8`) factor for the improvement of the evaluation of the penalty parameter
* `θ_ρ` – (`0.3`) the scaling factor of the penalty parameter
* `θ_ϵ` – (`(ϵ_min/ϵ)^(ϵ_exponent)`) the scaling factor of the accuracy tolerance
* `penalty` – evaluation of the current penalty term, initialized to `Inf`.
* `stopping_criterion` – (`(`[`StopAfterIteration`](@ref)`(300) | (`[`StopWhenSmallerOrEqual`](@ref)`(ϵ, ϵ_min) & `[`StopWhenChangeLess`](@ref)`(1e-10))`) a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.


# Constructor

    AugmentedLagrangianMethodState(M::AbstractManifold, P::ConstrainedProblem, x; kwargs...)

construct an augmented Lagrangian method options with the fields and defaults as above,
where the manifold `M` and the [`ConstrainedProblem`](@ref) `P` are used for defaults
in the keyword arguments.

# See also
[`augmented_Lagrangian_method`](@ref)
"""
mutable struct AugmentedLagrangianMethodState{
    P,Pr<:AbstractManoptProblem,Op<:AbstractManoptSolverState,TStopping<:StoppingCriterion
} <: AbstractManoptSolverState
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
    function AugmentedLagrangianMethodState(
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
        ϵ_exponent=1 / 100,
        θ_ϵ=(ϵ_min / ϵ)^(ϵ_exponent),
        stopping_criterion::StoppingCriterion=StopAfterIteration(300) | (
            StopWhenSmallerOrEqual(:ϵ, ϵ_min) & StopWhenChangeLess(1e-10)
        ),
    ) where {P,Pr<:AbstractManoptProblem,Op<:AbstractManoptSolverState}
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
get_iterate(s::AugmentedLagrangianMethodState) = s.x
function set_iterate!(s::AugmentedLagrangianMethodState, p)
    s.x = p
    return s
end
@doc raw"""
    AugmentedLagrangianCost{Pr,R,T}

Stores the parameters ``ρ ∈ \mathbb R``, ``μ ∈ \mathbb R^m``, ``λ ∈ \mathbb R^n``
of the augmented Lagrangian associated to the [`ConstrainedProblem`](@ref) `P`.

This struct is also a functor `(M,p) -> v` that can be used as a cost function within a solver,
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

Stores the parameters ``ρ ∈ \mathbb R``, ``μ ∈ \mathbb R^m``, ``λ ∈ \mathbb R^n``
of the augmented Lagrangian associated to the [`ConstrainedProblem`](@ref) `P`.

This struct is also a functor in both formats
* `(M, p) -> X` to compute the gradient in allocating fashion.
* `(M, X, p)` to compute the gradient in in-place fashion.

based on the internal [`ConstrainedProblem`](@ref) and computes the gradient
``\operatorname{grad} \mathcal L_{ρ}(p, μ, λ)``, see also [`AugmentedLagrangianCost`](@ref).
"""
mutable struct AugmentedLagrangianGrad{Pr,R,T}
    P::Pr
    ρ::R
    μ::T
    λ::T
end
function (LG::AugmentedLagrangianGrad)(M::AbstractManifold, p)
    X = zero_vector(M, p)
    return LG(M, X, p)
end
# default, that is especially when the grad_g and grad_h are functions.
function (LG::AugmentedLagrangianGrad)(::AbstractManifold, X, p)
    gp = get_inequality_constraints(LG.P, p)
    hp = get_equality_constraints(LG.P, p)
    m = length(gp)
    n = length(hp)
    get_gradient!(LG.P, X, p)
    (m > 0) && (
        X += sum(
            ((gp .* LG.ρ .+ LG.μ) .* get_grad_inequality_constraints(LG.P, p)) .*
            ((gp .+ LG.μ ./ LG.ρ) .> 0),
        )
    )
    (n > 0) && (X += sum((hp .* LG.ρ .+ LG.λ) .* get_grad_equality_constraints(LG.P, p)))
    return X
end
# Allocating vector -> we can omit a few of the ineq gradients.
function (
    LG::AugmentedLagrangianGrad{
        <:ConstrainedProblem{AllocatingEvaluation,<:VectorConstraint}
    }
)(
    ::AbstractManifold, X, p
)
    m = length(LG.P.G)
    n = length(LG.P.H)
    get_gradient!(LG.P, X, p)
    for i in 1:m
        gpi = get_inequality_constraint(LG.P, p, i)
        if (gpi + LG.μ[i] / LG.ρ) > 0 # only evaluate gradient if necessary
            X .+= (gpi * LG.ρ + LG.μ[i]) .* get_grad_inequality_constraint(LG.P, p, i)
        end
    end
    for j in 1:n
        hpj = get_equality_constraint(LG.P, p, j)
        X .+= (hpj * LG.ρ + LG.λ[j]) .* get_grad_equality_constraint(LG.P, p, j)
    end
    return X
end
# mutating vector -> we can omit a few of the ineq gradients and allocations.
function (
    LG::AugmentedLagrangianGrad{<:ConstrainedProblem{InplaceEvaluation,<:VectorConstraint}}
)(
    M::AbstractManifold, X, p
)
    m = length(LG.P.G)
    n = length(LG.P.H)
    get_gradient!(LG.P, X, p)
    Y = zero_vector(M, p)
    for i in 1:m
        gpi = get_inequality_constraint(LG.P, p, i)
        if (gpi + LG.μ[i] / LG.ρ) > 0 # only evaluate gradient if necessary
            # evaluate in place
            get_grad_inequality_constraint!(LG.P, Y, p, i)
            X .+= (gpi * LG.ρ + LG.μ[i]) .* Y
        end
    end
    for j in 1:n
        # evaluate in place
        hpj = get_equality_constraint(LG.P, p, j)
        get_grad_equality_constraint!(LG.P, Y, p, j)
        X .+= (hpj * LG.ρ + LG.λ[j]) * Y
    end
    get_gradient!(LG.P, Y, p)
    return X
end
