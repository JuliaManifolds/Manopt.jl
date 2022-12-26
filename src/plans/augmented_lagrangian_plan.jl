@doc raw"""
    AugmentedLagrangianCost{CO,R,T}

Stores the parameters ``ρ ∈ \mathbb R``, ``μ ∈ \mathbb R^m``, ``λ ∈ \mathbb R^n``
of the augmented Lagrangian associated to the [`ConstrainedObjective`](@ref) `co`.

This struct is also a functor `(M,p) -> v` that can be used as a cost function within a solver,
based on the internal [`ConstrainedObjective`](@ref) we can compute

```math
\mathcal L_\rho(p, μ, λ)
= f(x) + \frac{ρ}{2} \biggl(
    \sum_{j=1}^n \Bigl( h_j(p) + \frac{λ_j}{ρ} \Bigr)^2
    +
    \sum_{i=1}^m \max\Bigl\{ 0, \frac{μ_i}{ρ} + g_i(p) \Bigr\}^2
\Bigr)
```

## Fields

* `co::CO`, `ρ::R`, `μ::T`, `λ::T` as mentioned above
"""
mutable struct AugmentedLagrangianCost{CO,R,T}
    co::CO
    ρ::R
    μ::T
    λ::T
end
function (L::AugmentedLagrangianCost)(M::AbstractManifold, p)
    gp = get_inequality_constraints(M, L.co, p)
    hp = get_equality_constraints(M, L.co, p)
    m = length(gp)
    n = length(hp)
    c = get_cost(M, L.co, p)
    d = 0.0
    (m > 0) && (d += sum(max.(zeros(m), L.μ ./ L.ρ .+ gp) .^ 2))
    (n > 0) && (d += sum((hp .+ L.λ ./ L.ρ) .^ 2))
    return c + (L.ρ / 2) * d
end

@doc raw"""
    AugmentedLagrangianGrad{CO,R,T}

Stores the parameters ``ρ ∈ \mathbb R``, ``μ ∈ \mathbb R^m``, ``λ ∈ \mathbb R^n``
of the augmented Lagrangian associated to the [`ConstrainedObjective`](@ref) `co`.

This struct is also a functor in both formats
* `(M, p) -> X` to compute the gradient in allocating fashion.
* `(M, X, p)` to compute the gradient in in-place fashion.

based on the internal [`ConstrainedObjective`](@ref) and computes the gradient
``\operatorname{grad} \mathcal L_{ρ}(p, μ, λ)``, see also [`AugmentedLagrangianCost`](@ref).
"""
mutable struct AugmentedLagrangianGrad{CO,R,T}
    co::CO
    ρ::R
    μ::T
    λ::T
end
function (LG::AugmentedLagrangianGrad)(M::AbstractManifold, p)
    X = zero_vector(M, p)
    return LG(M, X, p)
end
# default, that is especially when the grad_g and grad_h are functions.
function (LG::AugmentedLagrangianGrad)(M::AbstractManifold, X, p)
    gp = get_inequality_constraints(M, LG.co, p)
    hp = get_equality_constraints(M, LG.co, p)
    m = length(gp)
    n = length(hp)
    get_gradient!(M, X, LG.co, p)
    (m > 0) && (
        X += sum(
            ((gp .* LG.ρ .+ LG.μ) .* get_grad_inequality_constraints(M, LG.co, p)) .*
            ((gp .+ LG.μ ./ LG.ρ) .> 0),
        )
    )
    (n > 0) &&
        (X += sum((hp .* LG.ρ .+ LG.λ) .* get_grad_equality_constraints(M, LG.co, p)))
    return X
end
# Allocating vector -> we can omit a few of the ineq gradients.
function (
    LG::AugmentedLagrangianGrad{
        <:ConstrainedObjective{AllocatingEvaluation,<:VectorConstraint}
    }
)(
    M::AbstractManifold, X, p
)
    m = length(LG.co.g)
    n = length(LG.co.h)
    get_gradient!(M, X, LG.co, p)
    for i in 1:m
        gpi = get_inequality_constraint(M, LG.co, p, i)
        if (gpi + LG.μ[i] / LG.ρ) > 0 # only evaluate gradient if necessary
            X .+= (gpi * LG.ρ + LG.μ[i]) .* get_grad_inequality_constraint(M, LG.co, p, i)
        end
    end
    for j in 1:n
        hpj = get_equality_constraint(M, LG.co, p, j)
        X .+= (hpj * LG.ρ + LG.λ[j]) .* get_grad_equality_constraint(M, LG.co, p, j)
    end
    return X
end
# mutating vector -> we can omit a few of the ineq gradients and allocations.
function (
    LG::AugmentedLagrangianGrad{
        <:ConstrainedObjective{InplaceEvaluation,<:VectorConstraint}
    }
)(
    M::AbstractManifold, X, p
)
    m = length(LG.co.g)
    n = length(LG.co.h)
    get_gradient!(M, X, LG.co, p)
    Y = zero_vector(M, p)
    for i in 1:m
        gpi = get_inequality_constraint(M, LG.co, p, i)
        if (gpi + LG.μ[i] / LG.ρ) > 0 # only evaluate gradient if necessary
            # evaluate in place
            get_grad_inequality_constraint!(M, Y, LG.co, p, i)
            X .+= (gpi * LG.ρ + LG.μ[i]) .* Y
        end
    end
    for j in 1:n
        # evaluate in place
        hpj = get_equality_constraint(M, LG.co, p, j)
        get_grad_equality_constraint!(M, Y, LG.co, p, j)
        X .+= (hpj * LG.ρ + LG.λ[j]) * Y
    end
    return X
end
