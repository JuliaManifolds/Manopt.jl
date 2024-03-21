@doc raw"""
    LagrangianCost{CO,R,T}

Stores the parameters ``μ ∈ ℝ^m``, ``λ ∈ ℝ^n``
of the Lagrangian associated to the [`ConstrainedManifoldObjective`](@ref) `co`.

This struct is also a functor `(M,p) -> v` that can be used as a cost function within a solver,
based on the internal [`ConstrainedManifoldObjective`](@ref) it computes

```math
\mathcal L_\rho(p, μ, λ)
= f(p) + \sum_{i=1}^m λ_i g_i(p) + \sum_{j=1}^n μ_j h_j(p)
= f(p) + λ^{\intercal} g(p) + μ^{\intercal} h(p)
```

## Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned in the formula, where ``T`` should be the
vector type used.

# Constructor

    LagrangianCost(co, μ, λ)
"""
mutable struct LagrangianCost{CO,T}
    co::CO
    μ::T
    λ::T
end
function set_manopt_parameter!(lc::LagrangianCost, ::Val{:μ}, μ)
    lc.μ = μ
    return lc
end
function set_manopt_parameter!(lc::LagrangianCost, ::Val{:λ}, λ)
    lc.λ = λ
    return lc
end
function (L::LagrangianCost)(M::AbstractManifold, p)
    c = get_cost(M, L.co, p)
    gp = get_equality_constraints(M, L.co, p)
    hp = get_inequality_constraints(M, L.co, p)
    return c + (L.λ)'gp + (L.μ)'hp
end

@doc raw"""
    LagrangianGrad{CO,R,T}

Stores the parameters ``ρ ∈ ℝ``, ``μ ∈ ℝ^m``, ``λ ∈ ℝ^n``
of the Lagrangian associated to the [`ConstrainedManifoldObjective`](@ref) `co`.

This struct is also a functor in both formats
* `(M, p) -> X` to compute the gradient in allocating fashion.
* `(M, X, p)` to compute the gradient in in-place fashion.

based on the internal [`ConstrainedManifoldObjective`](@ref) and computes the gradient
``\operatorname{grad} \mathcal L_{ρ}(p, μ, λ)``, see also [`LagrangianCost`](@ref).

## Fields

* `co::CO`, `μ::T`, `λ::T` as mentioned in the formula, where ``T`` should be the
vector type used.

# Constructor

    LagrangianGrad(co, μ, λ)

"""
mutable struct LagrangianGrad{CO,T}
    co::CO
    μ::T
    λ::T
end
function (LG::LagrangianGrad)(M::AbstractManifold, p)
    X = zero_vector(M, p)
    return LG(M, X, p)
end
function set_manopt_parameter!(lg::LagrangianGrad, ::Val{:μ}, μ)
    lg.μ = μ
    return lg
end
function set_manopt_parameter!(lg::LagrangianGrad, ::Val{:λ}, λ)
    lg.λ = λ
    return lg
end

# default, that is especially when the `grad_g` and `grad_h` are functions.
function (LG::LagrangianGrad)(M::AbstractManifold, X, p)
    grad_fp = get_gradient(M, X, LC.co, p)
    grad_gp = get_grad_equality_constraints(M, LG.co, p) 
    grad_hp = get_grad_equality_constraints(M, LG.co, p) 
    return grad_fp + (LG.λ)'grad_gp + (LG.μ)'grad_hp
end
# Allocating vector -> omit a few of the inequality gradient evaluations.
function (
    LG::LagrangianGrad{
        <:ConstrainedManifoldObjective{AllocatingEvaluation,<:VectorConstraint}
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
# mutating vector -> omit a few of the inequality gradients and allocations.
function (
    LG::LagrangianGrad{
        <:ConstrainedManifoldObjective{InplaceEvaluation,<:VectorConstraint}
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

