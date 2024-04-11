@doc raw"""
    LagrangianCost{CO,R,T}

Stores the parameters ``μ ∈ ℝ^m``, ``λ ∈ ℝ^n``
of the Lagrangian associated to the [`ConstrainedManifoldObjective`](@ref) `co`.

This struct is also a functor `(M,p) -> v` that can be used as a cost function within a solver,
based on the internal [`ConstrainedManifoldObjective`](@ref) it computes

```math
\mathcal L_\rho(p, μ, λ)
= f(p) + \sum_{i=1}^m μ_i g_i(p) + \sum_{j=1}^n λ_j h_j(p)
= f(p) + μ^{\intercal} g(p) + λ^{\intercal} h(p)
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

function (lc::LagrangianCost)(M::AbstractManifold, p)
    gp = get_inequality_constraints(M, lc.co, p)
    hp = get_equality_constraints(M, lc.co, p)
    m = length(gp)
    n = length(hp)
    L = get_cost(M, lc.co, p)
    (m > 0) && (L += (lc.μ)'gp)
    (n > 0) && (L += (lc.λ)'hp)
    return L
end

@doc raw"""
    LagrangianGrad{CO,R,T}

Stores the parameters ``μ ∈ ℝ^m``, ``λ ∈ ℝ^n``
of the Lagrangian associated to the [`ConstrainedManifoldObjective`](@ref) `co`.

This struct is also a functor in both formats
* `(M, p) -> X` to compute the gradient in allocating fashion.
* `(M, X, p)` to compute the gradient in in-place fashion.

based on the internal [`ConstrainedManifoldObjective`](@ref) and computes the gradient
``\operatorname{grad} \mathcal L(p, μ, λ)``, see also [`LagrangianCost`](@ref).

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

function (lg::LagrangianGrad)(M::AbstractManifold, p)
    X = zero_vector(M, p)
    return lg(M, X, p)
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
function (lg::LagrangianGrad)(M::AbstractManifold, X, p)
    grad_gp = get_grad_inequality_constraints(M, lg.co, p)
    grad_hp = get_grad_equality_constraints(M, lg.co, p)
    m = length(grad_gp)
    n = length(grad_hp)
    get_gradient!(M, X, lg.co, p)
    (m > 0) && (X .+= sum(lg.μ .* grad_gp))
    (n > 0) && (X .+= sum(lg.λ .* grad_hp))
    return X
end

# Allocating vector -> omit a few of the inequality gradient evaluations.
function (
    lg::LagrangianGrad{
        <:ConstrainedManifoldObjective{AllocatingEvaluation,<:VectorConstraint}
    }
)(
    M::AbstractManifold, X, p
)
    m = length(lg.co.g)
    n = length(lg.co.h)
    get_gradient!(M, X, lg.co, p)
    for i in 1:m
        gpi = get_inequality_constraint(M, lg.co, p, i)
        (gpi < 0) && (X .+= lg.μ[i] .* get_grad_inequality_constraint(M, lg.co, p, i))
    end
    for j in 1:n
        X .+= lg.λ[j] .* get_grad_equality_constraint(M, lg.co, p, j)
    end
    return X
end

# mutable struct LagrangianHess{CO,T}
#     co::CO
#     μ::T
#     λ::T
# end

# function (lh::LagrangianHess)(M::AbstractManifold, p)
#     d = manifold_dimension(M)
#     Y = zeros(d, d)
#     return lh(M, H, p)
# end

# function set_manopt_parameter!(lh::LagrangianHess, ::Val{:μ}, μ)
#     lh.μ = μ
#     return lh
# end
# function set_manopt_parameter!(lh::LagrangianHess, ::Val{:λ}, λ)
#     lh.λ = λ
#     return lh
# end

# # default, that is especially when the `grad_g` and `grad_h` are functions.
# function (lh::LagrangianHess)(M::AbstractManifold, Y, p, X)
#     H = get_hessian_function(lh.co)
#     hess_gp = get_grad_inequality_constraints(M, lg.co, p)
#     hess_hp = get_grad_equality_constraints(M, lg.co, p)
#     m = length(grad_gp)
#     n = length(grad_hp)

#     get_gradient!(M, X, lg.co, p)
#     (m > 0) && (X .+= sum(lg.μ .* grad_gp))
#     (n > 0) && (X .+= sum(lg.λ .* grad_hp))
#     return X
# end
