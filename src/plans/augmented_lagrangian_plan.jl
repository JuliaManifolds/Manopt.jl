@doc raw"""
    AugmentedLagrangianCost{CO,R,T}

Stores the parameters ``ρ ∈ ℝ``, ``μ ∈ ℝ^m``, ``λ ∈ ℝ^n``
of the augmented Lagrangian associated to the [`ConstrainedManifoldObjective`](@ref) `co`.

This struct is also a functor `(M,p) -> v` that can be used as a cost function within a solver,
based on the internal [`ConstrainedManifoldObjective`](@ref) it computes

```math
\mathcal L_\rho(p, μ, λ)
= f(x) + \frac{ρ}{2} \biggl(
    \sum_{j=1}^n \Bigl( h_j(p) + \frac{λ_j}{ρ} \Bigr)^2
    +
    \sum_{i=1}^m \max\Bigl\{ 0, \frac{μ_i}{ρ} + g_i(p) \Bigr\}^2
\Bigr)
```

## Fields

* `co::CO`, `ρ::R`, `μ::T`, `λ::T` as mentioned in the formula, where ``R`` should be the
number type used and ``T`` the vector type.

# Constructor

    AugmentedLagrangianCost(co, ρ, μ, λ)
"""
mutable struct AugmentedLagrangianCost{CO,R,T}
    co::CO
    ρ::R
    μ::T
    λ::T
end
function set_manopt_parameter!(alc::AugmentedLagrangianCost, ::Val{:ρ}, ρ)
    alc.ρ = ρ
    return alc
end
function set_manopt_parameter!(alc::AugmentedLagrangianCost, ::Val{:μ}, μ)
    alc.μ = μ
    return alc
end
function set_manopt_parameter!(alc::AugmentedLagrangianCost, ::Val{:λ}, λ)
    alc.λ = λ
    return alc
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

Stores the parameters ``ρ ∈ ℝ``, ``μ ∈ ℝ^m``, ``λ ∈ ℝ^n``
of the augmented Lagrangian associated to the [`ConstrainedManifoldObjective`](@ref) `co`.

This struct is also a functor in both formats
* `(M, p) -> X` to compute the gradient in allocating fashion.
* `(M, X, p)` to compute the gradient in in-place fashion.

additionally this gadient does accept a positional last argument to specify the `range`
for the internal gradient call of the contrained objective.

based on the internal [`ConstrainedManifoldObjective`](@ref) and computes the gradient
``\operatorname{grad} \mathcal L_{ρ}(p, μ, λ)``, see also [`AugmentedLagrangianCost`](@ref).

## Fields

* `co::CO`, `ρ::R`, `μ::T`, `λ::T` as mentioned in the formula, where ``R`` should be the
number type used and ``T`` the vector type.

# Constructor

    AugmentedLagrangianGrad(co, ρ, μ, λ)

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

function set_manopt_parameter!(alg::AugmentedLagrangianGrad, ::Val{:ρ}, ρ)
    alg.ρ = ρ
    return alg
end
function set_manopt_parameter!(alg::AugmentedLagrangianGrad, ::Val{:μ}, μ)
    alg.μ = μ
    return alg
end
function set_manopt_parameter!(alg::AugmentedLagrangianGrad, ::Val{:λ}, λ)
    alg.λ = λ
    return alg
end

# default, that is especially when the `grad_g` and `grad_h` are functions.
function (LG::AugmentedLagrangianGrad)(
    M::AbstractManifold, X, p, range=NestedPowerRepresentation()
)
    gp = get_inequality_constraint(M, LG.co, p, :)
    hp = get_equality_constraint(M, LG.co, p, :)
    m = length(gp)
    n = length(hp)
    get_gradient!(M, X, LG.co, p)
    if m > 0
        indices = (gp .+ LG.μ ./ LG.ρ) .> 0
        if sum(indices) > 0
            weights = (gp .* LG.ρ .+ LG.μ)[indices]
            X .+= sum(
                weights .* get_grad_inequality_constraint(M, LG.co, p, indices, range)
            )
        end
    end
    if n > 0
        X .+= sum((hp .* LG.ρ .+ LG.λ) .* get_grad_equality_constraints(M, LG.co, p, range))
    end
    return X
end
