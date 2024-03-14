mutable struct LagrangianCost{CO,T}
    co::CO
    λ::T
    μ::T
end

function set_manopt_parameter!(lc::LagrangianCost, ::Val{:λ}, λ)
    lc.λ = λ
    return lc
end

function set_manopt_parameter!(lc::LagrangianCost, ::Val{:μ}, μ)
    lc.μ = μ
    return lc
end

function (L::LagrangianCost)(M::AbstractManifold, p)
    gp = get_equality_constraints(M, L.co, p)
    hp = get_inequality_constraints(M, L.co, p)
    c = get_cost(M, L.co, p)
    return c + (L.λ)'gp + (L.μ)'hp
end


mutable struct LagrangianGrad{CO,T}
    co::CO
    λ::T
    μ::T
end

function (LG::LagrangianGrad)(M::AbstractManifold, p)
    X = zero_vector(M, p)
    return LG(M, X, p)
end

function set_manopt_parameter!(lg::LagrangianGrad, ::Val{:λ}, λ)
    lg.λ = λ
    return lg
end

function set_manopt_parameter!(lg::LagrangianGrad, ::Val{:μ}, μ)
    lg.μ = μ
    return lg
end

####
####
####
####
# ASK ABOUT THIS below
# does get_grad_equality_constraints give jacobian og constraint function??

# default, that is especially when the `grad_g` and `grad_h` are functions.
function (LG::LagrangianGrad)(M::AbstractManifold, X, p)
    gp = get_equality_constraints(M, LG.co, p)
    hp = get_inequality_constraints(M, LG.co, p)
    m = length(gp)
    n = length(hp)
    get_gradient!(M, X, LG.co, p)
    (m > 0) && (
        X .+= sum(
            ((gp .* LG.ρ .+ LG.μ) .* get_grad_inequality_constraints(M, LG.co, p)) .*
            ((gp .+ LG.μ ./ LG.ρ) .> 0),
        )
    )
    (n > 0) &&
        (X .+= sum((hp .* LG.ρ .+ LG.λ) .* get_grad_equality_constraints(M, LG.co, p)))
    return X
end
# Allocating vector -> omit a few of the inequality gradient evaluations.
function (
    LG::AugmentedLagrangianGrad{
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
    LG::AugmentedLagrangianGrad{
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
