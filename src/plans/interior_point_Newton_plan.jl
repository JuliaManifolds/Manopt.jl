mutable struct NegativeReducedLagrangianGrad{T,R}
    cmo::ConstrainedManifoldObjective
    μ::T
    λ::T
    s::T
    barrier_param::R
end

function set_manopt_parameter!(nrlg::NegativeReducedLagrangianGrad, ::Val{:μ}, μ)
    nrlg.μ = μ
    return nrlg
end

function set_manopt_parameter!(nrlg::NegativeReducedLagrangianGrad, ::Val{:λ}, λ)
    nrlg.λ = λ
    return nrlg
end

function set_manopt_parameter!(nrlg::NegativeReducedLagrangianGrad, ::Val{:s}, s)
    nrlg.s = s
    return nrlg
end

function set_manopt_parameter!(
    nrlg::NegativeReducedLagrangianGrad, ::Val{:barrier_param}, barrier_param
)
    nrlg.barrier_param = barrier_param
    return nrlg
end

function (nrlg::NegativeReducedLagrangianGrad)(M::AbstractManifold, p)
    m, n = length(nrlg.μ), length(nrlg.λ)
    g = get_inequality_constraints(M, nrlg.cmo, p)
    Jg = get_grad_inequality_constraints(M, nrlg.cmo, p)
    Jh = get_grad_equality_constraints(M, nrlg.cmo, p)
    grad = -get_gradient(M, nrlg.cmo, p)
    (m > 0) && (grad -= Jg' * (nrlg.μ + (nrlg.μ .* g .+ nrlg.barrier_param) ./ nrlg.s))
    #(n > 0) && (grad = ArrayPartition(grad + Jh' * λ, Jh' * λ))
    return grad
end

mutable struct ReducedLagrangianHess{T}
    cmo::ConstrainedManifoldObjective
    μ::T
    λ::T
    s::T
end

function set_manopt_parameter!(rlh::ReducedLagrangianHess, ::Val{:μ}, μ)
    rlh.μ = μ
    return rlh
end

function set_manopt_parameter!(rlh::ReducedLagrangianHess, ::Val{:λ}, λ)
    rlh.λ = λ
    return rlh
end

function set_manopt_parameter!(rlh::ReducedLagrangianHess, ::Val{:s}, s)
    rlh.s = s
    return rlh
end

function (rlh::ReducedLagrangianHess)(M::AbstractManifold, p, X)
    m, n = length(rlh.μ), length(rlh.λ)
    Jg = get_grad_inequality_constraints(M, rlh.cmo, p)
    Jh = get_grad_equality_constraints(M, rlh.cmo, p)
    hess = get_hessian(M, rlh.cmo, p, X)
    (m > 0) && (hess += Jg' * Diagonal(rlh.μ ./ rlh.s) * Jg * X) # plus Hess g and Hess h
    #(n > 0) && (hess = ArrayPartition(hess, Jh * X))
    return hess
end

# calculates σ for a given state, ref Lai & Yoshise Section 8.1 first paragraph
# might add more calculation methods for σ
function calculate_σ(M::AbstractManifold, cmo::ConstrainedManifoldObjective, p, μ, λ, s)
    m, n = length(μ), length(λ)
    g = get_inequality_constraints(M, cmo, p)
    h = get_equality_constraints(M, cmo, p)
    dg = get_grad_inequality_constraints(M, cmo, p)
    dh = get_grad_equality_constraints(M, cmo, p)
    F = get_gradient(M, cmo, p)
    d = inner(M, p, F, F)
    (m > 0) && (d += inner(M, p, dg'μ, dg'μ) + norm(g + s)^2 + norm(μ .* s)^2)
    (n > 0) && (d += inner(M, p, dh'λ, dh'λ) + norm(h)^2)
    return min(0.5, d^(1 / 4))
end

function is_feasible(M, cmo, p)
    g = get_inequality_constraints(M, cmo, p)
    h = get_equality_constraints(M, cmo, p)
    return is_point(M, p) && all(g .<= 0) && all(h .== 0)
end
