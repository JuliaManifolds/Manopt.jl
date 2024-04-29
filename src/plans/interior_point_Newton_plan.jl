mutable struct ReducedLagrangianGrad{CO,T,R}
    co::CO
    μ::T
    λ::T
    s::T
    barrier_param::R
end

function set_manopt_parameter!(rlg::ReducedLagrangianGrad, ::Val{:μ}, μ)
    rlg.μ = μ
    return rlg
end
function set_manopt_parameter!(rlg::ReducedLagrangianGrad, ::Val{:λ}, λ)
    rlg.λ = λ
    return rlg
end
function set_manopt_parameter!(rlg::ReducedLagrangianGrad, ::Val{:s}, s)
    rlg.s = s
    return rlg
end
function set_manopt_parameter!(rlg::ReducedLagrangianGrad, ::Val{:b}, barrier_param)
    rlg.barrier_param = barrier_param
    return rlg
end

function (G::ReducedLagrangianGrad)(M::AbstractManifold, p)
   
    m, n = length(G.μ), length(G.λ)

    g = get_inequality_constraints(M, G.co, p)
    Jg = get_grad_inequality_constraints(M, G.co, p)
    Jh = get_grad_equality_constraints(M, G.co, p)

    grad = get_gradient(M, G.co, p)

    (m > 0) && (grad += Jg'*(G.μ + ((G.μ).*g .+ G.barrier_param)./(G.s)))
    (n > 0) && (grad = ArrayPartition(grad + Jh'*(G.λ), Jh'*(G.λ)))

    return grad

end

mutable struct ReducedLagrangianHess{CO,T}
    co::CO
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

function (H::ReducedLagrangianHess)(M::AbstractManifold, p, X)

    m, n = length(H.μ), length(H.λ)

    Jg = get_grad_inequality_constraints(M, H.co, p)
    Jh = get_grad_equality_constraints(M, H.co, p)

    hess = get_hessian(M, H.co, p, X)

    (m > 0) && (hess += Jg'*Diagonal((H.μ)./(H.s))*Jg*X) # plus Hess g and Hess h
    (n > 0) && (hess = ArrayPartition(hess, Jh*X))

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

    return min(0.5, d^(1/4))

end

function is_feasible(M, cmo, p)

    # evaluate constraint functions at p
    g = get_inequality_constraints(M, cmo, p)
    h = get_equality_constraints(M, cmo, p)

    # check feasibility
    return is_point(M, p) && all(g .<= 0) && all(h .== 0)

end

