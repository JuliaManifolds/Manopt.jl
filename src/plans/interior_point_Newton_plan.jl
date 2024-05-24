include("../solvers/interior_point_Newton.jl")

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

function (nrlg::NegativeReducedLagrangianGrad)(N::AbstractManifold, q)
    m, n = length(nrlg.μ), length(nrlg.λ)
    g = get_inequality_constraints(N[1], nrlg.cmo, q[N,1])
    Jg = get_grad_inequality_constraints(N[1], nrlg.cmo, q[N,1])
    Jh = get_grad_equality_constraints(N[1], nrlg.cmo, q[N,1])
    X = zero_vector(N, q)
    grad = get_gradient(N[1], nrlg.cmo, q[N,1])
    (m > 0) && (grad += Jg' * (nrlg.μ + (nrlg.μ .* g .+ nrlg.barrier_param) ./ nrlg.s))
    (n > 0) && (grad +=  Jh'*λ)
    copyto!(N[1], X[N,1], grad)
    (n > 0) && (copyto!(N[2], X[N,2], Jh'*λ))
    return -X
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

function (rlh::ReducedLagrangianHess)(N::AbstractManifold, q, Y)
    TqN = TangentSpace(N, q)
    m, n = length(rlh.μ), length(rlh.λ)
    Jg = get_grad_inequality_constraints(N[1], rlh.cmo, q[N,1])
    Jh = get_grad_equality_constraints(N[1], rlh.cmo, q[N,1])
    X = zero_vector(N, q)
    hess = get_hessian(N[1], rlh.cmo, q[N,1], Y[N,1])
    (m > 0) && (hess += Jg' * Diagonal(rlh.μ ./ rlh.s) * Jg * Y[N,1]) # plus Hess g and Hess h
    copyto!(N[1], X[N,1], hess)
    (n > 0) && (copyto!(N[2], X[N,2], Jh*Y[N,1]))
    return X
end

function MeritFunction(N::AbstractManifold, cmo::ConstrainedManifoldObjective, q)
    p, μ, λ, s = q.x
    m, n = length(μ), length(λ)
    g = get_inequality_constraints(N[1], cmo, p)
    h = get_equality_constraints(N[1], cmo, p)
    dg = get_grad_inequality_constraints(N[1], cmo, p)
    dh = get_grad_equality_constraints(N[1], cmo, p)
    F = get_gradient(N[1], cmo, p)
    (m > 0) && (F += dg'μ)
    (n > 0) && (F += dh'λ)
    d = inner(N[1], p, F, F)
    (m > 0) && (d += norm(g + s)^2 + norm(μ .* s)^2)
    (n > 0) && (d += norm(h)^2)
    return d
end

function GradMeritFunction(N::AbstractManifold, cmo::ConstrainedManifoldObjective, q)
    p, μ, λ, s = q.x
    m, n = length(μ), length(λ)
    g = get_inequality_constraints(N[1], cmo, p)
    h = get_equality_constraints(N[1], cmo, p)
    dg = get_grad_inequality_constraints(N[1], cmo, p)
    dh = get_grad_equality_constraints(N[1], cmo, p)
    grad = get_gradient(N[1], cmo, p)
    X = zero_vector(N, q)
    (m > 0) && (grad += dg'μ)
    (n > 0) && (grad += dh'λ)
    copyto!(N[1], X[N,1], get_hessian(N[1], cmo, p, grad))
    (m > 0) && copyto!(N[2], X[N,2], dg*grad + μ .* s)
    (n > 0) && copyto!(N[3], X[N,3], dh*grad)
    (m > 0) && copyto!(N[4], X[N,4], s .* (g+s) + μ .* μ .* s)
    return 2*X
end

function is_feasible(M, cmo, p)
    g = get_inequality_constraints(M, cmo, p)
    h = get_equality_constraints(M, cmo, p)
    return is_point(M, p) && all(g .<= 0) && all(h .== 0)
end