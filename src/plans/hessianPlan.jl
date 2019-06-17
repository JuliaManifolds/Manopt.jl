
struct HessianProblem <: Problem
    M::mT where mT <: Manifold
    costFunction::Function
    gradient::Function
    hessian::Union{Function,Missing}
    precon::Function
    HessianProblem(M::mT,cost::Function,grad::Function,hess::Union{Function,Missing},pre::Function)=new(M,cost,grad,hess,pre)
end

getManifold(problem::HessianProblem) = problem.M;
getCostFunction(problem::HessianProblem) = problem.costFunction;
getGradient(problem::HessianProblem) = problem.gradient;
getHessian(problem::HessianProblem) = problem.hessian;
getPreconditioner(problem::HessianProblem) = problem.precon;

# injectivity_radius.

abstract type HessianOptions <: Options end

struct TruncatedConjugateGradientOptions <: HessianOptions
    x::P where {P <: MPoint}
    kappa::Float64
    theta::Float64
    useRand::Bool
    mininner::Int64
    maxinner::Int64
end

struct TrustRegionOptions <: HessianOptions
    # x::P where {P <: MPoint}
    # is actually passed as a parameter of the function
    stop::stoppingCriterion
    # do we pass the limits to the criterion or do we pass them the options?
    tolgradnorm::Float64
    maxiter::Int64
    miniter::Int64
    maxtime::Float64
    mininner::Int64
    maxinner::Int64
    Delta_bar::Float64
    Delta0::Float64
    useRand::Bool
    kappa::Float64
    theta::Float64
    rho_prime::Float64
    rho_regularization::Float64
end

getHessian(p::Pr,x::P,ξ::V) where {Pr <: Problem, P <: MPoint, V <: MTVector} = p.hessian(x,ξ)
getGradient(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} = p.gradient(x)
getPreconditioner(p::Pr,x::P, ξ::V) where {Pr <: Problem, P <: MPoint, V <: MTVector} = p.precon(x,ξ)

function approxHessianFD(problem::HessianProblem, x::MPoint, ξ::MTVec, stepsize::Float64=2.0^(-14))
    M = getManifold(problem)
    norm_xi = norm(M,x,ξ)

    if norm_xi < eps(Float64)
        return zeroTVector(M, x)
    end

    c = stepsize / norm_xi

    grad = getGradient(problem, x)

    x1 = retraction(M, x, ξ, c)
    grad1 = getGradient(problem, x1)

    grad1 = parallelTransport(M, x1, x, grad1)

    return TVector((getValue(grad1)-getValue(grad))/c)
end
