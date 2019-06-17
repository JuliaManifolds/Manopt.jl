
struct HessianProblem{mT <: Manifold} <: Problem
    M::mT
    costFunction::Function
    gradient::Function
    hessian::Union{Function,Missing}
    precon::Function
    HessianProblem(M::mT,cost::Function,grad::Function,hess::Union{Function,Missing},pre::Function) where {mT <: Manifold} = new(M,cost,grad,hess,pre)
end

# injectivity_radius.

abstract type HessianOptions <: Options end

struct TruncatedConjugateGradientOptions <: HessianOptions
    x::P where {P <: MPoint}
    kappa::Float64
    theta::Float64
    useRand::Bool
end

struct TrustRegionOptions <: HessianOptions
    x::P where {P <: MPoint}
    ∇::T where {T <: TVector}
    # is actually passed as a parameter of the function
    stop::stoppingCriterion
    # do we pass the limits to the criterion or do we pass them the options?
    δ_bar::Float64
    δ0::Float64
    useRand::Bool
    kappa::Float64
    theta::Float64
    ρ_prime::Float64
    ρ_regularization::Float64
end

getHessian(p::Pr,x::P,ξ::V) where {Pr <: HessianProblem, P <: MPoint, V <: TVector} = ismissing(p.hessian) ? approxHessianFD(p,x,ξ) : p.hessian(x,ξ)
getGradient(p::Pr,x::P) where {Pr <: HessianProblem, P <: MPoint} = p.gradient(x)
getCost(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} = p.costFunction(x)
getPreconditioner(p::Pr,x::P, ξ::V) where {Pr <: Problem, P <: MPoint, V <: TVector} = p.precon(x,ξ)

function approxHessianFD(p::HessianProblem, x::P, ξ::T, stepsize::Float64=2.0^(-14)) where {P <: MPoint, T <: TVector}
    norm_xi = norm(p.M,x,ξ)
    if norm_xi < eps(Float64)
        return zeroTVector(p.M, x)
    end
    c = stepsize / norm_xi
    grad = getGradient(p, x)
    x1 = retraction(M, x, ξ, c)
    grad1 = getGradient(p, x1)
    grad1 = parallelTransport(p.M, x1, x, grad1)
    return TVector((getValue(grad1)-getValue(grad))/c)
end
