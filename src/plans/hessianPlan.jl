
struct HessianProblem{mT <: Manifold} <: Problem
    M::mT where {mT <: Manifold}
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
    stop::stoppingCriterion
    η::T where {T <: TVector}
    Hη::T where {T <: TVector}
    mδ::T where {T <: TVector}
    Δ::Float64
    d_Pd::Float64
    e_Pd::Float64
    e_Pe::Float64
    residual::T where {T <: TVector}
    z::T where {T <: TVector}
    model_value::Float64
    useRand::Bool
    TruncatedConjugateGradientOptions(x::P,η::T,Hη::T,δ::T,Δ::Float64,d_Pd::Float64,e_Pd::Float64,e_Pe::Float64,residual::T,z::T,model_value::Float64,uR::Bool) where {P <: MPoint, T <: TVector} = new(x,η,Hη,δ,Δ,d_Pd,e_Pd,e_Pe,residual,z,model_value,uR)
end

struct TrustRegionOptions <: HessianOptions
    x::P where {P <: MPoint}
    stop::stoppingCriterion
    δ::Float64
    δ_bar::Float64
    δ0::Float64
    useRand::Bool
    ρ_prime::Float64
    ρ_regularization::Float64
    norm_grad::Float64
    TrustRegionOptions(x::P, stop::stoppingCriterion, δ::Float64, δ_bar::Float64,
    δ0::Float64, useRand::Bool, ρ_prime::Float64, ρ_regularization::Float64,
    norm_grad::Float64) where {P <: MPoint} = new(x,stop,δ,δ_bar,δ0,useRand,ρ_prime,ρ_regularization,norm_grad)
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
    x1 = retraction(p.M, x, ξ, c)
    grad1 = getGradient(p, x1)
    grad1 = parallelTransport(p.M, x1, x, grad1)
    return TVector((getValue(grad1)-getValue(grad))/c)
end
