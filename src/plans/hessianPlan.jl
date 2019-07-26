export HessianProblem, HessianOptions
export TruncatedConjugateGradientOptions, TrustRegionOptions
export stopResidualReducedByFactor, stopResidualReducedByPower

#
# Problem
#
@doc doc"""
    HessianProblem <: Problem
specify a problem for hessian based algorithms.

# Fields
* `M`            : a manifold $\mathcal M$
* `costFunction` : a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `gradient`     : the gradient $\nabla F\colon\mathcal M
  \to \mathcal T\mathcal M$ of the cost function $F$
* `hessian`      : the hessian matrix $Hess F(x) \colon \mathcal T_{x} \mathcal M
  \to \mathcal T_{x} \mathcal M$ of the cost function $F$
* `precon`       : the preconditioner for the Hessian of the cost function $F$

# See also
[`truncatedConjugateGradient`](@ref)
[`trustRegionsSolver`](@ref)

# """
struct HessianProblem{mT <: Manifold} <: Problem
    M::mT where {mT <: Manifold}
    costFunction::Function
    gradient::Function
    hessian::Union{Function,Missing}
    precon::Function
    HessianProblem{mT}(M::mT,cost::Function,grad::Function,hess::Union{Function,Missing},pre::Function= h -> h) where {mT <: Manifold} = new(M,cost,grad,hess,pre)
end
HessianProblem(M::mT,cost::Function,grad::Function,hess::Union{Function,Missing},pre::Function= h -> h) where {mT <: Manifold} = HessianProblem{mT}(M,cost,grad,hess,pre)

abstract type HessianOptions <: Options end
#
# Options
#
"""
    TruncatedConjugateGradientOptions <: HessianOptions

Describes the truncated Conjugate Gradient algorithm with the Steihaug-Toint
method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` : an [`MPoint`](@ref) as starting point
* `stoppingCriterion` : a function s,r = @(o,iter,ξ,x,xnew) returning a stop
    indicator and a reason based on an iteration number, the gradient and the last and
    current iterates
* `η` :
* `δ` :
* `Δ` :
* `residual` :
* `useRand` :

# Constructor

    TruncatedConjugateGradientOptions()

construct a truncated Conjugate Gradient Option with the fields as above.

# See also
[`truncatedConjugateGradient`](@ref), [`trustRegionsSolver`](@ref)
"""
struct TruncatedConjugateGradientOptions <: HessianOptions
    x::P where {P <: MPoint}
    stop::StoppingCriterion
    η::T where {T <: TVector}
    δ::T where {T <: TVector}
    Δ::Float64
    residual::T where {T <: TVector}
    useRand::Bool
    TruncatedConjugateGradientOptions(x::P,η::T,δ::T,Δ::Float64,residual::T,uR::Bool) where {P <: MPoint, T <: TVector} = new(x,η,δ,Δ,e_Pe,residual,uR)
end

"""
    TrustRegionOptions <: HessianOptions

Describes the Trust Regions Solver, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` : an [`MPoint`](@ref) as starting point
* `stop` : a function s,r = @(o,iter) returning a stop
    indicator and a reason based on an iteration number and the gradient
* `δ` : the (initial) trust-region radius
* `δ_bar` : the maximum trust-region radius
* `useRand` : indicates if the trust-region solve is to be initiated with a
        random tangent vector. If set to true, no preconditioner will be
        used. This option is set to true in some scenarios to escape saddle
        points, but is otherwise seldom activated.
* `ρ_prime` :
* `ρ_regularization` :

# Constructor

    TrustRegionOptions()

construct a Trust Regions Option with the fields as above.

# See also
[`trustRegionsSolver`](@ref)
"""
struct TrustRegionOptions <: HessianOptions
    x::P where {P <: MPoint}
    stop::stoppingCriterion
    δ::Float64
    δ_bar::Float64
    useRand::Bool
    ρ_prime::Float64
    ρ_regularization::Float64
    TrustRegionOptions(x::P, stop::stoppingCriterion, δ::Float64, δ_bar::Float64,
    useRand::Bool, ρ_prime::Float64, ρ_regularization::Float64,
    norm_grad::Float64) where {P <: MPoint} = new(x,stop,δ,δ_bar,useRand,ρ_prime,ρ_regularization,norm_grad)
end

"""
    getHessian(p,x,ξ)

evaluate the Hessian of a [`HessianProblem`](@ref)`p` at the [`MPoint`](@ref) `x`
times a [`TVector`](@ref) `ξ`.
"""
getHessian(p::Pr,x::P,ξ::V) where {Pr <: HessianProblem, P <: MPoint, V <: TVector} = ismissing(p.hessian) ? approxHessianFD(p,x,ξ) : p.hessian(x,ξ)
"""
    getGradient(p,x)

evaluate the gradient of a [`HessianProblem`](@ref)`p` at the [`MPoint`](@ref) `x`.
"""
getGradient(p::Pr,x::P) where {Pr <: HessianProblem, P <: MPoint} = p.gradient(x)
"""
    getCost(p,x)

evaluate the cost function of a [`HessianProblem`](@ref)`p` at the [`MPoint`](@ref) `x`.
"""
getCost(p::Pr,x::P) where {Pr <: Problem, P <: MPoint} = p.costFunction(x)
"""
    getPreconditioner(p,x,ξ)

evaluate a preconditioner of the Hessian of a [`HessianProblem`](@ref)`p` at the [`MPoint`](@ref) `x`
times a [`TVector`](@ref) `ξ`.
"""
getPreconditioner(p::Pr,x::P, ξ::V) where {Pr <: Problem, P <: MPoint, V <: TVector} = p.precon(x,ξ)

@doc doc"""
    approxHessianFD(p,x,ξ,[stepsize=2.0^(-14)])
"""
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

struct stopResidualReducedByFactor <: StoppingCriterion
    κ::Float64
    initialResidualNorm::Float64
    reason::String
    stopResidualReducedByFactor(iRN::Float64,κ::Float64) = new(κ,iRN,"")
end
function (c::stopResidualReducedByFactor)(p::P,o::O,i::Int) where {P <: Problem, O <: TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= c.initialResidualNorm*c.κ
        c.reason = "The algorithm reached linear convergence (residual at least reduced by κ=$(c.κ)).\n"
        return true
    end
    return false
end

struct stopResidualReducedByPower <: StoppingCriterion
    θ::Float64
    initialResidualNorm::Float64
    reason::String
    stopResidualReducedByPower(iRN::Float64,θ::Float64) = new(θ,iRN,"")
end
function (c::stopResidualReducedByPower)(p::P,o::O,i::Int) where {P <: Problem, O <: TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= c.initialResidualNorm^(1+c.θ)
        c.reason = "The algorithm reached superlinear convergence (residual at least reduced by power 1 + θ=$(c.θ)).\n"
        return true
    end
    return false
end

struct stopGradientTolerance <: StoppingCriterion
    tolgrad::Float64
    reason::String
    stopGradientTolerance(tol::Float64) = new(tol,"")
end
function (c::stopGradientTolerance)(p::P,o::O,i::Int) where {P <: Problem, O <: TrustRegionOptions}
    if norm(p.M, o.x, getGradient(p, o.x)) <= c.tolgrad
        c.reason = "The norm of the gradient reached tolerance ($norm(p.M, o.x, getGradient(p, o.x))).\n"
        return true
    end
    return false
end

struct stopExceededTrustRegion <: StoppingCriterion
    reason::String
    stopExceededTrustRegion(tol::Float64) = new(tol,"")
end
function (c::stopExceededTrustRegion)(p::P,o::O,i::Int) where {P <: Problem, O <: TruncatedConjugateGradientOptions}
    if dot(p.M, o.x, o.δ, getHessian(p, o.x, o.δ)) >= 0
        c.reason = "Exceeded trust region.\n"
        return true
    end
    return false
end

struct stopWhen CurvatureIsNegative <: StoppingCriterion
    reason::String
    stopNegativeCurvature() = new("")
end
function (c::stopNegativeCurvature)(p::P,o::O,i::Int) where {P <: Problem, O <: TruncatedConjugateGradientOptions}
    if dot(p.M, o.x, o.δ, getHessian(p, o.x, o.δ)) <= 0
        c.reason = "Negative curvature.\n"
        return true
    end
    return false
end
