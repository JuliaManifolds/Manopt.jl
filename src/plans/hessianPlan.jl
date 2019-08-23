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
    HessianProblem{mT}(M::mT,cost::Function,grad::Function,hess::Union{Function,Missing},pre::Function) where {mT <: Manifold} = new(M,cost,grad,hess,pre)
end
HessianProblem(M::mT,cost::Function,grad::Function,hess::Union{Function,Missing},pre::Function) where {mT <: Manifold} = HessianProblem{mT}(M,cost,grad,hess,pre)

abstract type HessianOptions <: Options end
#
# Options
#
"""
    TruncatedConjugateGradientOptions <: HessianOptions

describe the truncated Conjugate Gradient algorithm with the Steihaug-Toint
method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` : a [`MPoint`](@ref), where the trust-region subproblem needs
    to be solved
* `stoppingCriterion` : a function s,r = @(o,iter,ξ,x,xnew) returning a stop
    indicator and a reason based on an iteration number, the gradient and the
    last and current iterates
* `η` : a [`TVector`](@ref) (called update vector), which solves the
    trust-region subproblem after successful calculation by the algorithm
* `δ` : search direction
* `Δ` : the trust-region radius
* `residual` : the gradient
* `useRand` : indicates if the trust-region solve and so the algorithm is to be
        initiated with a random tangent vector. If set to true, no
        preconditioner will be used. This option is set to true in some
        scenarios to escape saddle points, but is otherwise seldom activated.

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
    TruncatedConjugateGradientOptions(x::P,stop::StoppingCriterion,η::T,δ::T,Δ::Float64,residual::T,uR::Bool) where {P <: MPoint, T <: TVector} = new(x,stop,η,δ,Δ,residual,uR)
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
* `ρ_prime` : a lower bound of the performance ratio for the
        iterate that decides if the iteration will be accepted or not. If not,
        the trust-region radius will have been decreased. To ensure this,
        ρ_prime >= 0 must be strictly smaller than 1/4. If ρ_prime is negative,
        the algorithm is not guaranteed to produce monotonically decreasing
        cost values. It is strongly recommended to set ρ_prime > 0, to aid
        convergence.
* `ρ_regularization` :

# Constructor

    TrustRegionOptions()

construct a Trust Regions Option with the fields as above.

# See also
[`trustRegionsSolver`](@ref)
"""
struct TrustRegionOptions <: HessianOptions
    x::P where {P <: MPoint}
    stop::StoppingCriterion
    δ::Float64
    δ_bar::Float64
    useRand::Bool
    ρ_prime::Float64
    ρ_regularization::Float64
    TrustRegionOptions(x::P, stop::StoppingCriterion, δ::Float64, δ_bar::Float64,
    useRand::Bool, ρ_prime::Float64, ρ_regularization::Float64) where {P <: MPoint} = new(x,stop,δ,δ_bar,useRand,ρ_prime,ρ_regularization)
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
function (c::stopResidualReducedByFactor)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= c.initialResidualNorm*c.κ && i > 0
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
function (c::stopResidualReducedByPower)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= c.initialResidualNorm^(1+c.θ) && i > 0
        c.reason = "The algorithm reached superlinear convergence (residual at least reduced by power 1 + θ=$(c.θ)).\n"
        return true
    end
    return false
end

struct stopExceededTrustRegion <: StoppingCriterion
    reason::String
    stopExceededTrustRegion(tol::Float64) = new(tol,"")
end
function (c::stopExceededTrustRegion)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if dot(p.M, o.x, o.δ, getHessian(p, o.x, o.δ)) >= 0 && i > 0
        c.reason = "Exceeded trust region.\n"
        return true
    end
    return false
end

struct stopWhenCurvatureIsNegative <: StoppingCriterion
    reason::String
    stopNegativeCurvature() = new("")
end
function (c::stopWhenCurvatureIsNegative)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if dot(p.M, o.x, o.δ, getHessian(p, o.x, o.δ)) <= 0 && i > 0
        c.reason = "Negative curvature.\n"
        return true
    end
    return false
end
