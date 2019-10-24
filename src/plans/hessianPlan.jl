export HessianProblem, HessianOptions
export TruncatedConjugateGradientOptions, TrustRegionsOptions
export approxHessianFD
export stopIfResidualIsReducedByFactor, stopIfResidualIsReducedByPower, stopWhenCurvatureIsNegative, stopWhenTrustRegionIsExceeded

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
* `hessian`      : the hessian $\operatorname{Hess}[F] (\cdot)_ {x} \colon \mathcal T_{x} \mathcal M
  \to \mathcal T_{x} \mathcal M$ of the cost function $F$
* `precon`       : the preconditioner for the Hessian of the cost function $F$

# See also
[`truncatedConjugateGradient`](@ref), [`trustRegions`](@ref)
"""
struct HessianProblem{mT <: Manifold} <: Problem
    M::mT
    costFunction::Function
    gradient::Function
    hessian::Union{Function,Missing}
    precon::Function
    HessianProblem(M::mT,cost::Function,grad::Function,hess::Union{Function,Missing},pre::Function) where {mT <: Manifold} = new{mT}(M,cost,grad,hess,pre)
end

abstract type HessianOptions <: Options end
#
# Options
#
@doc doc"""
    TruncatedConjugateGradientOptions <: HessianOptions

describe the Steihaug-Toint truncated conjugate-gradient method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` : a [`MPoint`](@ref), where the trust-region subproblem needs
    to be solved
* `stop` : a function s,r = @(o,iter,ξ,x,xnew) returning a stop
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

    TruncatedConjugateGradientOptions(x, stop, eta, delta, Delta, res, uR)

construct a truncated conjugate-gradient Option with the fields as above.

# See also
[`truncatedConjugateGradient`](@ref), [`trustRegions`](@ref)
"""
mutable struct TruncatedConjugateGradientOptions <: HessianOptions
    x::P where {P <: MPoint}
    stop::StoppingCriterion
    η::T where {T <: TVector}
    δ::T where {T <: TVector}
    Δ::Float64
    residual::T where {T <: TVector}
    useRand::Bool
    TruncatedConjugateGradientOptions(x::P,stop::StoppingCriterion,η::T,δ::T,Δ::Float64,residual::T,uR::Bool) where {P <: MPoint, T <: TVector} = new(x,stop,η,δ,Δ,residual,uR)
end

@doc doc"""
    TrustRegionsOptions <: HessianOptions

describe the trust-regions solver, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` : a [`MPoint`](@ref) as starting point
* `stop` : a function s,r = @(o,iter) returning a stop
    indicator and a reason based on an iteration number and the gradient
* `Δ` : the (initial) trust-region radius
* `Δ_bar` : the maximum trust-region radius
* `useRand` : indicates if the trust-region solve is to be initiated with a
        random tangent vector. If set to true, no preconditioner will be
        used. This option is set to true in some scenarios to escape saddle
        points, but is otherwise seldom activated.
* `ρ_prime` : a lower bound of the performance ratio for the iterate that
        decides if the iteration will be accepted or not. If not, the
        trust-region radius will have been decreased. To ensure this,
        ρ'>= 0 must be strictly smaller than 1/4. If ρ' is negative,
        the algorithm is not guaranteed to produce monotonically decreasing
        cost values. It is strongly recommended to set ρ' > 0, to aid
        convergence.
* `ρ_regularization` : Close to convergence, evaluating the performance ratio ρ
        is numerically challenging. Meanwhile, close to convergence, the
        quadratic model should be a good fit and the steps should be
        accepted. Regularization lets ρ go to 1 as the model decrease and
        the actual decrease go to zero. Set this option to zero to disable
        regularization (not recommended). When this is not zero, it may happen
        that the iterates produced are not monotonically improving the cost
        when very close to convergence. This is because the corrected cost
        improvement could change sign if it is negative but very small.

# Constructor

    TrustRegionsOptions(x, stop, delta, delta_bar, uR, rho_prime, rho_reg)

construct a trust-regions Option with the fields as above.

# See also
[`trustRegions`](@ref)
"""
mutable struct TrustRegionsOptions <: HessianOptions
    x::P where {P <: MPoint}
    stop::StoppingCriterion
    Δ::Float64
    Δ_bar::Float64
    useRand::Bool
    ρ_prime::Float64
    ρ_regularization::Float64
    TrustRegionsOptions(x::P, stop::StoppingCriterion, δ::Float64, δ_bar::Float64,
    useRand::Bool, ρ_prime::Float64, ρ_regularization::Float64) where {P <: MPoint} = new(x,stop,δ,δ_bar,useRand,ρ_prime,ρ_regularization)
end

@doc doc"""
    getHessian(p,x,ξ)

evaluate the Hessian of a [`HessianProblem`](@ref)`p` at the [`MPoint`](@ref) `x`
times a [`TVector`](@ref) `ξ`.
"""
getHessian(p::Pr,x::P,ξ::V) where {Pr <: HessianProblem, P <: MPoint, V <: TVector} = ismissing(p.hessian) ? approxHessianFD(p,x,ξ) : p.hessian(p.M,x,ξ)
@doc doc"""
    getGradient(p,x)

evaluate the gradient of a [`HessianProblem`](@ref)`p` at the [`MPoint`](@ref) `x`.
"""
getGradient(p::Pr,x::P) where {Pr <: HessianProblem, P <: MPoint} = p.gradient(p.M,x)
@doc doc"""
    getPreconditioner(p,x,ξ)

evaluate a preconditioner of the Hessian of a [`HessianProblem`](@ref)`p` at the [`MPoint`](@ref) `x`
times a [`TVector`](@ref) `ξ`.
"""
getPreconditioner(p::Pr,x::P, ξ::V) where {Pr <: HessianProblem, P <: MPoint, V <: TVector} = p.precon(p.M, x, ξ)

@doc doc"""
    approxHessianFD(p,x,ξ,[stepsize=2.0^(-14)])

return an approximated solution of the Hessian of the cost function applied to
a [`TVector`](@ref) ξ by using a generic finite difference approximation
based on computations of the gradient.

Input
* `p` – a Manopt problem structure (already containing the manifold and enough
        information to compute the cost gradient)
* `x` – a [`MPoint`](@ref) where the Hessian is ​​to be approximated
* `ξ` – a [`TVector`](@ref) on which the approximated Hessian is ​​to be applied

# Optional
* `stepsize` – the length of the step with which the method should work

# Output
* a [`TVector`](@ref) generated by applying the approximated Hessian to the
    [`TVector`](@ref) ξ
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

@doc doc"""
    stopIfResidualIsReducedByFactor <: StoppingCriterion

A functor for testing if the norm of residual at the current iterate is reduced
by a factor compared to the norm of the initial residual, i.e.
$\Vert r_k \Vert \leqq \kappa \Vert r_0 \Vert$

# Fields
* `κ` – the factor
* `initialResidualNorm` - stores the norm of the residual at the initial vector
    $\eta$ of the Steihaug-Toint tcg mehtod [`truncatedConjugateGradient`](@ref)
* `reason` – stores a reason of stopping if the stopping criterion has one be
  reached, see [`getReason`](@ref).

# Constructor

    stopIfResidualIsReducedByFactor(iRN, κ)

initialize the stopIfResidualIsReducedByFactor functor to indicate to stop after
the norm of the current residual is lesser than the norm of the initial residual
iRN times κ.

# See also
[`truncatedConjugateGradient`](@ref), [`trustRegions`](@ref)
"""
mutable struct stopIfResidualIsReducedByFactor <: StoppingCriterion
    κ::Float64
    initialResidualNorm::Float64
    reason::String
    stopIfResidualIsReducedByFactor(iRN::Float64,κ::Float64) = new(κ,iRN,"")
end
function (c::stopIfResidualIsReducedByFactor)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= c.initialResidualNorm*c.κ && i > 0
        c.reason = "The algorithm reached linear convergence (residual at least reduced by κ=$(c.κ)).\n"
        return true
    end
    return false
end

@doc doc"""
    stopIfResidualIsReducedByPower <: StoppingCriterion

A functor for testing if the norm of residual at the current iterate is reduced
by a power compared to the norm of the initial residual, i.e.
$\Vert r_k \Vert \leqq  \Vert r_0 \Vert^{1+\theta}$

# Fields
* `κ` – stores the maximal iteration number where to stop at
* `initialResidualNorm` - stores the norm of the residual at the initial vector
$\eta$ of the Steihaug-Toint tcg mehtod [`truncatedConjugateGradient`](@ref)
* `reason` – stores a reason of stopping if the stopping criterion has one be
reached, see [`getReason`](@ref).

# Constructor

stopIfResidualIsReducedByPower(iRN, θ)

initialize the stopIfResidualIsReducedByFactor functor to indicate to stop after
the norm of the current residual is lesser than the norm of the initial residual
iRN times θ.

# See also
[`truncatedConjugateGradient`](@ref), [`trustRegions`](@ref)
"""
mutable struct stopIfResidualIsReducedByPower <: StoppingCriterion
    θ::Float64
    initialResidualNorm::Float64
    reason::String
    stopIfResidualIsReducedByPower(iRN::Float64,θ::Float64) = new(θ,iRN,"")
end
function (c::stopIfResidualIsReducedByPower)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= c.initialResidualNorm^(1+c.θ) && i > 0
        c.reason = "The algorithm reached superlinear convergence (residual at least reduced by power 1 + θ=$(1+(c.θ))).\n"
        return true
    end
    return false
end

@doc doc"""
    stopWhenTrustRegionIsExceeded <: StoppingCriterion

terminate the algorithm when the trust region has been left.
"""
mutable struct stopWhenTrustRegionIsExceeded <: StoppingCriterion
    reason::String
    stopWhenTrustRegionIsExceeded() = new("")
end
function (c::stopWhenTrustRegionIsExceeded)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    a1 = dot(p.M, o.x, o.useRand ? getPreconditioner(p, o.x, o.residual) : o.residual, o.residual)
    a2 = dot(p.M, o.x, o.δ, getHessian(p, o.x, o.δ))
    a3 = dot( p.M, o.x, o.η, getPreconditioner(p, o.x, o.δ))
    a4 = dot(p.M, o.x, o.δ, getPreconditioner(p, o.x, o.δ))
    if dot(p.M, o.x, o.η, o.η) - 2*( a1 / a2 ) * a3 + (a1 / a2)^2 * a4 >= o.Δ^2 && i > 0
        c.reason = "Exceeded trust region.\n"
        return true
    end
    return false
end

@doc doc"""
    stopWhenCurvatureIsNegative <: StoppingCriterion

terminate the algorithm when the curvature is negative. In this case, the model
is not strictly convex, and the stepsize as computed does not give a reduction
of the model.
"""
mutable struct stopWhenCurvatureIsNegative <: StoppingCriterion
    reason::String
    stopWhenCurvatureIsNegative() = new("")
end
function (c::stopWhenCurvatureIsNegative)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if dot(p.M, o.x, o.δ, getHessian(p, o.x, o.δ)) <= 0 && i > 0
        c.reason = "Negative curvature.\n"
        return true
    end
    return false
end
