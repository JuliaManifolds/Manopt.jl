export HessianProblem, HessianOptions
export TruncatedConjugateGradientOptions, TrustRegionsOptions
export approxHessianFD, getHessian
export stopIfResidualIsReducedByFactor, stopIfResidualIsReducedByPower, stopWhenCurvatureIsNegative, stopWhenTrustRegionIsExceeded

#
# Problem
#
@doc raw"""
    HessianProblem <: Problem

specify a problem for hessian based algorithms.

# Fields
* `M`            : a manifold $\mathcal M$
* `costFunction` : a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `gradient`     : the gradient $\nabla F\colon\mathcal M
  \to \mathcal T\mathcal M$ of the cost function $F$
* `hessian`      : the hessian $\operatorname{Hess}[F] (\cdot)_ {x} \colon \mathcal T_{x} \mathcal M
  \to \mathcal T_{x} \mathcal M$ of the cost function $F$
* `precon`       : the symmetric, positive deﬁnite
    preconditioner (approximation of the inverse of the Hessian of $F$)

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
@doc raw"""
    TruncatedConjugateGradientOptions <: HessianOptions

describe the Steihaug-Toint truncated conjugate-gradient method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` : a point, where the trust-region subproblem needs
    to be solved
* `stop` : a function s,r = @(o,iter,ξ,x,xnew) returning a stop
    indicator and a reason based on an iteration number, the gradient and the
    last and current iterates
* `η` : a tangent vector (called update vector), which solves the
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
mutable struct TruncatedConjugateGradientOptions{P,T} <: HessianOptions
    x::P
    stop::StoppingCriterion
    η::T
    δ::T
    Δ
    residual::T
    useRand::Bool
    function TruncatedConjugateGradientOptions(
        x::P,
        stop::StoppingCriterion,
        η::T,
        δ::T,
        Δ,
        residual::T,
        uR::Bool
        ) where {P, T}
        return new{typeof(x),typeof(η)}(x,stop,η,δ,Δ,residual,uR)
    end
end

@doc raw"""
    TrustRegionsOptions <: HessianOptions

describe the trust-regions solver, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` : a point as starting point
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
    x
    stop::StoppingCriterion
    Δ
    Δ_bar
    retraction::Function
    useRand::Bool
    ρ_prime
    ρ_regularization
    function TrustRegionsOptions(
        x,
        stop::StoppingCriterion,
        Δ,
        Δ_bar,
        retr::Function,
        useRand::Bool,
        ρ_prime,
        ρ_regularization
        )
        return new(x, stop, Δ, Δ_bar, retr, useRand, ρ_prime, ρ_regularization)
    end
end

@doc raw"""
    getHessian(p,x,ξ)

evaluate the Hessian of a [`HessianProblem`](@ref) `p` at the point `x`
applied to a tangent vector `ξ`.
"""
function getHessian(p::Pr,x, ξ; stepsize=2*10^(-14)) where {Pr <: HessianProblem}
    if ismissing(p.hessian)
        return approxHessianFD(p.M, x -> getGradient(p.M,x) ,x,ξ; stepsize=stepsize)
    else
        return p.hessian(p.M,x,ξ)
    end
end

@doc raw"""
    getGradient(p,x)

evaluate the gradient of a [`HessianProblem`](@ref)`p` at the
point `x`.
"""
getGradient(p::Pr,x) where {Pr <: HessianProblem} = p.gradient(p.M,x)
@doc raw"""
    getPreconditioner(p,x,ξ)

evaluate the symmetric, positive deﬁnite preconditioner (approximation of the
inverse of the Hessian of the cost function `F`) of a
[`HessianProblem`](@ref) `p` at the point `x`applied to a
tangent vector `ξ`.
"""
getPreconditioner(p::Pr, x, ξ) where {Pr <: HessianProblem} = p.precon(p.M, x, ξ)

@doc raw"""
    approxHessianFD(p,x,ξ,[stepsize=2.0^(-14)])

return an approximated solution of the Hessian of the cost function applied to
a tangent vector `ξ` by using a generic finite difference approximation
based on computations of the gradient.

Input
* `p` – a Manopt problem structure (already containing the manifold and enough
        information to compute the cost gradient)
* `x` – a point where the Hessian is ​​to be approximated
* `ξ` – a tangent vector on which the approximated Hessian is ​​to be applied

# Optional
* `stepsize` – the length of the step with which the method should work

# Output
* a tangent vector generated by applying the approximated Hessian to the
    tangent vector ξ
"""
function approxHessianFD(M::MT, x, gradFct::Function, ξ; stepsize::Float64=2.0^(-14)) where {MT <: Manifold}
    norm_xi = norm(M,x,ξ)
    if norm_xi < eps(Float64)
        return zero_tangent_vector(M, x)
    end
    c = stepsize / norm_xi
    grad = gradFct(x)
    x1 = retraction(M, x, ξ, c)
    grad1 = gradFct(x1)
    grad1 = parallelTransport(M, x1, x, grad1)
    return (grad1 - grad)/c
end

@doc raw"""
    stopIfResidualIsReducedByFactor <: StoppingCriterion

A functor for testing if the norm of residual at the current iterate is reduced
by a factor compared to the norm of the initial residual, i.e.
$\Vert r_k \Vert_x \leqq \kappa \Vert r_0 \Vert_x$.
In this case the algorithm reached linear convergence.

# Fields
* `κ` – the reduction factor
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

@doc raw"""
    stopIfResidualIsReducedByPower <: StoppingCriterion

A functor for testing if the norm of residual at the current iterate is reduced
by a power of 1+θ compared to the norm of the initial residual, i.e.
$\Vert r_k \Vert_x \leqq  \Vert r_0 \Vert_{x}^{1+\theta}$. In this case the
algorithm reached superlinear convergence.

# Fields
* `θ` – part of the reduction power
* `initialResidualNorm` - stores the norm of the residual at the initial vector
    $\eta$ of the Steihaug-Toint tcg mehtod [`truncatedConjugateGradient`](@ref)
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`getReason`](@ref).

# Constructor

    stopIfResidualIsReducedByPower(iRN, θ)

initialize the stopIfResidualIsReducedByFactor functor to indicate to stop after
the norm of the current residual is lesser than the norm of the initial residual
iRN to the power of 1+θ.

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

@doc raw"""
    stopWhenTrustRegionIsExceeded <: StoppingCriterion

A functor for testing if the norm of the next iterate in the  Steihaug-Toint tcg
mehtod is larger than the trust-region radius, i.e. $\Vert η_{k}^{*} \Vert_x
≧ Δ$. terminate the algorithm when the trust region has been left.

# Fields
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`getReason`](@ref).
* `storage` – stores the necessary parameters `η, δ, residual` to check the
    criterion.

# Constructor

    stopWhenTrustRegionIsExceeded([a])

initialize the stopWhenTrustRegionIsExceeded functor to indicate to stop after
the norm of the next iterate is greater than the trust-region radius using the
[`StoreOptionsAction`](@ref) `a`, which is initialized to store
`:η, :δ, :residual` by default.

# See also
[`truncatedConjugateGradient`](@ref), [`trustRegions`](@ref)
"""
mutable struct stopWhenTrustRegionIsExceeded <: StoppingCriterion
    reason::String
    storage::StoreOptionsAction
    stopWhenTrustRegionIsExceeded(a::StoreOptionsAction=StoreOptionsAction( (:η, :δ, :residual) )) = new("", a)
end
function (c::stopWhenTrustRegionIsExceeded)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if hasStorage(c.storage,:δ) && hasStorage(c.storage,:η) && hasStorage(c.storage,:residual)
        η = getStorage(c.storage,:η)
        δ = getStorage(c.storage,:δ)
        residual = getStorage(c.storage,:residual)
        a1 = inner(p.M, o.x, o.useRand ? getPreconditioner(p, o.x, residual) : residual, residual)
        a2 = inner(p.M, o.x, δ, getHessian(p, o.x, δ))
        a3 = inner(p.M, o.x, η, getPreconditioner(p, o.x, δ))
        a4 = inner(p.M, o.x, δ, getPreconditioner(p, o.x, δ))
        norm = inner(p.M, o.x, η, η) - 2*( a1 / a2 ) * a3 + (a1 / a2)^2 * a4
        if norm >= o.Δ^2 && i >= 0
            c.reason = "Trust-region radius violation (‖η‖² = $norm >= $(o.Δ^2) = Δ²). \n"
            c.storage(p,o,i)
            return true
        end
    end
    c.storage(p,o,i)
    return false
end

@doc raw"""
    stopWhenCurvatureIsNegative <: StoppingCriterion

A functor for testing if the curvature of the model is negative, i.e.
$\langle \delta_k, \operatorname{Hess}[F](\delta_k)\rangle_x \leqq 0$.
In this case, the model is not strictly convex, and the stepsize as computed
does not give a reduction of the model.

# Fields
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`getReason`](@ref).
* `storage` – stores the necessary parameter `δ` to check the
    criterion.

# Constructor

    stopWhenCurvatureIsNegative([a])

initialize the stopWhenCurvatureIsNegative functor to indicate to stop after
the inner product of the search direction and the hessian applied on the search
dircetion is less than zero using the [`StoreOptionsAction`](@ref) `a`, which
is initialized to just store `:δ` by default.

# See also
[`truncatedConjugateGradient`](@ref), [`trustRegions`](@ref)
"""
mutable struct stopWhenCurvatureIsNegative <: StoppingCriterion
    reason::String
    storage::StoreOptionsAction
    stopWhenCurvatureIsNegative(a::StoreOptionsAction=StoreOptionsAction( (:δ, ) )) = new("", a)
end
function (c::stopWhenCurvatureIsNegative)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if hasStorage(c.storage,:δ)
        δ = getStorage(c.storage,:δ)
        if inner(p.M, o.x, δ, getHessian(p, o.x, δ)) <= 0 && i > 0
            c.reason = "Negative curvature. The model is not strictly convex (⟨δ,Hδ⟩_x = $(inner(p.M, o.x, δ, getHessian(p, o.x, δ))) <= 0).\n"
            c.storage(p,o,i)
            return true
        end
    end
    c.storage(p,o,i)
    return false
end
