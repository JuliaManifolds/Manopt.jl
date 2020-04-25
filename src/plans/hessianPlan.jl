
@doc raw"""
    HessianProblem <: Problem

specify a problem for hessian based algorithms.

# Fields
* `M`            : a manifold $\mathcal M$
* `cost` : a function $F\colon\mathcal M\to\mathbb R$ to minimize
* `gradient`     : the gradient $\nabla F\colon\mathcal M
  \to \mathcal T\mathcal M$ of the cost function $F$
* `hessian`      : the hessian $\operatorname{Hess}[F] (\cdot)_ {x} \colon \mathcal T_{x} \mathcal M
  \to \mathcal T_{x} \mathcal M$ of the cost function $F$
* `precon`       : the symmetric, positive deﬁnite
    preconditioner (approximation of the inverse of the Hessian of $F$)

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
struct HessianProblem{mT <: Manifold} <: Problem
    M::mT
    cost::Function
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
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
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
[`trust_regions`](@ref)
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
function getHessian(p::HessianProblem, x, ξ; stepsize=2*10^(-14))
    if ismissing(p.hessian)
        return approxHessianFD(p.M, x -> get_gradient(p.M,x) ,x,ξ; stepsize=stepsize)
    else
        return p.hessian(p.M,x,ξ)
    end
end

@doc raw"""
    get_gradient(p,x)

evaluate the gradient of a [`HessianProblem`](@ref)`p` at the
point `x`.
"""
get_gradient(p::HessianProblem,x) = p.gradient(p.M,x)
@doc raw"""
    get_preconditioner(p,x,ξ)

evaluate the symmetric, positive deﬁnite preconditioner (approximation of the
inverse of the Hessian of the cost function `F`) of a
[`HessianProblem`](@ref) `p` at the point `x`applied to a
tangent vector `ξ`.
"""
get_preconditioner(p::Pr, x, ξ) where {Pr <: HessianProblem} = p.precon(p.M, x, ξ)

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
function approxHessianFD(M::MT, x, gradFct::Function, ξ; stepsize=2.0^(-14), transport=ParallelTransport()) where {MT <: Manifold}
    norm_xi = norm(M,x,ξ)
    if norm_xi < eps(Float64)
        return zero_tangent_vector(M, x)
    end
    c = stepsize / norm_xi
    grad = gradFct(x)
    x1 = exp(M, x, ξ, c)
    grad1 = gradFct(x1)
    grad1 = vector_transport_to(M, x1, grad1, x, transport)
    return (1/c)*(grad1 - grad)
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
    $\eta$ of the Steihaug-Toint tcg mehtod [`truncated_conjugate_gradient_descent`](@ref)
* `reason` – stores a reason of stopping if the stopping criterion has one be
  reached, see [`get_reason`](@ref).

# Constructor

    stopIfResidualIsReducedByFactor(iRN, κ)

initialize the stopIfResidualIsReducedByFactor functor to indicate to stop after
the norm of the current residual is lesser than the norm of the initial residual
iRN times κ.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
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
    $\eta$ of the Steihaug-Toint tcg mehtod [`truncated_conjugate_gradient_descent`](@ref)
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`get_reason`](@ref).

# Constructor

    stopIfResidualIsReducedByPower(iRN, θ)

initialize the stopIfResidualIsReducedByFactor functor to indicate to stop after
the norm of the current residual is lesser than the norm of the initial residual
iRN to the power of 1+θ.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
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
    StopWhenTrustRegionIsExceeded <: StoppingCriterion

A functor for testing if the norm of the next iterate in the  Steihaug-Toint tcg
mehtod is larger than the trust-region radius, i.e. $\Vert η_{k}^{*} \Vert_x
≧ Δ$. terminate the algorithm when the trust region has been left.

# Fields
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`get_reason`](@ref).
* `storage` – stores the necessary parameters `η, δ, residual` to check the
    criterion.

# Constructor

    StopWhenTrustRegionIsExceeded([a])

initialize the StopWhenTrustRegionIsExceeded functor to indicate to stop after
the norm of the next iterate is greater than the trust-region radius using the
[`StoreOptionsAction`](@ref) `a`, which is initialized to store
`:η, :δ, :residual` by default.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenTrustRegionIsExceeded <: StoppingCriterion
    reason::String
    storage::StoreOptionsAction
    StopWhenTrustRegionIsExceeded(a::StoreOptionsAction=StoreOptionsAction( (:η, :δ, :residual) )) = new("", a)
end
function (c::StopWhenTrustRegionIsExceeded)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if has_storage(c.storage,:δ) && has_storage(c.storage,:η) && has_storage(c.storage,:residual)
        η = get_storage(c.storage,:η)
        δ = get_storage(c.storage,:δ)
        residual = get_storage(c.storage,:residual)
        a1 = inner(p.M, o.x, o.useRand ? get_preconditioner(p, o.x, residual) : residual, residual)
        a2 = inner(p.M, o.x, δ, getHessian(p, o.x, δ))
        a3 = inner(p.M, o.x, η, get_preconditioner(p, o.x, δ))
        a4 = inner(p.M, o.x, δ, get_preconditioner(p, o.x, δ))
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
    StopWhenCurvatureIsNegative <: StoppingCriterion

A functor for testing if the curvature of the model is negative, i.e.
$\langle \delta_k, \operatorname{Hess}[F](\delta_k)\rangle_x \leqq 0$.
In this case, the model is not strictly convex, and the stepsize as computed
does not give a reduction of the model.

# Fields
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`get_reason`](@ref).
* `storage` – stores the necessary parameter `δ` to check the
    criterion.

# Constructor

    StopWhenCurvatureIsNegative([a])

initialize the StopWhenCurvatureIsNegative functor to indicate to stop after
the inner product of the search direction and the hessian applied on the search
dircetion is less than zero using the [`StoreOptionsAction`](@ref) `a`, which
is initialized to just store `:δ` by default.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenCurvatureIsNegative <: StoppingCriterion
    reason::String
    storage::StoreOptionsAction
    StopWhenCurvatureIsNegative(a::StoreOptionsAction=StoreOptionsAction( (:δ, ) )) = new("", a)
end
function (c::StopWhenCurvatureIsNegative)(p::P,o::O,i::Int) where {P <: HessianProblem, O <: TruncatedConjugateGradientOptions}
    if has_storage(c.storage,:δ)
        δ = get_storage(c.storage,:δ)
        if inner(p.M, o.x, δ, getHessian(p, o.x, δ)) <= 0 && i > 0
            c.reason = "Negative curvature. The model is not strictly convex (⟨δ,Hδ⟩_x = $(inner(p.M, o.x, δ, getHessian(p, o.x, δ))) <= 0).\n"
            c.storage(p,o,i)
            return true
        end
    end
    c.storage(p,o,i)
    return false
end
