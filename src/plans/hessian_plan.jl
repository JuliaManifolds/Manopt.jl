
@doc raw"""
    HessianProblem <: Problem

specify a problem for hessian based algorithms.

# Fields
* `M`            : a manifold $\mathcal M$
* `cost` : a function $F\colon\mathcal M→ℝ$ to minimize
* `gradient`     : the gradient $\operatorname{grad}F:\mathcal M
  → \mathcal T\mathcal M$ of the cost function $F$
* `hessian`      : the hessian $\operatorname{Hess}F(x)[\cdot]: \mathcal T_{x} \mathcal M
  → \mathcal T_{x} \mathcal M$ of the cost function $F$
* `precon`       : the symmetric, positive deﬁnite
    preconditioner (approximation of the inverse of the Hessian of $F$)

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
struct HessianProblem{T,mT<:Manifold,C,G,H,Pre} <: AbstractGradientProblem{T}
    M::mT
    cost::C
    gradient!!::G
    hessian!!::H
    precon::Pre
    function HessianProblem(
        M::mT,
        cost::C,
        grad::G,
        hess::H,
        pre::Pre;
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ) where {mT<:Manifold,C,G,H,Pre}
        return new{typeof(evaluation),mT,C,G,H,Pre}(M, cost, grad, hess, pre)
    end
end

@doc raw"""
    AbstractHessianOptions <: Options

An [`Options`](@ref) type to represent algorithms that employ the Hessian.
These options are assumed to have a field (`gradient`) to store the current gradient ``\operatorname{grad}f(x)``
"""
abstract type AbstractHessianOptions <: AbstractGradientOptions end

@doc raw"""
    TruncatedConjugateGradientOptions <: AbstractHessianOptions

describe the Steihaug-Toint truncated conjugate-gradient method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` : a point, where the trust-region subproblem needs
    to be solved
* `stop` : a function s,r = @(o,iter,ξ,x,xnew) returning a stop
    indicator and a reason based on an iteration number, the gradient and the
    last and current iterates
* `gradient` : the gradient at the current iterate
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
mutable struct TruncatedConjugateGradientOptions{P,T,R<:Real} <: AbstractHessianOptions
    x::P
    stop::StoppingCriterion
    η::T
    δ::T
    gradient::T
    Δ::R
    residual::T
    useRand::Bool
    function TruncatedConjugateGradientOptions(
        x::P, stop::StoppingCriterion, η::T, δ::T, grad::T, Δ, residual::T, uR::Bool
    ) where {P,T}
        return new{typeof(x),typeof(η),typeof(Δ)}(x, stop, η, δ, grad, Δ, residual, uR)
    end
end

@doc raw"""
    TrustRegionsOptions <: AbstractHessianOptions

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
mutable struct TrustRegionsOptions{
    P,TStop<:StoppingCriterion,TΔ,TΔ_bar,TRetr<:AbstractRetractionMethod,Tρ_prime,Tρ_reg
} <: AbstractHessianOptions
    x::P
    gradient::P
    stop::TStop
    Δ::TΔ
    Δ_bar::TΔ_bar
    retraction_method::TRetr
    useRand::Bool
    ρ_prime::Tρ_prime
    ρ_regularization::Tρ_reg
end

@doc raw"""
    get_hessian(p::HessianProblem{T}, q, X)
    get_hessian!(p::HessianProblem{T}, Y, q, X)

evaluate the Hessian of a [`HessianProblem`](@ref) `p` at the point `q`
applied to a tangent vector `X`, i.e. ``\operatorname{Hess}f(q)[X]``.

The evaluation is done in place of `Y` for the `!`-variant.
The `T=`[`AllocatingEvaluation`](@ref) problem might still allocate memory within.
When the non-mutating variant is called with a `T=`[`MutatingEvaluation`](@ref)
memory for the result is allocated.
"""
function get_hessian(p::HessianProblem{AllocatingEvaluation}, q, X)
    return p.hessian!!(q, X)
end
function get_hessian(p::HessianProblem{MutatingEvaluation}, q, X)
    Y = zero_tangent_vector(p.M, q)
    return p.hessian!!(Y, q, X)
end
function get_hessian!(p::HessianProblem{AllocatingEvaluation}, Y, q, X)
    return copyto!(Y, p.hessian!!(q, X))
end
function get_hessian!(p::HessianProblem{MutatingEvaluation}, Y, q, X)
    Y = zero_tangent_vector(p.M, q)
    return p.hessian!!(Y, q, X)
end

@doc raw"""
    get_preconditioner(p,x,ξ)

evaluate the symmetric, positive deﬁnite preconditioner (approximation of the
inverse of the Hessian of the cost function `F`) of a
[`HessianProblem`](@ref) `p` at the point `x`applied to a
tangent vector `ξ`.
"""
get_preconditioner(p::HessianProblem, x, ξ) = p.precon(p.M, x, ξ)

@doc raw"""
    approxHessianFiniteDifference{T, mT, P, G}

A functor to approximate the Hessian by a finite difference of gradient evaluations
"""
struct ApproxHessianFiniteDifference{E,mT<:Manifold,P,T,G,RTR,VTR,R<:Real}
    M::mT
    x_dir::P
    gradient!!::G
    grad_tmp::T
    grad_tmp_dir::T
    retraction_method::RTR
    vector_transport_method::VTR
    steplength::R
end
function ApproxHessianFiniteDifference(
    M::mT,
    x::P,
    grad::G;
    steplength::R=2^-14,
    evaluation=AllocatingEvaluation(),
    retraction_method::RTR=ExponentialRetraction(),
    vector_transport_method::VTR=ParallelTransport(),
) where {
    mT<:Manifold,
    P,
    G,
    R<:Real,
    RTR<:AbstractRetractionMethod,
    VTR<:AbstractVectorTransportMethod,
}
    X = zero_tangent_vector(M, x)
    Y = zero_tangent_vector(M, x)
    return ApproxHessianFiniteDifference{typeof(evaluation),mT,P,typeof(X),G,RTR,VTR,R}(
        M, x, grad, X, Y, retraction_method, vector_transport_method, steplength
    )
end
function (f::ApproxHessianFiniteDifference{AllocatingEvaluation})(x, X)
    X = zero_tangent_vector(f.M, x)
    norm_X = norm(f.M, x, X)
    (norm_X ≈ zero(norm_X)) && return zero_tangent_vector!(f.M, X, x)
    c = f.stepsize / norm_X
    f.grad_tmp = f.gradient!!(x)
    f.x_dir = retract(f.M, x, c * X, f.retraction_method)
    f.grad_tmp_dir = f.gradient!!(f.x_dir)
    f.grad_tmp_dir = vector_transport_to(
        f.M, f.x_dir, f.grad_tmp_dir, x, f.vector_transport_method
    )
    return (1 / c) * (f.grad_tmp_dir - f.grad_tmp)
end
function (f::ApproxHessianFiniteDifference{MutatingEvaluation})(Y, x, X)
    norm_X = norm(f.M, x, X)
    (norm_X ≈ zero(norm_X)) && return zero_tangent_vector!(f.M, X, x)
    c = f.stepsize / norm_X
    f.gradient!!(f.grad_tmp, x)
    retract!(f.M, f.x_dir, x, c * X, f.retraction_method)
    f.gradient!!(f.grad_tmp_dir, f.x_dir)
    vector_transport_to!(
        f.M, f.grad_tmp_dir, f.x_dir, f.grad_tmp_dir, x, vector_transport_method
    )
    Y .= (1 / c) .* (f.grad_tmp_dir .- f.grad_tmp)
    return Y
end

@doc raw"""
    StopIfResidualIsReducedByFactor <: StoppingCriterion

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

    StopIfResidualIsReducedByFactor(iRN, κ)

initialize the StopIfResidualIsReducedByFactor functor to indicate to stop after
the norm of the current residual is lesser than the norm of the initial residual
iRN times κ.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopIfResidualIsReducedByFactor <: StoppingCriterion
    κ::Float64
    initialResidualNorm::Float64
    reason::String
    StopIfResidualIsReducedByFactor(iRN::Float64, κ::Float64) = new(κ, iRN, "")
end
function (c::StopIfResidualIsReducedByFactor)(
    p::P, o::O, i::Int
) where {P<:HessianProblem,O<:TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= c.initialResidualNorm * c.κ && i > 0
        c.reason = "The algorithm reached linear convergence (residual at least reduced by κ=$(c.κ)).\n"
        return true
    end
    return false
end

@doc raw"""
    StopIfResidualIsReducedByPower <: StoppingCriterion

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

    StopIfResidualIsReducedByPower(iRN, θ)

initialize the StopIfResidualIsReducedByFactor functor to indicate to stop after
the norm of the current residual is lesser than the norm of the initial residual
iRN to the power of 1+θ.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopIfResidualIsReducedByPower <: StoppingCriterion
    θ::Float64
    initialResidualNorm::Float64
    reason::String
    StopIfResidualIsReducedByPower(iRN::Float64, θ::Float64) = new(θ, iRN, "")
end
function (c::StopIfResidualIsReducedByPower)(
    p::P, o::O, i::Int
) where {P<:HessianProblem,O<:TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= c.initialResidualNorm^(1 + c.θ) && i > 0
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
    function StopWhenTrustRegionIsExceeded(
        a::StoreOptionsAction=StoreOptionsAction((:η, :δ, :residual))
    )
        return new("", a)
    end
end
function (c::StopWhenTrustRegionIsExceeded)(
    p::P, o::O, i::Int
) where {P<:HessianProblem,O<:TruncatedConjugateGradientOptions}
    if has_storage(c.storage, :δ) &&
       has_storage(c.storage, :η) &&
       has_storage(c.storage, :residual)
        η = get_storage(c.storage, :η)
        δ = get_storage(c.storage, :δ)
        residual = get_storage(c.storage, :residual)
        a1 = inner(
            p.M, o.x, o.useRand ? get_preconditioner(p, o.x, residual) : residual, residual
        )
        a2 = inner(p.M, o.x, δ, get_hessian(p, o.x, δ))
        a3 = inner(p.M, o.x, η, get_preconditioner(p, o.x, δ))
        a4 = inner(p.M, o.x, δ, get_preconditioner(p, o.x, δ))
        norm = inner(p.M, o.x, η, η) - 2 * (a1 / a2) * a3 + (a1 / a2)^2 * a4
        if norm >= o.Δ^2 && i >= 0
            c.reason = "Trust-region radius violation (‖η‖² = $norm >= $(o.Δ^2) = Δ²). \n"
            c.storage(p, o, i)
            return true
        end
    end
    c.storage(p, o, i)
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
    function StopWhenCurvatureIsNegative(a::StoreOptionsAction=StoreOptionsAction((:δ,)))
        return new("", a)
    end
end
function (c::StopWhenCurvatureIsNegative)(
    p::P, o::O, i::Int
) where {P<:HessianProblem,O<:TruncatedConjugateGradientOptions}
    if has_storage(c.storage, :δ)
        δ = get_storage(c.storage, :δ)
        if inner(p.M, o.x, δ, get_hessian(p, o.x, δ)) <= 0 && i > 0
            c.reason = "Negative curvature. The model is not strictly convex (⟨δ,Hδ⟩_x = $(inner(p.M, o.x, δ, get_hessian(p, o.x, δ))) <= 0).\n"
            c.storage(p, o, i)
            return true
        end
    end
    c.storage(p, o, i)
    return false
end
