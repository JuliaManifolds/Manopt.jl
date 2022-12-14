
@doc raw"""
    ManifoldHessianObjective{T<:AbstractEvaluationType,C,G,H,Pre} <: AbstractManifoldGradientObjective{T}

specify a problem for hessian based algorithms.

# Fields

* `cost` : a function $F:\mathcal M→ℝ$ to minimize
* `gradient`     : the gradient $\operatorname{grad}F:\mathcal M
  → \mathcal T\mathcal M$ of the cost function $F$
* `hessian`      : the hessian $\operatorname{Hess}F(x)[⋅]: \mathcal T_{x} \mathcal M
  → \mathcal T_{x} \mathcal M$ of the cost function $F$
* `preconditioner`       : the symmetric, positive definite
    preconditioner (approximation of the inverse of the Hessian of $F$)

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
struct ManifoldHessianObjective{T<:AbstractEvaluationType,C,G,H,Pre} <:
       AbstractManifoldGradientObjective{T}
    cost::C
    gradient!!::G
    hessian!!::H
    preconditioner::Pre
    function ManifoldHessianObjective(
        cost::C,
        grad::G,
        hess::H,
        pre::Pre;
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    ) where {C,G,H,Pre}
        return new{typeof(evaluation),C,G,H,Pre}(M, cost, grad, hess, pre)
    end
end

@doc raw"""
    Y = get_hessian(mp::AbstractManoptProblem{T}, p, X)
    get_hessian!(mp::AbstractManoptProblem{T}, Y, p, X)

evaluate the Hessian of an [`AbstractManoptProblem`](@ref) `mp` at `p`
applied to a tangent vector `X`, i.e. compute ``\operatorname{Hess}f(q)[X]``,
which can also happen in-place of `Y`.
"""
function get_hessian(mp::AbstractManoptProblem, p, X)
    return get_hessian(get_manifold(mp), get_objective(mp), p, X)
end
function get_hessian!(mp::AbstractManoptProblem, Y, p, X)
    return get_hessian!(get_manifold(mp), Y, get_objective(mp), p, X)
end
function get_hessian(
    M::AbstractManifold, mho::ManifoldHessianObjective{AllocatingEvaluation}, p, X
)
    return mho.hessian!!(M, p, X)
end
function get_hessian(
    M::AbstractManifold, mho::ManifoldHessianObjective{InplaceEvaluation}, p, X
)
    Y = zero_vector(M, p)
    mho.hessian!!(M, Y, p, X)
    return Y
end
function get_hessian!(
    M::AbstractManifold, Y, mho::ManifoldHessianObjective{AllocatingEvaluation}, p, X
)
    copyto!(M, Y, mho.hessian!!(M, p, X))
    return Y
end
function get_hessian!(
    M::AbstractManifold, Y, mho::ManifoldHessianObjective{InplaceEvaluation}, p, X
)
    mho.hessian!!(M, Y, p, X)
    return Y
end

@doc raw"""
    get_preconditioner(mp::AbstractManoptProblem, p, X)

evaluate the symmetric, positive definite preconditioner (approximation of the
inverse of the Hessian of the cost function `f`) of a
[`HessianPrAbstractManoptProblemoblem`](@ref) `mp` at the point `p` applied to a
tangent vector `X`.
"""
function get_preconditioner(mp::HessianProblem, p, X)
    return get_preconditioner(get_manifold(mp), get_objective(mp), p, X)
end

@doc raw"""
    get_preconditioner(M::AbstractManifold, mho::ManifoldHessianObjective, p, X)

evaluate the symmetric, positive definite preconditioner (approximation of the
inverse of the Hessian of the cost function `F`) of a
[`ManifoldHessianObjective`](@ref) `mho` at the point `p` applied to a
tangent vector `X`.
"""
function get_preconditioner(M::AbstractManifold, mho::ManifoldHessianObjective, p, X)
    return mho.preconditioner(M, p, X)
end

@doc raw"""
    ApproxHessianFiniteDifference{T, mT, P, G}

A functor to approximate the Hessian by a finite difference of gradient evaluations

# Constructor

    ApproxHessianFiniteDifference(M, p, grad_f; kwargs...)

Initialize the approximate hessian to compute ``\operatorname{Hess}f`` based on the gradient
gradient `grad_f(M, p)` of a function ``f`` on `M`.

## Optional Keyword arguments

* `tangent_vector` – (`zero_vector(M,p)`) specify the tangent vector type to be used indernally
* `steplength` - (`2*1e-14`) default step size for the approximation
* `evaluation` - ([`AllocatingEvaluation`](@ref)`()`) specify whether the gradient is allocating or mutating.
* `retraction_method` – (`default_retraction_method(M)`) a `retraction(M, p, X)` to use in the approximation.
* `vector_transport_method` - (`default_vector_transport_method(M)`) a vector transport to use
"""
mutable struct ApproxHessianFiniteDifference{E,P,T,G,RTR,VTR,R<:Real}
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
    p::P,
    grad_f::G;
    tangent_vector=zero_vector(M, p)steplength::R = 4e-14,
    evaluation=AllocatingEvaluation(),
    retraction_method::RTR=default_retraction_method(M),
    vector_transport_method::VTR=default_vector_transport_method(M),
) where {
    mT<:AbstractManifold,
    P,
    G,
    R<:Real,
    RTR<:AbstractRetractionMethod,
    VTR<:AbstractVectorTransportMethod,
}
    X = copy(M, p, tangent_vector)
    Y = copy(M, p, tangent_vector)
    return ApproxHessianFiniteDifference{typeof(evaluation),P,typeof(X),G,RTR,VTR,R}(
        p, grad_f, X, Y, retraction_method, vector_transport_method, steplength
    )
end
function (f::ApproxHessianFiniteDifference{AllocatingEvaluation})(M, x, X)
    norm_X = norm(M, x, X)
    (norm_X ≈ zero(norm_X)) && return zero_vector(M, x)
    c = f.steplength / norm_X
    f.grad_tmp .= f.gradient!!(M, x)
    f.x_dir .= retract(M, x, c * X, f.retraction_method)
    f.grad_tmp_dir .= f.gradient!!(M, f.x_dir)
    f.grad_tmp_dir .= vector_transport_to(
        M, f.x_dir, f.grad_tmp_dir, x, f.vector_transport_method
    )
    return (1 / c) * (f.grad_tmp_dir - f.grad_tmp)
end
function (f::ApproxHessianFiniteDifference{InplaceEvaluation})(M, Y, x, X)
    norm_X = norm(M, x, X)
    (norm_X ≈ zero(norm_X)) && return zero_vector!(M, X, x)
    c = f.steplength / norm_X
    f.gradient!!(M, f.grad_tmp, x)
    retract!(M, f.x_dir, x, c * X, f.retraction_method)
    f.gradient!!(M, f.grad_tmp_dir, f.x_dir)
    vector_transport_to!(
        M, f.grad_tmp_dir, f.x_dir, f.grad_tmp_dir, x, f.vector_transport_method
    )
    Y .= (1 / c) .* (f.grad_tmp_dir .- f.grad_tmp)
    return Y
end

@doc raw"""
    AbstractHessianOSolverptions <: AbstractManoptSolverState

An [`AbstractManoptSolverState`](@ref) type to represent algorithms that employ the Hessian.
These options are assumed to have a field (`gradient`) to store the current gradient ``\operatorname{grad}f(x)``
"""
abstract type AbstractHessianSolverState <: AbstractGradientSolverState end

@doc raw"""
    StopIfResidualIsReducedByFactor <: StoppingCriterion

A functor for testing if the norm of residual at the current iterate is reduced
by a factor compared to the norm of the initial residual, i.e.
$\Vert r_k \Vert_x \leqq κ \Vert r_0 \Vert_x$.
In this case the algorithm reached linear convergence.

# Fields
* `κ` – the reduction factor
* `initialResidualNorm` - stores the norm of the residual at the initial vector
    ``η`` of the Steihaug-Toint tcg mehtod [`truncated_conjugate_gradient_descent`](@ref)
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
    reason::String
    StopIfResidualIsReducedByFactor(κ::Float64) = new(κ, "")
end
function (c::StopIfResidualIsReducedByFactor)(
    p::HessianProblem, s::TruncatedConjugateGradientState, i
)
    if norm(p.M, s.x, s.residual) <= s.initialResidualNorm * c.κ && i > 0
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
    $η$ of the Steihaug-Toint tcg mehtod [`truncated_conjugate_gradient_descent`](@ref)
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
    reason::String
    StopIfResidualIsReducedByPower(θ::Float64) = new(θ, "")
end
function (c::StopIfResidualIsReducedByPower)(
    p::HessianProblem, tcgs::TruncatedConjugateGradientState, i
)
    if norm(p.M, tcgs.x, tcgs.residual) <= tcgs.initialResidualNorm^(1 + c.θ) && i > 0
        c.reason = "The algorithm reached superlinear convergence (residual at least reduced by power 1 + θ=$(1+(c.θ))).\n"
        return true
    end
    return false
end
@doc raw"""
    update_stopping_criterion!(c::StopIfResidualIsReducedByPower, :ResidualPower, v)

Update the residual Power ``θ`` time period after which an algorithm shall stop.
"""
function update_stopping_criterion!(
    c::StopIfResidualIsReducedByPower, ::Val{:ResidualPower}, v
)
    c.θ = v
    return c
end
@doc raw"""
    StopWhenTrustRegionIsExceeded <: StoppingCriterion

A functor for testing if the norm of the next iterate in the  Steihaug-Toint tcg
mehtod is larger than the trust-region radius, i.e. $\Vert η_{k}^{*} \Vert_x
≧ trust_region_radius$. terminate the algorithm when the trust region has been left.

# Fields
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`get_reason`](@ref).
* `storage` – stores the necessary parameters `η, δ, residual` to check the
    criterion.

# Constructor

    StopWhenTrustRegionIsExceeded([a])

initialize the StopWhenTrustRegionIsExceeded functor to indicate to stop after
the norm of the next iterate is greater than the trust-region radius using the
[`StoreStateAction`](@ref) `a`, which is initialized to store
`:η, :δ, :residual` by default.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenTrustRegionIsExceeded <: StoppingCriterion
    reason::String
end
StopWhenTrustRegionIsExceeded() = StopWhenTrustRegionIsExceeded("")
function (c::StopWhenTrustRegionIsExceeded)(
    ::HessianProblem, s::TruncatedConjugateGradientState, i::Int
)
    if s.ηPη >= s.trust_region_radius^2 && i >= 0
        c.reason = "Trust-region radius violation (‖η‖² = $(s.ηPη)) >= $(s.trust_region_radius^2) = trust_region_radius²). \n"
        return true
    end
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

# Constructor

    StopWhenCurvatureIsNegative()

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenCurvatureIsNegative <: StoppingCriterion
    reason::String
end
StopWhenCurvatureIsNegative() = StopWhenCurvatureIsNegative("")
function (c::StopWhenCurvatureIsNegative)(
    ::HessianProblem, s::TruncatedConjugateGradientState, i::Int
)
    if s.δHδ <= 0 && i > 0
        c.reason = "Negative curvature. The model is not strictly convex (⟨δ,Hδ⟩_x = $(s.δHδ))) <= 0).\n"
        return true
    end
    return false
end

@doc raw"""
    StopWhenModelIncreased <: StoppingCriterion

A functor for testing if the curvature of the model value increased.

# Fields
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`get_reason`](@ref).

# Constructor

    StopWhenModelIncreased()

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenModelIncreased <: StoppingCriterion
    reason::String
end
StopWhenModelIncreased() = StopWhenModelIncreased("")
function (c::StopWhenModelIncreased)(
    ::HessianProblem, s::TruncatedConjugateGradientState, i::Int
)
    if i > 0 && (s.new_model_value > s.model_value)
        c.reason = "Model value increased from $(s.model_value) to $(s.new_model_value).\n"
        return true
    end
    return false
end
