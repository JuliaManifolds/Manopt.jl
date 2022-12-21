
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

abstract type AbstractHessianOptions <: AbstractGradientOptions end

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
    ApproxHessianFiniteDifference{E, P, T, G, RTR,, VTR, R <: Real}

A functor to approximate the Hessian by a finite difference of gradient evaluation.

Given a point `p` and a direction `X` and the gradient ``\operatorname{grad}F: \mathcal M \to T\mathcal M``
of a function ``F`` the Hessian is approximated as follows:
Let ``c`` be a stepsize, ``X∈ T_p\mathcal M`` a tangent vector and ``q = \operatorname{retr}_p(\frac{c}{\lVert X \rVert_p}X)``
be a step in direction ``X`` of length ``c`` following a retraction
Then we approximate the Hessian by the finite difference of the gradients, where ``\mathcal T_{\cdot\gets\cdot}`` is a vector transport.

```math
\operatorname{Hess}F(p)[X]
 ≈
\frac{\lVert X \rVert_p}{c}\Bigl( \mathcal T_{p\gets q}\bigr(\operatorname{grad}F(q)\bigl) - \operatorname{grad}F(p)\Bigl)
```

 # Fields

* `gradient!!` the gradient function (either allocating or mutating, see `evaluation` parameter)
* `step_length` a step length for the finite difference
* `retraction_method` - a retraction to use
* `vector_transport_method` a vector transport to use

## Internal temporary fields

* `grad_tmp` a temporary storage for the gradient at the current `p`
* `grad_dir_tmp` a temporary storage for the gradient at the current `p_dir`
* `p_dir::P` a temporary storage to the forward direction (i.e. ``q`` above)

# Constructor

    ApproximateFinniteDifference(M, p, grad_f; kwargs...)

## Keyword arguments

* `evaluation` (`AllocatingEvaluation()`) whether the gradient is given as an allocation function or an in-place (`MutatingEvaluation()`).
* `steplength` (``2^{-14}``) step length ``c`` to approximate the gradient evaluations
* `retraction_method` (`ExponentialRetraction()`) retraction ``\operatorname{retr}_p`` to use
* `vector_transport_method` (`ParallelTransport()`) vector transport ``\mathcal T_{\cdot\gets\cdot}`` to use.
"""
mutable struct ApproxHessianFiniteDifference{E,P,T,G,RTR,VTR,R<:Real}
    p_dir::P
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
    tangent_vector=zero_vector(M, p),
    steplength::R=2^-14,
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

function (f::ApproxHessianFiniteDifference{AllocatingEvaluation})(M, p, X)
    norm_X = norm(M, p, X)
    (norm_X ≈ zero(norm_X)) && return zero_vector(M, p)
    c = f.steplength / norm_X
    f.grad_tmp .= f.gradient!!(M, p)
    f.p_dir .= retract(M, p, c * X, f.retraction_method)
    f.grad_tmp_dir .= f.gradient!!(M, f.p_dir)
    f.grad_tmp_dir .= vector_transport_to(
        M, f.p_dir, f.grad_tmp_dir, p, f.vector_transport_method
    )
    return (1 / c) * (f.grad_tmp_dir - f.grad_tmp)
end
function (f::ApproxHessianFiniteDifference{InplaceEvaluation})(M, Y, p, X)
    norm_X = norm(M, p, X)
    (norm_X ≈ zero(norm_X)) && return zero_vector!(M, X, p)
    c = f.steplength / norm_X
    f.gradient!!(M, f.grad_tmp, p)
    retract!(M, f.p_dir, p, c * X, f.retraction_method)
    f.gradient!!(M, f.grad_tmp_dir, f.p_dir)
    vector_transport_to!(
        M, f.grad_tmp_dir, f.p_dir, f.grad_tmp_dir, p, f.vector_transport_method
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
    ApproxHessianSymmetricRankOne{E, P, G, T, B<:AbstractBasis{ℝ}, VTR, R<:Real}

A functor to approximate the Hessian by the symmetric rank one update.

# Fields

* `gradient!!` the gradient function (either allocating or mutating, see `evaluation` parameter).
* `ν` a small real number to ensure that the denominator in the update does not become too small and thus the method does not break down.
* `vector_transport_method` a vector transport to use.

## Internal temporary fields

* `p_tmp` a temporary storage the current point `p`.
* `grad_tmp` a temporary storage for the gradient at the current `p`.
* `matrix` a temporary storage for the matrix representation of the approximating operator.
* `basis` a temporary storage for an orthonormal basis at the current `p`.

# Constructor

    ApproxHessianSymmetricRankOne(M, p, gradF; kwargs...)

## Keyword arguments

* `initial_operator` (`Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M))`) the matrix representation of the initial approximating operator.
* `basis` (`DefaultOrthonormalBasis()`) an orthonormal basis in the tangent space of the initial iterate p.
* `nu` (`-1`)
* `evaluation` (`AllocatingEvaluation()`) whether the gradient is given as an allocation function or an in-place (`MutatingEvaluation()`).
* `vector_transport_method` (`ParallelTransport()`) vector transport ``\mathcal T_{\cdot\gets\cdot}`` to use.
"""
mutable struct ApproxHessianSymmetricRankOne{E,P,G,T,B<:AbstractBasis{ℝ},VTR,R<:Real}
    p_tmp::P
    gradient!!::G
    grad_tmp::T
    matrix::Matrix
    basis::B
    vector_transport_method::VTR
    ν::R
end
function ApproxHessianSymmetricRankOne(
    M::mT,
    p::P,
    gradient::G;
    initial_operator::AbstractMatrix=Matrix{Float64}(
        I, manifold_dimension(M), manifold_dimension(M)
    ),
    basis::B=DefaultOrthonormalBasis(),
    nu::R=-1.0,
    evaluation=AllocatingEvaluation(),
    vector_transport_method::VTR=ParallelTransport(),
) where {
    mT<:AbstractManifold,P,G,B<:AbstractBasis{ℝ},R<:Real,VTR<:AbstractVectorTransportMethod
}
    grad_tmp = gradient(M, p)
    return ApproxHessianSymmetricRankOne{typeof(evaluation),P,G,typeof(grad_tmp),B,VTR,R}(
        p, gradient, grad_tmp, initial_operator, basis, vector_transport_method, nu
    )
end

function (f::ApproxHessianSymmetricRankOne{AllocatingEvaluation})(M, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(f.p_tmp, p)
        f.grad_tmp = f.gradient!!(M, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    return get_vector(
        M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis
    )
end

function (f::ApproxHessianSymmetricRankOne{MutatingEvaluation})(M, Y, p, X)
    # Update Basis if necessary
    # if distance(M, p, f.p_tmp) >= eps(Float64)
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(f.p_tmp, p)
        f.grad_tmp = f.gradient!!(M, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    Y .= get_vector(M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis)

    return Y
end

function update_hessian!(M, f::ApproxHessianSymmetricRankOne, p, p_proposal, X)
    yk_c = get_coordinates(
        M,
        p,
        vector_transport_to(
            M, p_proposal, f.gradient!!(M, p_proposal), p, f.vector_transport_method
        ) - f.grad_tmp,
        f.basis,
    )
    sk_c = get_coordinates(M, p, X, f.basis)
    srvec = yk_c - f.matrix * sk_c
    if f.ν < 0 || abs(dot(srvec, sk_c)) >= f.ν * norm(srvec) * norm(sk_c)
        f.matrix = f.matrix + srvec * srvec' / (srvec' * sk_c)
    end
end

function update_hessian_basis!(M, f::ApproxHessianSymmetricRankOne, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    return f.grad_tmp = f.gradient!!(M, f.p_tmp)
end

function update_hessian!(M, f, p, p_proposal, X) end

function update_hessian_basis!(M, f, p) end

@doc raw"""
    ApproxHessianBFGS{E, P, G, T, B<:AbstractBasis{ℝ}, VTR, R<:Real}

A functor to approximate the Hessian by the BFGS update.

# Fields

* `gradient!!` the gradient function (either allocating or mutating, see `evaluation` parameter).
* `scale`
* `vector_transport_method` a vector transport to use.

## Internal temporary fields

* `p_tmp` a temporary storage the current point `p`.
* `grad_tmp` a temporary storage for the gradient at the current `p`.
* `matrix` a temporary storage for the matrix representation of the approximating operator.
* `basis` a temporary storage for an orthonormal basis at the current `p`.

# Constructor

    ApproxHessianBFGS(M, p, gradF; kwargs...)

## Keyword arguments

* `initial_operator` (`Matrix{Float64}(I, manifold_dimension(M), manifold_dimension(M))`) the matrix representation of the initial approximating operator.
* `basis` (`DefaultOrthonormalBasis()`) an orthonormal basis in the tangent space of the initial iterate p.
* `nu` (`-1`)
* `evaluation` (`AllocatingEvaluation()`) whether the gradient is given as an allocation function or an in-place (`MutatingEvaluation()`).
* `vector_transport_method` (`ParallelTransport()`) vector transport ``\mathcal T_{\cdot\gets\cdot}`` to use.
"""
mutable struct ApproxHessianBFGS{
    E,P,G,T,B<:AbstractBasis{ℝ},VTR<:AbstractVectorTransportMethod
}
    p_tmp::P
    gradient!!::G
    grad_tmp::T
    matrix::Matrix
    basis::B
    vector_transport_method::VTR
    scale::Bool
end
function ApproxHessianBFGS(
    M::mT,
    p::P,
    gradient::G;
    initial_operator::AbstractMatrix=Matrix{Float64}(
        I, manifold_dimension(M), manifold_dimension(M)
    ),
    basis::B=DefaultOrthonormalBasis(),
    scale::Bool=true,
    evaluation=AllocatingEvaluation(),
    vector_transport_method::VTR=ParallelTransport(),
) where {mT<:AbstractManifold,P,G,B<:AbstractBasis{ℝ},VTR<:AbstractVectorTransportMethod}
    grad_tmp = gradient(M, p)
    return ApproxHessianBFGS{typeof(evaluation),P,G,typeof(grad_tmp),B,VTR}(
        p, gradient, grad_tmp, initial_operator, basis, vector_transport_method, scale
    )
end

function (f::ApproxHessianBFGS{AllocatingEvaluation})(M, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(f.p_tmp, p)
        f.grad_tmp = f.gradient!!(M, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    return get_vector(
        M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis
    )
end

function (f::ApproxHessianBFGS{MutatingEvaluation})(M, Y, p, X)
    # Update Basis if necessary
    if p != f.p_tmp
        update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
        copyto!(f.p_tmp, p)
        f.grad_tmp = f.gradient!!(M, f.p_tmp)
    end

    # Apply Hessian approximation on vector
    Y .= get_vector(M, f.p_tmp, f.matrix * get_coordinates(M, f.p_tmp, X, f.basis), f.basis)

    return Y
end

function update_hessian!(M, f::ApproxHessianBFGS, p, p_proposal, X)
    yk_c = get_coordinates(
        M,
        p,
        vector_transport_to(
            M, p_proposal, f.gradient!!(M, p_proposal), p, f.vector_transport_method
        ) - f.grad_tmp,
        f.basis,
    )
    sk_c = get_coordinates(M, p, X, f.basis)
    skyk_c = dot(sk_c, yk_c)
    return f.matrix =
        f.matrix + yk_c * yk_c' / skyk_c -
        f.matrix * sk_c * sk_c' * f.matrix / dot(sk_c, f.matrix * sk_c)
end

function update_hessian_basis!(M, f::ApproxHessianBFGS, p)
    update_basis!(f.basis, M, f.p_tmp, p, f.vector_transport_method)
    copyto!(f.p_tmp, p)
    return f.grad_tmp = f.gradient!!(M, f.p_tmp)
end

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
