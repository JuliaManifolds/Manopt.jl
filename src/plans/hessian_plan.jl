
@doc raw"""
    HessianProblem <: Problem

specify a problem for hessian based algorithms.

# Fields
* `M`            : a manifold $\mathcal M$
* `cost` : a function $F:\mathcal M→ℝ$ to minimize
* `gradient`     : the gradient $\operatorname{grad}F:\mathcal M
  → \mathcal T\mathcal M$ of the cost function $F$
* `hessian`      : the hessian $\operatorname{Hess}F(x)[⋅]: \mathcal T_{x} \mathcal M
  → \mathcal T_{x} \mathcal M$ of the cost function $F$
* `precon`       : the symmetric, positive definite
    preconditioner (approximation of the inverse of the Hessian of $F$)

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
struct HessianProblem{T,mT<:AbstractManifold,C,G,H,Pre} <: AbstractGradientProblem{T}
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
    ) where {mT<:AbstractManifold,C,G,H,Pre}
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
* `trust_region_radius` : the trust-region radius
* `residual` : the gradient
* `randomize` : indicates if the trust-region solve and so the algorithm is to be
        initiated with a random tangent vector. If set to true, no
        preconditioner will be used. This option is set to true in some
        scenarios to escape saddle points, but is otherwise seldom activated.
* `project_vector!` : (`copyto!`) specify a projection operation for tangent vectors
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.

# Constructor

    TruncatedConjugateGradientOptions(x, stop, eta, delta, Delta, res, uR)

construct a truncated conjugate-gradient Option with the fields as above.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct TruncatedConjugateGradientOptions{P,T,R<:Real,SC<:StoppingCriterion,Proj} <:
               AbstractHessianOptions
    x::P
    stop::SC
    gradient::T
    η::T
    Hη::T
    δ::T
    Hδ::T
    δHδ::R
    ηPδ::R
    δPδ::R
    ηPη::R
    z::T
    z_r::R
    residual::T
    trust_region_radius::R
    model_value::R
    new_model_value::R
    κ::R
    randomize::Bool
    project_vector!::Proj
    initialResidualNorm::Float64
    function TruncatedConjugateGradientOptions(
        p::HessianProblem,
        x::P,
        η::T,
        trust_region_radius::R,
        randomize::Bool;
        project_vector!::Proj=copyto!,
        θ::Float64=1.0,
        κ::Float64=0.1,
        stop::StoppingCriterion=StopWhenAny(
            StopAfterIteration(manifold_dimension(p.M)),
            StopWhenAll(
                StopIfResidualIsReducedByPower(θ), StopIfResidualIsReducedByFactor(κ)
            ),
            StopWhenTrustRegionIsExceeded(),
            StopWhenCurvatureIsNegative(),
            StopWhenModelIncreased(),
        ),
    ) where {H,G,P,T,R<:Real,Proj}
        o = new{
            typeof(x),
            typeof(η),
            typeof(trust_region_radius),
            typeof(stop),
            typeof(project_vector!),
        }()
        o.x = x
        o.stop = stop
        o.η = η
        o.trust_region_radius = trust_region_radius
        o.randomize = randomize
        o.project_vector! = project_vector!
        o.model_value = zero(trust_region_radius)
        o.κ = zero(trust_region_radius)
        return o
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
* `trust_region_radius` : the (initial) trust-region radius
* `max_trust_region_radius` : the maximum trust-region radius
* `randomize` : indicates if the trust-region solve is to be initiated with a
        random tangent vector. If set to true, no preconditioner will be
        used. This option is set to true in some scenarios to escape saddle
        points, but is otherwise seldom activated.
* `project_vector!` : (`copyto!`) specify a projection operation for tangent vectors
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.
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
    P,T,SC<:StoppingCriterion,RTR<:AbstractRetractionMethod,R<:Real,Proj
} <: AbstractHessianOptions
    x::P
    gradient::T
    stop::SC
    trust_region_radius::R
    max_trust_region_radius::R
    retraction_method::RTR
    randomize::Bool
    project_vector!::Proj
    ρ_prime::R
    ρ_regularization::R

    tcg_options::TruncatedConjugateGradientOptions{P,T,R}

    x_proposal::P
    f_proposal::R

    # Random
    Hgrad::T
    η::T
    Hη::T
    η_Cauchy::T
    Hη_Cauchy::T
    τ::R
    function TrustRegionsOptions{P,T,SC,RTR,R,Proj}(
        x::P,
        grad::T,
        trust_region_radius::R,
        max_trust_region_radius::R,
        ρ_prime::R,
        ρ_regularization::R,
        randomize::Bool,
        stopping_citerion::SC,
        retraction_method::RTR;
        project_vector!::Proj=copyto!,
    ) where {P,T,SC<:StoppingCriterion,RTR<:AbstractRetractionMethod,R<:Real,Proj}
        o = new{P,T,SC,RTR,R,Proj}()
        o.x = x
        o.gradient = grad
        o.stop = stopping_citerion
        o.retraction_method = retraction_method
        o.trust_region_radius = trust_region_radius
        o.max_trust_region_radius = max_trust_region_radius::R
        o.ρ_prime = ρ_prime
        o.ρ_regularization = ρ_regularization
        o.randomize = randomize
        o.project_vector! = project_vector!
        return o
    end
end
function TrustRegionsOptions(
    x::P,
    grad::T,
    trust_region_radius::R,
    max_trust_region_radius::R,
    ρ_prime::R,
    ρ_regularization::R,
    randomize::Bool,
    stopping_citerion::SC;
    retraction_method::RTR=ExponentialRetraction(),
    project_vector!::Proj=copyto!,
) where {P,T,R<:Real,SC<:StoppingCriterion,RTR<:AbstractRetractionMethod,Proj}
    return TrustRegionsOptions{P,T,SC,RTR,R,Proj}(
        x,
        grad,
        trust_region_radius,
        max_trust_region_radius,
        ρ_prime,
        ρ_regularization,
        randomize,
        stopping_citerion,
        retraction_method;
        (project_vector!)=project_vector!,
    )
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
    return p.hessian!!(p.M, q, X)
end
function get_hessian(p::HessianProblem{MutatingEvaluation}, q, X)
    Y = zero_vector(p.M, q)
    return p.hessian!!(p.M, Y, q, X)
end
function get_hessian!(p::HessianProblem{AllocatingEvaluation}, Y, q, X)
    return copyto!(p.M, Y, p.hessian!!(p.M, q, X))
end
function get_hessian!(p::HessianProblem{MutatingEvaluation}, Y, q, X)
    return p.hessian!!(p.M, Y, q, X)
end

@doc raw"""
    get_preconditioner(p,x,ξ)

evaluate the symmetric, positive definite preconditioner (approximation of the
inverse of the Hessian of the cost function `F`) of a
[`HessianProblem`](@ref) `p` at the point `x`applied to a
tangent vector `ξ`.
"""
get_preconditioner(p::HessianProblem, x, X) = p.precon(p.M, x, X)

@doc raw"""
    approxHessianFiniteDifference{T, mT, P, G}

A functor to approximate the Hessian by a finite difference of gradient evaluations
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
    x::P,
    grad::G;
    steplength::R=2^-14,
    evaluation=AllocatingEvaluation(),
    retraction_method::RTR=ExponentialRetraction(),
    vector_transport_method::VTR=ParallelTransport(),
) where {
    mT<:AbstractManifold,
    P,
    G,
    R<:Real,
    RTR<:AbstractRetractionMethod,
    VTR<:AbstractVectorTransportMethod,
}
    X = zero_vector(M, x)
    Y = zero_vector(M, x)
    return ApproxHessianFiniteDifference{typeof(evaluation),P,typeof(X),G,RTR,VTR,R}(
        x, grad, X, Y, retraction_method, vector_transport_method, steplength
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
function (f::ApproxHessianFiniteDifference{MutatingEvaluation})(M, Y, x, X)
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
    p::P, o::O, i::Int
) where {P<:HessianProblem,O<:TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= o.initialResidualNorm * c.κ && i > 0
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
    p::P, o::O, i::Int
) where {P<:HessianProblem,O<:TruncatedConjugateGradientOptions}
    if norm(p.M, o.x, o.residual) <= o.initialResidualNorm^(1 + c.θ) && i > 0
        c.reason = "The algorithm reached superlinear convergence (residual at least reduced by power 1 + θ=$(1+(c.θ))).\n"
        return true
    end
    return false
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
[`StoreOptionsAction`](@ref) `a`, which is initialized to store
`:η, :δ, :residual` by default.

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenTrustRegionIsExceeded <: StoppingCriterion
    reason::String
end
StopWhenTrustRegionIsExceeded() = StopWhenTrustRegionIsExceeded("")
function (c::StopWhenTrustRegionIsExceeded)(
    ::HessianProblem, o::TruncatedConjugateGradientOptions, i::Int
)
    if o.ηPη >= o.trust_region_radius^2 && i >= 0
        c.reason = "Trust-region radius violation (‖η‖² = $(o.ηPη)) >= $(o.trust_region_radius^2) = trust_region_radius²). \n"
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
    ::HessianProblem, o::TruncatedConjugateGradientOptions, i::Int
)
    if o.δHδ <= 0 && i > 0
        c.reason = "Negative curvature. The model is not strictly convex (⟨δ,Hδ⟩_x = $(o.δHδ))) <= 0).\n"
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
    ::HessianProblem, o::TruncatedConjugateGradientOptions, i::Int
)
    if i > 0 && (o.new_model_value > o.model_value)
        c.reason = "Model value increased from $(o.model_value) to $(o.new_model_value).\n"
        return true
    end
    return false
end
