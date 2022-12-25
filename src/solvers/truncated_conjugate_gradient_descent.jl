@doc raw"""
    TruncatedConjugateGradientState <: AbstractHessianSolverState

describe the Steihaug-Toint truncated conjugate-gradient method, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x` : a point, where the trust-region subproblem needs
    to be solved
* `η` : a tangent vector (called update vector), which solves the
    trust-region subproblem after successful calculation by the algorithm
* `stop` : a [`StoppingCriterion`](@ref).
* `gradient` : the gradient at the current iterate
* `δ` : search direction
* `trust_region_radius` : (`injectivity_radius(M)/4`) the trust-region radius
* `residual` : the gradient
* `randomize` : indicates if the trust-region solve and so the algorithm is to be
        initiated with a random tangent vector. If set to true, no
        preconditioner will be used. This option is set to true in some
        scenarios to escape saddle points, but is otherwise seldom activated.
* `project!` : (`copyto!`) specify a projection operation for tangent vectors
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.

# Constructor

    TruncatedConjugateGradientState(M, p, x, η;
        trust_region_radius=injectivity_radius(M)/4,
        randomize=false,
        θ=1.0,
        κ=0.1,
        project!=copyto!,
    )

    and a slightly involved `stopping_criterion`

# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct TruncatedConjugateGradientState{P,T,R<:Real,SC<:StoppingCriterion,Proj} <:
               AbstractHessianSolverState
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
    randomize::Bool
    project!::Proj
    initialResidualNorm::Float64
    function TruncatedConjugateGradientState(
        M::AbstractManifold,
        x::P,
        η::T;
        trust_region_radius::R=injectivity_radius(M) / 4.0,
        randomize::Bool=false,
        project!::F=copyto!,
        θ::Float64=1.0,
        κ::Float64=0.1,
        stopping_criterion::StoppingCriterion=StopAfterIteration(manifold_dimension(p.M)) |
                                              StopIfResidualIsReducedByFactorOrPower(;
                                                  κ=κ, θ=θ
                                              ) |
                                              StopWhenTrustRegionIsExceeded() |
                                              StopWhenCurvatureIsNegative() |
                                              StopWhenModelIncreased(),
    ) where {P,T,R<:Real,F}
        tcgs = new{P,T,R,typeof(stopping_criterion),F}()
        tcgs.x = x
        tcgs.stop = stopping_criterion
        tcgs.η = η
        tcgs.trust_region_radius = trust_region_radius
        tcgs.randomize = randomize
        tcgs.project! = project!
        tcgs.model_value = zero(trust_region_radius)
        tcgs.κ = zero(trust_region_radius)
        return tcgs
    end
end

#
# Spcial stopping Criteria
#

@doc raw"""
    StopIfResidualIsReducedByFactorOrPower <: StoppingCriterion
A functor for testing if the norm of residual at the current iterate is reduced
either by a power of 1+θ or by a factor κ compared to the norm of the initial
residual, i.e. $\Vert r_k \Vert_x \leqq \Vert r_0 \Vert_{x} \
\min \left( \kappa, \Vert r_0 \Vert_{x}^{\theta} \right)$.
# Fields
* `κ` – the reduction factor
* `θ` – part of the reduction power
* `reason` – stores a reason of stopping if the stopping criterion has one be
    reached, see [`get_reason`](@ref).
# Constructor
    StopIfResidualIsReducedByFactorOrPower(; κ=0.1, θ=1.0)
initialize the StopIfResidualIsReducedByFactorOrPower functor to indicate to stop after
the norm of the current residual is lesser than either the norm of the initial residual
to the power of 1+θ or the norm of the initial residual times κ.
# See also
[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopIfResidualIsReducedByFactorOrPower <: StoppingCriterion
    κ::Float64
    θ::Float64
    reason::String
    StopIfResidualIsReducedByFactorOrPower(; κ::Float64=0.1, θ::Float64=1.0) = new(κ, θ, "")
end
function (c::StopIfResidualIsReducedByFactorOrPower)(
    mp::AbstractManoptProblem, tcgstate::TruncatedConjugateGradientState, i::Int
)
    if norm(get_manifold(mp), tcgstate.x, tcgstate.residual) <=
       tcgstate.initialResidualNorm * min(c.κ, tcgstate.initialResidualNorm^(c.θ)) && i > 0
        c.reason = "The norm of the residual is less than or equal either to κ=$(c.κ) times the norm of the initial residual or to the norm of the initial residual to the power 1 + θ=$(1+(c.θ)). \n"
        return true
    end
    return false
end
@doc raw"""
    update_stopping_criterion!(c::StopIfResidualIsReducedByFactorOrPower, :ResidualPower, v)
Update the residual Power `θ`  to `v`.
"""
function update_stopping_criterion!(
    c::StopIfResidualIsReducedByFactorOrPower, ::Val{:ResidualPower}, v
)
    c.θ = v
    return c
end

@doc raw"""
    update_stopping_criterion!(c::StopIfResidualIsReducedByFactorOrPower, :ResidualFactor, v)
Update the residual Factor `κ` to `v`.
"""
function update_stopping_criterion!(
    c::StopIfResidualIsReducedByFactorOrPower, ::Val{:ResidualFactor}, v
)
    c.κ = v
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
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if tcgs.ηPη >= tcgs.trust_region_radius^2 && i >= 0
        c.reason = "Trust-region radius violation (‖η‖² = $(tcgs.ηPη)) >= $(tcgs.trust_region_radius^2) = trust_region_radius²). \n"
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
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if tcgs.δHδ <= 0 && i > 0
        c.reason = "Negative curvature. The model is not strictly convex (⟨δ,Hδ⟩_x = $(tcgs.δHδ))) <= 0).\n"
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
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if i > 0 && (tcgs.new_model_value > tcgs.model_value)
        c.reason = "Model value increased from $(tcgs.model_value) to $(tcgs.new_model_value).\n"
        return true
    end
    return false
end

@doc raw"""
    truncated_conjugate_gradient_descent(M, F, gradF, x, η, HessF, trust_region_radius)

solve the trust-region subproblem

```math
\operatorname*{arg\,min}_{η ∈ T_xM}
m_x(η) \quad\text{where}
m_x(η) = F(x) + ⟨\operatorname{grad}F(x),η⟩_x + \frac{1}{2}⟨\operatorname{Hess}F(x)[η],η⟩_x,
```

```math
\text{such that}\quad ⟨η,η⟩_x ≤ Δ^2
```

with the [`truncated_conjugate_gradient_descent`](@ref).
For a description of the algorithm and theorems offering convergence guarantees,
see the reference:

* P.-A. Absil, C.G. Baker, K.A. Gallivan,
    Trust-region methods on Riemannian manifolds, FoCM, 2007.
    doi: [10.1007/s10208-005-0179-9](https://doi.org/10.1007/s10208-005-0179-9)
* A. R. Conn, N. I. M. Gould, P. L. Toint, Trust-region methods, SIAM,
    MPS, 2000. doi: [10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857)

# Input

* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F: \mathcal M → ℝ`` to minimize
* `gradF` – the gradient ``\operatorname{grad}F: \mathcal M → T\mathcal M`` of `F`
* `HessF` – the hessian ``\operatorname{Hess}F(x): T_x\mathcal M → T_x\mathcal M``, ``X ↦ \operatorname{Hess}F(x)[X] = ∇_X\operatorname{grad}f(x)``
* `x` – a point on the manifold ``x ∈ \mathcal M``
* `η` – an update tangential vector ``η ∈ T_x\mathcal M``
* `HessF` – the hessian ``\operatorname{Hess}F(x): T_x\mathcal M → T_x\mathcal M``, ``X ↦ \operatorname{Hess}F(x)[X] = ∇_ξ\operatorname{grad}f(x)``

# Optional

* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient and hessian work by
   allocation (default) or [`InplaceEvaluation`](@ref) in place
* `preconditioner` – a preconditioner for the hessian H
* `θ` – (`1.0`) 1+θ is the superlinear convergence target rate. The algorithm will
    terminate early if the residual was reduced by a power of 1+theta.
* `κ` – (`0.1`) the linear convergence target rate: algorithm will terminate
    early if the residual was reduced by a factor of kappa.
* `randomize` – set to true if the trust-region solve is to be initiated with a
    random tangent vector. If set to true, no preconditioner will be
    used. This option is set to true in some scenarios to escape saddle
    points, but is otherwise seldom activated.
* `trust_region_radius` – (`injectivity_radius(M)/4`) a trust-region radius
* `project!` : (`copyto!`) specify a projection operation for tangent vectors
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.
* `stopping_criterion` – ([`StopWhenAny`](@ref), [`StopAfterIteration`](@ref),
    [`StopIfResidualIsReducedByFactor`](@ref), [`StopIfResidualIsReducedByPower`](@ref),
    [`StopWhenCurvatureIsNegative`](@ref), [`StopWhenTrustRegionIsExceeded`](@ref) )
    a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop,
    where for the default, the maximal number of iterations is set to the dimension of the
    manifold, the power factor is `θ`, the reduction factor is `κ`.

and the ones that are passed to [`decorate_state`](@ref) for decorators.

# Output

# Output

the obtained (approximate) minimizer ``\eta^*``, see [`get_solver_return`](@ref) for details

# see also
[`trust_regions`](@ref)
"""
function truncated_conjugate_gradient_descent(
    M::AbstractManifold, F::TF, gradF::TG, x, η, H::TH; kwargs...
) where {TF,TG,TH}
    x_res = copy(M, x)
    return truncated_conjugate_gradient_descent!(M, F, gradF, x_res, η, H; kwargs...)
end
@doc raw"""
    truncated_conjugate_gradient_descent!(M, F, gradF, x, η, HessF, trust_region_radius; kwargs...)

solve the trust-region subproblem in place of `x`.

# Input
# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F: \mathcal M → ℝ`` to minimize
* `gradF` – the gradient ``\operatorname{grad}F: \mathcal M → T\mathcal M`` of `F`
* `HessF` – the hessian ``\operatorname{Hess}F(x): T_x\mathcal M → T_x\mathcal M``, ``X ↦ \operatorname{Hess}F(x)[X] = ∇_X\operatorname{grad}f(x)``
* `x` – a point on the manifold ``x ∈ \mathcal M``
* `η` – an update tangential vector ``η ∈ T_x\mathcal M``

For more details and all optional arguments, see [`truncated_conjugate_gradient_descent`](@ref).
"""
function truncated_conjugate_gradient_descent!(
    M::AbstractManifold,
    F::TF,
    gradF::TG,
    x,
    η,
    H::TH;
    trust_region_radius::Float64=injectivity_radius(M) / 4,
    evaluation=AllocatingEvaluation(),
    preconditioner::Tprec=(M, x, ξ) -> ξ,
    θ::Float64=1.0,
    κ::Float64=0.1,
    randomize::Bool=false,
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(manifold_dimension(M)),
        StopWhenAll(StopIfResidualIsReducedByPower(θ), StopIfResidualIsReducedByFactor(κ)),
        StopWhenTrustRegionIsExceeded(),
        StopWhenCurvatureIsNegative(),
        StopWhenModelIncreased(),
    ),
    project!::Proj=copyto!,
    kwargs..., #collect rest
) where {TF,TG,TH,Tprec,Proj}
    p = HessianProblem(M, F, gradF, H, preconditioner; evaluation=evaluation)
    o = TruncatedConjugateGradientState(
        M,
        x,
        η;
        trust_region_radius=trust_region_radius,
        randomize=randomize,
        θ=θ,
        κ=κ,
        stopping_criterion=stopping_criterion,
        (project!)=project!,
    )
    o = decorate_state(o; kwargs...)
    return get_solver_return(solve!(p, o))
end

function initialize_solver!(
    mp::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState
)
    M = get_manifold(mp)
    (tcgs.randomize) || zero_vector!(M, tcgs.η, tcgs.x)
    tcgs.Hη = tcgs.randomize ? get_hessian(mp, tcgs.x, tcgs.η) : zero_vector(M, tcgs.x)
    tcgs.gradient = get_gradient(mp, tcgs.x)
    tcgs.residual = tcgs.randomize ? tcgs.gradient + tcgs.Hη : tcgs.gradient
    tcgs.z = tcgs.randomize ? tcgs.residual : get_preconditioner(mp, tcgs.x, tcgs.residual)
    tcgs.δ = -deepcopy(tcgs.z)
    tcgs.Hδ = zero_vector(M, tcgs.x)
    tcgs.δHδ = inner(M, tcgs.x, tcgs.δ, tcgs.Hδ)
    tcgs.ηPδ = tcgs.randomize ? inner(M, tcgs.x, tcgs.η, tcgs.δ) : zero(tcgs.δHδ)
    tcgs.δPδ = inner(M, tcgs.x, tcgs.residual, tcgs.z)
    tcgs.ηPη = tcgs.randomize ? inner(M, tcgs.x, tcgs.η, tcgs.η) : zero(tcgs.δHδ)
    if tcgs.randomize
        tcgs.model_value =
            inner(M, tcgs.x, tcgs.η, tcgs.gradient) +
            0.5 * inner(M, tcgs.x, tcgs.η, tcgs.Hη)
    else
        tcgs.model_value = 0
    end
    tcgs.z_r = inner(M, tcgs.x, tcgs.z, tcgs.residual)
    tcgs.initialResidualNorm = sqrt(inner(M, tcgs.x, tcgs.residual, tcgs.residual))
    return tcgs
end
function step_solver!(p::AbstractManoptProblem, s::TruncatedConjugateGradientState, ::Any)
    # Updates
    get_hessian!(p, s.Hδ, s.x, s.δ)
    s.δHδ = inner(p.M, s.x, s.δ, s.Hδ)
    α = s.z_r / s.δHδ
    ηPη_new = s.ηPη + 2 * α * s.ηPδ + α^2 * s.δPδ
    # Check against negative curvature and trust-region radius violation.
    if s.δHδ <= 0 || ηPη_new >= s.trust_region_radius^2
        τ = (-s.ηPδ + sqrt(s.ηPδ^2 + s.δPδ * (s.trust_region_radius^2 - s.ηPη))) / s.δPδ
        s.η = s.η + τ * s.δ
        s.Hη = s.Hη + τ * s.Hδ
        s.ηPη = ηPη_new
        return o
    end
    s.ηPη = ηPη_new
    new_η = s.η + α * s.δ
    new_Hη = s.Hη + α * s.Hδ
    # No negative curvature and s.η - α * (s.δ) inside TR: accept it.
    s.new_model_value =
        inner(p.M, s.x, new_η, s.gradient) + 0.5 * inner(p.M, s.x, new_η, new_Hη)
    s.new_model_value >= s.model_value && return o
    copyto!(p.M, s.η, s.x, new_η)
    s.model_value = s.new_model_value
    copyto!(p.M, s.Hη, s.x, new_Hη)
    s.residual = s.residual + α * s.Hδ

    # Precondition the residual.
    s.z = s.randomize ? s.residual : get_preconditioner(p, s.x, s.residual)
    zr = inner(p.M, s.x, s.z, s.residual)
    # Compute new search direction.
    β = zr / s.z_r
    s.z_r = zr
    s.δ = -s.z + β * s.δ
    s.project!(p.M, s.δ, s.x, s.δ)
    s.ηPδ = β * (α * s.δPδ + s.ηPδ)
    s.δPδ = s.z_r + β^2 * s.δPδ
    return o
end
get_solver_result(s::TruncatedConjugateGradientState) = s.η
