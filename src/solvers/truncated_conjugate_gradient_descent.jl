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
    p::P
    stop::SC
    X::T
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
        p::P,
        η::T;
        trust_region_radius::R=injectivity_radius(M) / 4.0,
        randomize::Bool=false,
        project!::F=copyto!,
        θ::Float64=1.0,
        κ::Float64=0.1,
        stopping_criterion::StoppingCriterion=StopAfterIteration(manifold_dimension(M)) |
                                              StopWhenResidualIsReducedByFactorOrPower(;
                                                  κ=κ, θ=θ
                                              ) |
                                              StopWhenTrustRegionIsExceeded() |
                                              StopWhenCurvatureIsNegative() |
                                              StopWhenModelIncreased(),
    ) where {P,T,R<:Real,F}
        tcgs = new{P,T,R,typeof(stopping_criterion),F}()
        tcgs.p = p
        tcgs.stop = stopping_criterion
        tcgs.η = η
        tcgs.trust_region_radius = trust_region_radius
        tcgs.randomize = randomize
        tcgs.project! = project!
        tcgs.model_value = zero(trust_region_radius)
        return tcgs
    end
end
function show(io::IO, tcgs::TruncatedConjugateGradientState)
    i = get_count(tcgs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(tcgs.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Truncated Conjugate Gradient Descent
    $Iter
    ## Parameters
    * randomize: $(tcgs.randomize)
    * trust region radius: $(tcgs.trust_region_radius)

    ## Stopping Criterion
    $(status_summary(tcgs.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

#
# Spcial stopping Criteria
#

@doc raw"""
    StopWhenResidualIsReducedByFactorOrPower <: StoppingCriterion
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
    StopWhenResidualIsReducedByFactorOrPower(; κ=0.1, θ=1.0)
initialize the StopWhenResidualIsReducedByFactorOrPower functor to indicate to stop after
the norm of the current residual is lesser than either the norm of the initial residual
to the power of 1+θ or the norm of the initial residual times κ.
# See also

[`truncated_conjugate_gradient_descent`](@ref), [`trust_regions`](@ref)
"""
mutable struct StopWhenResidualIsReducedByFactorOrPower <: StoppingCriterion
    κ::Float64
    θ::Float64
    reason::String
    at_iteration::Int
    function StopWhenResidualIsReducedByFactorOrPower(; κ::Float64=0.1, θ::Float64=1.0)
        return new(κ, θ, "", 0)
    end
end
function (c::StopWhenResidualIsReducedByFactorOrPower)(
    mp::AbstractManoptProblem, tcgstate::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if norm(get_manifold(mp), tcgstate.p, tcgstate.residual) <=
       tcgstate.initialResidualNorm * min(c.κ, tcgstate.initialResidualNorm^(c.θ)) && i > 0
        c.reason = "The norm of the residual is less than or equal either to κ=$(c.κ) times the norm of the initial residual or to the norm of the initial residual to the power 1 + θ=$(1+(c.θ)). \n"
        return true
    end
    return false
end
function status_summary(c::StopWhenResidualIsReducedByFactorOrPower)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Residual reduced by factor $(c.κ) or power $(c.θ):\t$s"
end
function show(io::IO, c::StopWhenResidualIsReducedByFactorOrPower)
    return print(
        io,
        "StopWhenResidualIsReducedByFactorOrPower($(c.κ), $(c.θ))\n    $(status_summary(c))",
    )
end

@doc raw"""
    update_stopping_criterion!(c::StopWhenResidualIsReducedByFactorOrPower, :ResidualPower, v)
Update the residual Power `θ`  to `v`.
"""
function update_stopping_criterion!(
    c::StopWhenResidualIsReducedByFactorOrPower, ::Val{:ResidualPower}, v
)
    c.θ = v
    return c
end

@doc raw"""
    update_stopping_criterion!(c::StopWhenResidualIsReducedByFactorOrPower, :ResidualFactor, v)
Update the residual Factor `κ` to `v`.
"""
function update_stopping_criterion!(
    c::StopWhenResidualIsReducedByFactorOrPower, ::Val{:ResidualFactor}, v
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
    at_iteration::Int
end
StopWhenTrustRegionIsExceeded() = StopWhenTrustRegionIsExceeded("", 0)
function (c::StopWhenTrustRegionIsExceeded)(
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if tcgs.ηPη >= tcgs.trust_region_radius^2 && i >= 0
        c.reason = "Trust-region radius violation (‖η‖² = $(tcgs.ηPη)) >= $(tcgs.trust_region_radius^2) = trust_region_radius²). \n"
        c.at_iteration = i
        return true
    end
    return false
end
function status_summary(c::StopWhenTrustRegionIsExceeded)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Trust region exceeded:\t$s"
end
function show(io::IO, c::StopWhenTrustRegionIsExceeded)
    return print(io, "StopWhenTrustRegionIsExceeded()\n    $(status_summary(c))")
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
    at_iteration::Int
end
StopWhenCurvatureIsNegative() = StopWhenCurvatureIsNegative("", 0)
function (c::StopWhenCurvatureIsNegative)(
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if tcgs.δHδ <= 0 && i > 0
        c.reason = "Negative curvature. The model is not strictly convex (⟨δ,Hδ⟩_x = $(tcgs.δHδ))) <= 0).\n"
        c.at_iteration = i
        return true
    end
    return false
end
function status_summary(c::StopWhenCurvatureIsNegative)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Cuvature is negative:\t$s"
end
function show(io::IO, c::StopWhenCurvatureIsNegative)
    return print(io, "StopWhenCurvatureIsNegative()\n    $(status_summary(c))")
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
    at_iteration::Int
end
StopWhenModelIncreased() = StopWhenModelIncreased("", 0)
function (c::StopWhenModelIncreased)(
    ::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, i::Int
)
    if i == 0 # reset on init
        c.reason = ""
        c.at_iteration = 0
    end
    if i > 0 && (tcgs.new_model_value > tcgs.model_value)
        c.reason = "Model value increased from $(tcgs.model_value) to $(tcgs.new_model_value).\n"
        return true
    end
    return false
end
function status_summary(c::StopWhenModelIncreased)
    has_stopped = length(c.reason) > 0
    s = has_stopped ? "reached" : "not reached"
    return "Model Increased:\t$s"
end
function show(io::IO, c::StopWhenModelIncreased)
    return print(io, "StopWhenModelIncreased()\n    $(status_summary(c))")
end

@doc raw"""
    truncated_conjugate_gradient_descent(M, f, grad_f, p, η, Hess_f, trust_region_radius)

solve the trust-region subproblem

```math
\operatorname*{arg\,min}_{η ∈ T_pM}
m_p(η) \quad\text{where}
m_p(η) = f(p) + ⟨\operatorname{grad} f(p),η⟩_x + \frac{1}{2}⟨\operatorname{Hess} f(p)[η],η⟩_x,
```

```math
\text{such that}\quad ⟨η,η⟩_x ≤ Δ^2
```

on a manifold M by using the Steihaug-Toint truncated conjugate-gradient method,
abbreviated tCG-method.
For a description of the algorithm and theorems offering convergence guarantees,
see the reference:

* P.-A. Absil, C.G. Baker, K.A. Gallivan,
    Trust-region methods on Riemannian manifolds, FoCM, 2007.
    doi: [10.1007/s10208-005-0179-9](https://doi.org/10.1007/s10208-005-0179-9)
* A. R. Conn, N. I. M. Gould, P. L. Toint, Trust-region methods, SIAM,
    MPS, 2000. doi: [10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857)

# Input

* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``F: \mathcal M → ℝ`` to minimize
* `grad_f` – the gradient ``\operatorname{grad}f: \mathcal M → T\mathcal M`` of `F`
* `Hess_f` – the hessian ``\operatorname{Hess}f: T_p\mathcal M → T_p\mathcal M``, ``X ↦ \operatorname{Hess}F(p)[X] = ∇_X\operatorname{grad}f(p)``
* `p` – a point on the manifold ``p ∈ \mathcal M``
* `η` – an update tangential vector ``η ∈ T_p\mathcal M``

# Optional

* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient and hessian work by
   allocation (default) or [`InplaceEvaluation`](@ref) in place
* `preconditioner` – a preconditioner for the hessian H
* `θ` – (`1.0`) 1+θ is the superlinear convergence target rate. The method aborts
    if the residual is less than or equal to the initial residual to the power of 1+θ.
* `κ` – (`0.1`) the linear convergence target rate. The method aborts if the
    residual is less than or equal to κ times the initial residual.
* `randomize` – set to true if the trust-region solve is to be initiated with a
    random tangent vector. If set to true, no preconditioner will be
    used. This option is set to true in some scenarios to escape saddle
    points, but is otherwise seldom activated.
* `trust_region_radius` – (`injectivity_radius(M)/4`) a trust-region radius
* `project!` : (`copyto!`) specify a projection operation for tangent vectors
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.
* `stopping_criterion` – ([`StopAfterIteration`](@ref)` | [`StopWhenResidualIsReducedByFactorOrPower`](@ref)` | '[`StopWhenCurvatureIsNegative`](@ref)` | `[`StopWhenTrustRegionIsExceeded`](@ref) )
    a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop,
    where for the default, the maximal number of iterations is set to the dimension of the
    manifold, the power factor is `θ`, the reduction factor is `κ`.

and the ones that are passed to [`decorate_state!`](@ref) for decorators.

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
* `f` – a cost function ``F: \mathcal M → ℝ`` to minimize
* `grad_f` – the gradient ``\operatorname{grad}f: \mathcal M → T\mathcal M`` of `f`
* `Hess_f` – the hessian ``\operatorname{Hess}f(x): T_p\mathcal M → T_p\mathcal M``, ``X ↦ \operatorname{Hess}f(p)[X]``
* `p` – a point on the manifold ``p ∈ \mathcal M``
* `X` – an update tangential vector ``X ∈ T_x\mathcal M``

For more details and all optional arguments, see [`truncated_conjugate_gradient_descent`](@ref).
"""
function truncated_conjugate_gradient_descent!(
    M::AbstractManifold,
    f::TF,
    grad_f::TG,
    p,
    X,
    Hess_f::TH;
    trust_region_radius::Float64=injectivity_radius(M) / 4,
    evaluation=AllocatingEvaluation(),
    preconditioner::Tprec=(M, x, ξ) -> ξ,
    θ::Float64=1.0,
    κ::Float64=0.1,
    randomize::Bool=false,
    stopping_criterion::StoppingCriterion=StopAfterIteration(manifold_dimension(M)) |
                                          StopWhenResidualIsReducedByFactorOrPower(;
                                              κ=κ, θ=θ
                                          ) |
                                          StopWhenTrustRegionIsExceeded() |
                                          StopWhenCurvatureIsNegative() |
                                          StopWhenModelIncreased(),
    project!::Proj=copyto!,
    kwargs..., #collect rest
) where {TF,TG,TH,Tprec,Proj}
    mho = ManifoldHessianObjective(f, grad_f, Hess_f, preconditioner; evaluation=evaluation)
    dmho = decorate_objective!(M, mho; kwargs...)
    mp = DefaultManoptProblem(M, dmho)
    tcgs = TruncatedConjugateGradientState(
        M,
        p,
        X;
        trust_region_radius=trust_region_radius,
        randomize=randomize,
        θ=θ,
        κ=κ,
        stopping_criterion=stopping_criterion,
        (project!)=project!,
    )
    tcgs = decorate_state!(tcgs; kwargs...)
    return get_solver_return(solve!(mp, tcgs))
end

function initialize_solver!(
    mp::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState
)
    M = get_manifold(mp)
    (tcgs.randomize) || zero_vector!(M, tcgs.η, tcgs.p)
    tcgs.Hη = tcgs.randomize ? get_hessian(mp, tcgs.p, tcgs.η) : zero_vector(M, tcgs.p)
    tcgs.X = get_gradient(mp, tcgs.p)
    tcgs.residual = tcgs.randomize ? tcgs.X + tcgs.Hη : tcgs.X
    tcgs.z = tcgs.randomize ? tcgs.residual : get_preconditioner(mp, tcgs.p, tcgs.residual)
    tcgs.δ = -deepcopy(tcgs.z)
    tcgs.Hδ = zero_vector(M, tcgs.p)
    tcgs.δHδ = inner(M, tcgs.p, tcgs.δ, tcgs.Hδ)
    tcgs.ηPδ = tcgs.randomize ? inner(M, tcgs.p, tcgs.η, tcgs.δ) : zero(tcgs.δHδ)
    tcgs.δPδ = inner(M, tcgs.p, tcgs.residual, tcgs.z)
    tcgs.ηPη = tcgs.randomize ? inner(M, tcgs.p, tcgs.η, tcgs.η) : zero(tcgs.δHδ)
    if tcgs.randomize
        tcgs.model_value =
            real(inner(M, tcgs.p, tcgs.η, tcgs.X)) +
            0.5 * real(inner(M, tcgs.p, tcgs.η, tcgs.Hη))
    else
        tcgs.model_value = 0
    end
    tcgs.z_r = inner(M, tcgs.p, tcgs.z, tcgs.residual)
    tcgs.initialResidualNorm = norm(M, tcgs.p, tcgs.residual)
    return tcgs
end
function step_solver!(
    mp::AbstractManoptProblem, tcgs::TruncatedConjugateGradientState, ::Any
)
    M = get_manifold(mp)
    get_hessian!(mp, tcgs.Hδ, tcgs.p, tcgs.δ)
    tcgs.δHδ = inner(M, tcgs.p, tcgs.δ, tcgs.Hδ)
    α = tcgs.z_r / tcgs.δHδ
    ηPη_new = tcgs.ηPη + 2 * α * tcgs.ηPδ + α^2 * tcgs.δPδ
    # Check against negative curvature and trust-region radius violation.
    if tcgs.δHδ <= 0 || ηPη_new >= tcgs.trust_region_radius^2
        τ =
            (
                -tcgs.ηPδ +
                sqrt(tcgs.ηPδ^2 + tcgs.δPδ * (tcgs.trust_region_radius^2 - tcgs.ηPη))
            ) / tcgs.δPδ
        tcgs.η = tcgs.η + τ * tcgs.δ
        tcgs.Hη = tcgs.Hη + τ * tcgs.Hδ
        tcgs.ηPη = ηPη_new
        return tcgs
    end
    tcgs.ηPη = ηPη_new
    new_η = tcgs.η + α * tcgs.δ
    new_Hη = tcgs.Hη + α * tcgs.Hδ
    # No negative curvature and s.η - α * (s.δ) inside TR: accept it.
    tcgs.new_model_value =
        real(inner(M, tcgs.p, new_η, tcgs.X)) + 0.5 * real(inner(M, tcgs.p, new_η, new_Hη))
    tcgs.new_model_value >= tcgs.model_value && return tcgs
    copyto!(M, tcgs.η, tcgs.p, new_η)
    tcgs.model_value = tcgs.new_model_value
    copyto!(M, tcgs.Hη, tcgs.p, new_Hη)
    tcgs.residual = tcgs.residual + α * tcgs.Hδ

    # Precondition the residual.
    tcgs.z = tcgs.randomize ? tcgs.residual : get_preconditioner(mp, tcgs.p, tcgs.residual)
    zr = inner(M, tcgs.p, tcgs.z, tcgs.residual)
    # Compute new search direction.
    β = zr / tcgs.z_r
    tcgs.z_r = zr
    tcgs.δ = -tcgs.z + β * tcgs.δ
    tcgs.project!(M, tcgs.δ, tcgs.p, tcgs.δ)
    tcgs.ηPδ = β * (α * tcgs.δPδ + tcgs.ηPδ)
    tcgs.δPδ = tcgs.z_r + β^2 * tcgs.δPδ
    return tcgs
end
get_solver_result(s::TruncatedConjugateGradientState) = s.η
