
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
        project! = copyto!,
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
        p::HessianProblem,
        x::P,
        η::T,
        trust_region_radius::R,
        randomize::Bool;
        project!::Proj=copyto!,
        θ::Float64=1.0,
        κ::Float64=0.1,
        stop::StoppingCriterion=StopAfterIteration(manifold_dimension(p.M)) |
                                StopIfResidualIsReducedByFactorOrPower(; κ=κ, θ=θ) |
                                StopWhenTrustRegionIsExceeded() |
                                StopWhenCurvatureIsNegative() |
                                StopWhenModelIncreased(),
    ) where {P,T,R<:Real,Proj}
        return TruncatedConjugateGradientState(
            p.M,
            x,
            η;
            trust_region_radius=trust_region_radius,
            (project!)=project!,
            randomize=randomize,
            stopping_criterion=stop,
        )
    end
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
        o = new{P,T,R,typeof(stopping_criterion),F}()
        o.x = x
        o.stop = stopping_criterion
        o.η = η
        o.trust_region_radius = trust_region_radius
        o.randomize = randomize
        o.project! = project!
        o.model_value = zero(trust_region_radius)
        o.κ = zero(trust_region_radius)
        return o
    end
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

function initialize_solver!(p::HessianProblem, s::TruncatedConjugateGradientState)
    (s.randomize) || zero_vector!(p.M, s.η, s.x)
    s.Hη = s.randomize ? get_hessian(p, s.x, s.η) : zero_vector(p.M, s.x)
    s.gradient = get_gradient(p, s.x)
    s.residual = s.randomize ? s.gradient + s.Hη : s.gradient
    s.z = s.randomize ? s.residual : get_preconditioner(p, s.x, s.residual)
    s.δ = -deepcopy(s.z)
    s.Hδ = zero_vector(p.M, s.x)
    s.δHδ = inner(p.M, s.x, s.δ, s.Hδ)
    s.ηPδ = s.randomize ? inner(p.M, s.x, s.η, s.δ) : zero(s.δHδ)
    s.δPδ = inner(p.M, s.x, s.residual, s.z)
    s.ηPη = s.randomize ? inner(p.M, s.x, s.η, s.η) : zero(s.δHδ)
    if s.randomize
        s.model_value = inner(p.M, s.x, s.η, s.gradient) + 0.5 * inner(p.M, s.x, s.η, s.Hη)
    else
        s.model_value = 0
    end
    s.z_r = inner(p.M, s.x, s.z, s.residual)
    s.initialResidualNorm = sqrt(inner(p.M, s.x, s.residual, s.residual))
    return s
end
function step_solver!(p::HessianProblem, s::TruncatedConjugateGradientState, ::Any)
    # Updates
    get_hessian!(p, o.Hδ, o.x, o.δ)
    o.δHδ = inner(p.M, o.x, o.δ, o.Hδ)
    α = o.z_r / o.δHδ
    ηPη_new = o.ηPη + 2 * α * o.ηPδ + α^2 * o.δPδ
    # Check against negative curvature and trust-region radius violation.
    if o.δHδ <= 0 || ηPη_new >= o.trust_region_radius^2
        τ = (-o.ηPδ + sqrt(o.ηPδ^2 + o.δPδ * (o.trust_region_radius^2 - o.ηPη))) / o.δPδ
        o.η = o.η + τ * o.δ
        o.Hη = o.Hη + τ * o.Hδ
        o.ηPη = ηPη_new
        return o
    end
    o.ηPη = ηPη_new
    new_η = o.η + α * o.δ
    new_Hη = o.Hη + α * o.Hδ
    # No negative curvature and o.η - α * (o.δ) inside TR: accept it.
    o.new_model_value =
        inner(p.M, o.x, new_η, o.gradient) + 0.5 * inner(p.M, o.x, new_η, new_Hη)
    o.new_model_value >= o.model_value && return o
    copyto!(p.M, o.η, o.x, new_η)
    o.model_value = o.new_model_value
    copyto!(p.M, o.Hη, o.x, new_Hη)
    o.residual = o.residual + α * o.Hδ

    #=
    if norm(p.M, o.x, o.residual) <= o.initialResidualNorm * min(o.initialResidualNorm^(0.1), 0.9)
        if 0.9 < o.initialResidualNorm^(0.1)
            print("Linear \n")
        else
            print("Superlinear \n")
        end
    end
    =#

    # Precondition the residual.
    o.z = o.randomize ? o.residual : get_preconditioner(p, o.x, o.residual)
    zr = inner(p.M, o.x, o.z, o.residual)
    # Compute new search direction.
    β = zr / o.z_r
    o.z_r = zr
    o.δ = -o.z + β * o.δ
    o.project!(p.M, o.δ, o.x, o.δ)
    o.ηPδ = β * (α * o.δPδ + o.ηPδ)
    o.δPδ = o.z_r + β^2 * o.δPδ
    return o
end
get_solver_result(s::TruncatedConjugateGradientState) = s.η
