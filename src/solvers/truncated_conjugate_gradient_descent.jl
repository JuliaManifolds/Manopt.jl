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
* `F` – a cost function ``F: \mathcal M → ℝ`` to minimize
* `gradF` – the gradient ``\operatorname{grad}F: \mathcal M → T\mathcal M`` of `F`
* `HessF` – the hessian ``\operatorname{Hess}F(x): T_x\mathcal M → T_x\mathcal M``, ``X ↦ \operatorname{Hess}F(x)[X] = ∇_X\operatorname{grad}f(x)``
* `x` – a point on the manifold ``x ∈ \mathcal M``
* `η` – an update tangential vector ``η ∈ T_x\mathcal M``
* `HessF` – the hessian ``\operatorname{Hess}F(x): T_x\mathcal M → T_x\mathcal M``, ``X ↦ \operatorname{Hess}F(x)[X] = ∇_ξ\operatorname{grad}f(x)``

# Optional

* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient and hessian work by
   allocation (default) or [`MutatingEvaluation`](@ref) in place
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
* `stopping_criterion` – ([`StopWhenAny`](@ref), [`StopAfterIteration`](@ref),
    [`StopIfResidualIsReducedByFactor`](@ref), [`StopIfResidualIsReducedByPower`](@ref),
    [`StopWhenCurvatureIsNegative`](@ref), [`StopWhenTrustRegionIsExceeded`](@ref) )
    a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop,
    where for the default, the maximal number of iterations is set to the dimension of the
    manifold, the power factor is `θ`, the reduction factor is `κ`.

and the ones that are passed to [`decorate_options`](@ref) for decorators.

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
@deprecate truncated_conjugate_gradient_descent(M, F, gradF, x, η, H, r; kwargs...) truncated_conjugate_gradient_descent(
    M, F, gradF, x, η, H; trust_region_radius=r, kwargs...
)
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
    stopping_criterion::StoppingCriterion=(
        StopAfterIteration(manifold_dimension(M)) |
        StopIfResidualIsReducedByFactorOrPower(κ, θ) |
        StopWhenTrustRegionIsExceeded() |
        StopWhenCurvatureIsNegative() |
        StopWhenModelIncreased()
    ),
    project!::Proj=copyto!,
    kwargs..., #collect rest
) where {TF,TG,TH,Tprec,Proj}
    p = HessianProblem(M, F, gradF, H, preconditioner; evaluation=evaluation)
    o = TruncatedConjugateGradientOptions(
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
    o = decorate_options(o; kwargs...)
    return get_solver_return(solve(p, o))
end
@deprecate truncated_conjugate_gradient_descent!(M, F, gradF, x, η, H, r; kwargs...) truncated_conjugate_gradient_descent!(
    M, F, gradF, x, η, H; trust_region_radius=r, kwargs...
)

function initialize_solver!(p::HessianProblem, o::TruncatedConjugateGradientOptions)
    (o.randomize) || zero_vector!(p.M, o.η, o.x)
    o.Hη = o.randomize ? get_hessian(p, o.x, o.η) : zero_vector(p.M, o.x)
    o.gradient = get_gradient(p, o.x)
    o.residual = o.randomize ? o.gradient + o.Hη : o.gradient
    o.z = o.randomize ? o.residual : get_preconditioner(p, o.x, o.residual)
    o.δ = -deepcopy(o.z)
    o.Hδ = zero_vector(p.M, o.x)
    o.δHδ = inner(p.M, o.x, o.δ, o.Hδ)
    o.ηPδ = o.randomize ? inner(p.M, o.x, o.η, o.δ) : zero(o.δHδ)
    o.δPδ = inner(p.M, o.x, o.residual, o.z)
    o.ηPη = o.randomize ? inner(p.M, o.x, o.η, o.η) : zero(o.δHδ)
    if o.randomize
        o.model_value = inner(p.M, o.x, o.η, o.gradient) + 0.5 * inner(p.M, o.x, o.η, o.Hη)
    else
        o.model_value = 0
    end
    o.z_r = inner(p.M, o.x, o.z, o.residual)
    o.initialResidualNorm = sqrt(inner(p.M, o.x, o.residual, o.residual))
    return o
end
function step_solver!(
    p::P, o::O, ::Int
) where {P<:HessianProblem,O<:TruncatedConjugateGradientOptions}
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
get_solver_result(o::O) where {O<:TruncatedConjugateGradientOptions} = o.η
