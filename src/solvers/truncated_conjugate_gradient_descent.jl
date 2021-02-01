@doc raw"""
    truncated_conjugate_gradient_descent(M, F, gradF, x, η, H, trust_region_radius)

solve the trust-region subproblem

```math
\operatorname*{arg\,min}_{η  ∈  T_{x}M}
m_x(η) \quad\text{where} 
m_x(η) = F(x) + ⟨\operatorname{grad}F(x),η⟩_x + \frac{1}{2}⟨\operatorname{Hess}F(x)[η],η⟩_x,
```
```math
\text{such that}\quad ⟨η,η⟩_x \leqq {\Delta}^2
```

with the [truncated_conjugate_gradient_descent](@ref).
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
* `HessF` – the hessian ``Hf(x)[X]`` given as `HessF(M,p,X)`
* `x` – a point on the manifold ``x ∈ \mathcal M``
* `η` – an update tangential vector ``η ∈ T_x\mathcal M``
* `trust_region_radius` – a trust-region radius

# Optional
* `preconditioner` – a preconditioner for the hessian H
* `θ` – 1+θ is the superlinear convergence target rate. The algorithm will
    terminate early if the residual was reduced by a power of 1+theta.
* `κ` – the linear convergence target rate: algorithm will terminate
    early if the residual was reduced by a factor of kappa.
* `randomize` – set to true if the trust-region solve is to be initiated with a
    random tangent vector. If set to true, no preconditioner will be
    used. This option is set to true in some scenarios to escape saddle
    points, but is otherwise seldom activated.
* `stopping_criterion` – ([`StopWhenAny`](@ref), [`StopAfterIteration`](@ref),
    [`StopIfResidualIsReducedByFactor`](@ref), [`StopIfResidualIsReducedByPower`](@ref),
    [`StopWhenCurvatureIsNegative`](@ref), [`StopWhenTrustRegionIsExceeded`](@ref) )
    a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop,
    where for the default, the maximal number of iterations is set to the dimension of the
    manifold, the power factor is `θ`, the reduction factor is `κ`.
    .
* `return_options` – (`false`) – if actiavated, the extended result, i.e. the
    complete [`Options`](@ref) re returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `x_opt` is returned

and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `η` – an approximate solution of the trust-region subproblem in
    $\mathcal{T_{x}M}$.
OR
* `options` - the options returned by the solver (see `return_options`)

# see also
[`trust_regions`](@ref)
"""
function truncated_conjugate_gradient_descent(
    M::Manifold, F::TF, gradF::TG, x, η, H::TH, trust_region_radius::Float64; kwargs...
) where {TF,TG,TH}
    x_res = allocate(x)
    copyto!(x_res, x)
    return truncated_conjugate_gradient_descent!(
        M, F, gradF, x_res, η, H, trust_region_radius; kwargs...
    )
end
@doc raw"""
    truncated_conjugate_gradient_descent!(M, F, gradF, x, η, H, trust_region_radius; kwargs...)

solve the trust-region subproblem in place of `x`.

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F: \mathcal M→ℝ$ to minimize
* `gradF` – the gradient $\operatorname{grad}F:\mathcal M → T\mathcal M$ of F
* `x` – a point on the manifold ``x ∈ \mathcal M``
* `η` – an update tangential vector ``η ∈ \mathcal{T_{x}M}``
* `H` – the hessian $H( \mathcal M, x, \xi)$ of F
* `trust_region_radius` – a trust-region radius

For more details and all optional arguments, see [`truncated_conjugate_gradient_descent`](@ref).
"""
function truncated_conjugate_gradient_descent!(
    M::Manifold,
    F::TF,
    gradF::TG,
    x,
    η,
    H::TH,
    trust_region_radius::Float64;
    evaluation=AllocatingEvaluation(),
    preconditioner::Tprec=(M, x, ξ) -> ξ,
    θ::Float64=1.0,
    κ::Float64=0.1,
    randomize::Bool=false,
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(manifold_dimension(M)),
        StopIfResidualIsReducedByPower(
            sqrt(
                inner(
                    M,
                    x,
                    gradF(M, x) + (randomize ? H(M, x, η) : zero_tangent_vector(M, x)),
                    gradF(M, x) + (randomize ? H(M, x, η) : zero_tangent_vector(M, x)),
                ),
            ),
            θ,
        ),
        StopIfResidualIsReducedByFactor(
            sqrt(
                inner(
                    M,
                    x,
                    gradF(M, x) + (randomize ? H(M, x, η) : zero_tangent_vector(M, x)),
                    gradF(M, x) + (randomize ? H(M, x, η) : zero_tangent_vector(M, x)),
                ),
            ),
            κ,
        ),
        StopWhenTrustRegionIsExceeded(),
        StopWhenCurvatureIsNegative(),
    ),
    return_options=false,
    kwargs..., #collect rest
) where {TF,TG,TH,Tprec}
    p = HessianProblem(M, F, gradF, H, preconditioner; evaluation=evaluation)
    o = TruncatedConjugateGradientOptions(
        x, η, trust_region_radius, randomize, stopping_criterion
    )
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        resultO
    else
        return get_solver_result(resultO)
    end
end
function initialize_solver!(p::HessianProblem, o::TruncatedConjugateGradientOptions)
    o.η = o.randomize ? o.η : zero_tangent_vector(p.M, o.x)
    o.Hη = o.randomize ? get_hessian(p, o.x, o.η) : zero_tangent_vector(p.M, o.x)
    o.gradient = get_gradient(p, o.x)
    o.residual = o.gradient + o.Hη
    o.precon_residual = zero_tangent_vector(p.M, o.x)
    o.δ = o.randomize ? o.residual : get_preconditioner(p, o.x, o.residual)
    o.Hδ = zero_tangent_vector(p.M, o.x)
    if o.randomize
        o.model_value = 0
    else
        o.model_value = inner(p.M, o.x, o.η, o.gradient) + 0.5 * inner(p.M, o.x, o.η, o.Hη)
    end
    o.res_precon_res = inner(p.M, o.x, o.precon_residual, o.residual)
    return o
end
function step_solver!(
    p::P, o::O, i::Int
) where {P<:HessianProblem,O<:TruncatedConjugateGradientOptions}
    get_hessian!(p, o.Hδ, o.x, o.δ)
    o.δHδ = inner(p.M, o.x, o.δ, o.Hδ)
    # Note that if d_Hd == 0, we will exit at the next "if" anyway.
    α = o.res_precon_res / o.δHδ
    # <neweta,neweta>_P =
    # <eta,eta>_P + 2*alpha*<eta,delta>_P + alpha*alpha*<delta,delta>_P
    e_Pd = -inner(p.M, o.x, o.η, o.randomize ? o.δ : get_preconditioner(p, o.x, o.δ)) # It must be clarified if it's negative or not
    d_Pd = inner(p.M, o.x, o.δ, o.randomize ? o.δ : get_preconditioner(p, o.x, o.δ))
    e_Pe = inner(p.M, o.x, o.η, o.randomize ? o.η : get_preconditioner(p, o.x, o.η))
    e_Pe_new = e_Pe + 2α * e_Pd + α^2 * d_Pd
    # Check against negative curvature and trust-region radius violation.
    # If either condition triggers, we bail out.
    if o.δHδ <= 0 || e_Pe_new >= o.trust_region_radius^2
        τ = (-e_Pd + sqrt(e_Pd^2 + d_Pd * (o.trust_region_radius^2 - e_Pe))) / d_Pd
        o.η = o.η - τ * (o.δ)
    else
        # No negative curvature and o.η - α * (o.δ) inside TR: accept it.
        new_model_value =
            inner(p.M, o.x, o.η - α * (o.δ), get_gradient(p, o.x)) +
            0.5 * inner(p.M, o.x, o.η - α * (o.δ), get_hessian(p, o.x, o.η - α * (o.δ)))
        if new_model_value <= o.model_value
            o.η = o.η - α * (o.δ)
            o.model_value = new_model_value
        end
    end
    # Updates
    get_hessian!(p, o.Hη, o.x, o.η)
    o.residual = o.residual - α * o.Hδ
    # Precondition the residual.
    o.precon_residual = o.randomize ? o.residual : get_preconditioner(p, o.x, o.residual)
    zr = inner(p.M, o.x, o.precon_residual, o.residual)
    # Compute new search direction.
    β = zr / o.res_precon_res
    o.res_precon_res = zr
    o.δ = project!(p.M, o.δ, o.x, o.precon_residual + β * o.δ)
    return o
end
get_solver_result(o::O) where {O<:TruncatedConjugateGradientOptions} = o.η
