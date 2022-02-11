@doc raw"""
    trust_regions(M, F, gradF, hessF, x)

evaluate the Riemannian trust-regions solver for optimization on manifolds.
It will attempt to minimize the cost function F on the Manifold M.
If no Hessian H is provided, a standard approximation of the Hessian based on
the gradient `gradF` will be computed.
For solving the the inner trust-region subproblem of finding an update-vector,
it uses the Steihaug-Toint truncated conjugate-gradient method.
For a description of the algorithm and more details see

* P.-A. Absil, C.G. Baker, K.A. Gallivan,
    Trust-region methods on Riemannian manifolds, FoCM, 2007.
    doi: [10.1007/s10208-005-0179-9](https://doi.org/10.1007/s10208-005-0179-9)
* A. R. Conn, N. I. M. Gould, P. L. Toint, Trust-region methods, SIAM,
    MPS, 2000. doi: [10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857)

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F : \mathcal M → ℝ`` to minimize
* `gradF`- the gradient ``\operatorname{grad}F : \mathcal M → T \mathcal M`` of ``F``
* `x` – an initial value ``x  ∈  \mathcal M``
* `HessF` – the hessian ``\operatorname{Hess}F(x): T_x\mathcal M → T_x\mathcal M``, ``X ↦ \operatoname{Hess}F(x)[X] = ∇_ξ\operatorname{grad}f(x)``

# Optional
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient and hessian work by
   allocation (default) or [`MutatingEvaluation`](@ref) in place
* `max_trust_region_radius` – the maximum trust-region radius
* `preconditioner` – a preconditioner (a symmetric, positive definite operator
  that should approximate the inverse of the Hessian)
* `randomize` – set to true if the trust-region solve is to be initiated with a
  random tangent vector. If set to true, no preconditioner will be
  used. This option is set to true in some scenarios to escape saddle
  points, but is otherwise seldom activated.
* `project_vector!` : (`copyto!`) specify a projection operation for tangent vectors within the TCG
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.
* `retraction` – (`default_retraction_method(M)`) approximation of the exponential map
* `stopping_criterion` – ([`StopWhenAny`](@ref)([`StopAfterIteration`](@ref)`(1000)`,
  [`StopWhenGradientNormLess`](@ref)`(10^(-6))`) a functor inheriting
  from [`StoppingCriterion`](@ref) indicating when to stop.
* `trust_region_radius` - the initial trust-region radius
* `ρ_prime` – Accept/reject threshold: if ρ (the performance ratio for the
  iterate) is at least ρ', the outer iteration is accepted.
  Otherwise, it is rejected. In case it is rejected, the trust-region
  radius will have been decreased. To ensure this, ρ' >= 0 must be
  strictly smaller than 1/4. If ρ_prime is negative, the algorithm is not
  guaranteed to produce monotonically decreasing cost values. It is
  strongly recommended to set ρ' > 0, to aid convergence.
* `ρ_regularization` – Close to convergence, evaluating the performance ratio ρ
  is numerically challenging. Meanwhile, close to convergence, the
  quadratic model should be a good fit and the steps should be
  accepted. Regularization lets ρ go to 1 as the model decrease and
  the actual decrease go to zero. Set this option to zero to disable
  regularization (not recommended). When this is not zero, it may happen
  that the iterates produced are not monotonically improving the cost
  when very close to convergence. This is because the corrected cost
  improvement could change sign if it is negative but very small.
* `return_options` – (`false`) – if activated, the extended result, i.e. the
  complete [`Options`](@ref) are returned. This can be used to access recorded values.
  If set to false (default) just the optimal value `x_opt` is returned

# Output
* `x` – the last reached point on the manifold

# see also
[`truncated_conjugate_gradient_descent`](@ref)
"""
function trust_regions(
    M::AbstractManifold, F::TF, gradF::TdF, hessF::TH, x; kwargs...
) where {TF,TdF,TH}
    x_res = allocate(x)
    copyto!(M, x_res, x)
    return trust_regions!(M, F, gradF, hessF, x_res; kwargs...)
end
@doc raw"""
    trust_regions!(M, F, gradF, hessF, x; kwargs...)

evaluate the Riemannian trust-regions solver for optimization on manifolds in place of `x`.

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F: \mathcal M → ℝ`` to minimize
* `gradF`- the gradient ``\operatorname{grad}F: \mathcal M → T \mathcal M`` of ``F``
* `x` – an initial value ``x  ∈  \mathcal M``
* `H` – the hessian ``H( \mathcal M, x, ξ)`` of ``F``

for more details and all options, see [`trust_regions`](@ref)
"""
function trust_regions!(
    M::AbstractManifold,
    F::TF,
    gradF::TdF,
    hessF::TH,
    x;
    evaluation=AllocatingEvaluation(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    preconditioner::Tprec=(M, x, ξ) -> ξ,
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(1000), StopWhenGradientNormLess(10^(-6))
    ),
    max_trust_region_radius=sqrt(manifold_dimension(M)),
    trust_region_radius=max_trust_region_radius / 8,
    randomize::Bool=false,
    project_vector!::Proj=copyto!,
    ρ_prime::Float64=0.1,
    ρ_regularization=1000.0,
    return_options=false,
    kwargs..., #collect rest
) where {TF,TdF,TH,Tprec,Proj}
    (ρ_prime >= 0.25) && throw(
        ErrorException("ρ_prime must be strictly smaller than 0.25 but it is $ρ_prime.")
    )
    (max_trust_region_radius <= 0) && throw(
        ErrorException(
            "max_trust_region_radius must be positive but it is $max_trust_region_radius.",
        ),
    )
    (trust_region_radius <= 0 || trust_region_radius > max_trust_region_radius) && throw(
        ErrorException(
            "trust_region_radius must be positive and smaller than max_trust_region_radius (=$max_trust_region_radius) but it is $trust_region_radius.",
        ),
    )
    p = HessianProblem(M, F, gradF, hessF, preconditioner; evaluation=evaluation)
    o = TrustRegionsOptions(
        x,
        get_gradient(p, x),
        trust_region_radius,
        max_trust_region_radius,
        ρ_prime,
        ρ_regularization,
        randomize,
        stopping_criterion;
        retraction_method=retraction_method,
        (project_vector!)=project_vector!,
    )
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end

function initialize_solver!(p::HessianProblem, o::TrustRegionsOptions)
    get_gradient!(p, o.gradient, o.x)
    o.η = zero_vector(p.M, o.x)
    o.Hη = zero_vector(p.M, o.x)
    o.x_proposal = deepcopy(o.x)
    o.f_proposal = zero(o.trust_region_radius)

    o.η_Cauchy = zero_vector(p.M, o.x)
    o.Hη_Cauchy = zero_vector(p.M, o.x)
    o.τ = zero(o.trust_region_radius)
    o.Hgrad = zero_vector(p.M, o.x)
    o.tcg_options = TruncatedConjugateGradientOptions(
        p, o.x, o.η, o.trust_region_radius, o.randomize; (project_vector!)=o.project_vector!
    )
    return o
end

function step_solver!(p::HessianProblem, o::TrustRegionsOptions, iter)
    # Determine eta0
    if o.randomize
        # Random vector in T_x M (this has to be very small)
        o.η = random_tangent(p.M, o.x, 10.0^(-6))
        while norm(p.M, o.x, o.η) > o.trust_region_radius
            # inside trust-region
            o.η *= sqrt(sqrt(eps(Float64)))
        end
    else
        zero_vector!(p.M, o.η, o.x)
    end
    # Solve TR subproblem - update options
    o.tcg_options.x = o.x
    o.tcg_options.η = o.η
    o.tcg_options.trust_region_radius = o.trust_region_radius
    solve(p, o.tcg_options)
    #
    o.η = o.tcg_options.η
    o.Hη = o.tcg_options.Hη

    # Initialize the cost function F und the gradient of the cost function
    # gradF at the point x
    o.gradient = o.tcg_options.gradient
    fx = get_cost(p, o.x)
    # If using randomized approach, compare result with the Cauchy point.
    if o.randomize
        norm_grad = norm(p.M, o.x, o.gradient)
        # Check the curvature,
        get_hessian!(p, o.Hgrad, o.x, o.gradient)
        o.τ = inner(p.M, o.x, o.gradient, o.Hgrad)
        o.τ = (o.τ <= 0) ? one(o.τ) : min(norm_grad^3 / (o.trust_region_radius * o.τ), 1)
        # compare to Cauchy point and store best
        model_value =
            fx + inner(p.M, o.x, o.gradient, o.η) + 0.5 * inner(p.M, o.x, o.Hη, o.η)
        modle_value_Cauchy = fx
        -o.τ * o.trust_region_radius * norm(p.M, o.x, o.gradient)
        +0.5 * o.τ^2 * o.trust_region_radius^2 / (norm_grad^2) *
        inner(p.M, o.x, o.Hgrad, o.gradient)
        if modle_value_Cauchy < model_value
            copyto!(p.M, o.η, (-o.τ * o.trust_region_radius / norm_grad) * o.gradient)
            copyto!(p.M, o.Hη, (-o.τ * o.trust_region_radius / norm_grad) * o.Hgrad)
        end
    end
    # Compute the tentative next iterate (the proposal)
    retract!(p.M, o.x_proposal, o.x, o.η, o.retraction_method)
    # Check the performance of the quadratic model against the actual cost.
    ρ_reg = max(1, abs(fx)) * eps(Float64) * o.ρ_regularization
    ρnum = fx - get_cost(p, o.x_proposal)
    ρden = -inner(p.M, o.x, o.η, o.gradient) - 0.5 * inner(p.M, o.x, o.η, o.Hη)
    ρnum = ρnum + ρ_reg
    ρden = ρden + ρ_reg
    ρ = ρnum / ρden
    model_decreased = ρden ≥ 0
    # Choose the new TR radius based on the model performance.
    # If the actual decrease is smaller than 1/4 of the predicted decrease,
    # then reduce the TR radius.
    if ρ < 1 / 4 || !model_decreased || isnan(ρ)
        o.trust_region_radius /= 4
    elseif ρ > 3 / 4 &&
        ((o.tcg_options.ηPη >= o.trust_region_radius^2) || (o.tcg_options.δHδ <= 0))
        o.trust_region_radius = min(2 * o.trust_region_radius, o.max_trust_region_radius)
    end
    # Choose to accept or reject the proposed step based on the model
    # performance. Note the strict inequality.
    if model_decreased && ρ > o.ρ_prime
        copyto!(p.M, o.x, o.x_proposal)
    end
    return o
end
get_solver_result(o::TrustRegionsOptions) = o.x
