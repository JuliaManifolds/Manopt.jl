@doc raw"""
    trust_regions(M, F, ∇F, x, H)

evaluate the Riemannian trust-regions solver for optimization on manifolds.
It will attempt to minimize the cost function F on the Manifold M.
If no Hessian H is provided, a standard approximation of the Hessian based on
the gradient ∇F will be computed.
For solving the the inner trust-region subproblem of finding an update-vector,
it uses the Steihaug-Toint truncated conjugate-gradient method.
For a description of the algorithm and more details see

* P.-A. Absil, C.G. Baker, K.A. Gallivan,
    Trust-region methods on Riemannian manifolds, FoCM, 2007.
    doi: [10.1007/s10208-005-0179-9](https://doi.org/10.1007/s10208-005-0179-9)
* A. R. Conn, N. I. M. Gould, P. L. Toint, Trust-region methods, SIAM,
    MPS, 2000. doi: [10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857)

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F \colon \mathcal M \to \mathbb R$ to minimize
* `∇F`- the gradient $\nabla F \colon \mathcal M \to T \mathcal M$ of $F$
* `x` – an initial value $x  ∈  \mathcal M$
* `H` – the hessian $H( \mathcal M, x, \xi)$ of $F$

# Optional
* `retraction` – approximation of the exponential map
* `preconditioner` – a preconditioner (a symmetric, positive definite operator
  that should approximate the inverse of the Hessian)
* `stopping_criterion` – ([`StopWhenAny`](@ref)([`StopAfterIteration`](@ref)`(1000)`,
  [`StopWhenGradientNormLess`](@ref)`(10^(-6))`) a functor inheriting
  from [`StoppingCriterion`](@ref) indicating when to stop.
* `Δ_bar` – the maximum trust-region radius
* `Δ` - the (initial) trust-region radius
* `useRandom` – set to true if the trust-region solve is to be initiated with a
  random tangent vector. If set to true, no preconditioner will be
  used. This option is set to true in some scenarios to escape saddle
  points, but is otherwise seldom activated.
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
* `return_options` – (`false`) – if actiavated, the extended result, i.e. the
  complete [`Options`](@ref) are returned. This can be used to access recorded values.
  If set to false (default) just the optimal value `x_opt` is returned

# Output
* `x` – the last reached point on the manifold

# see also
[`truncated_conjugate_gradient_descent`](@ref)
"""
function trust_regions(M::Manifold, F::TF, ∇F::TdF, x, H::TH; kwargs...) where {TF,TdF,TH}
    x_res = allocate(x)
    copyto!(x_res, x)
    return trust_regions!(M, F, ∇F, x_res, H; kwargs...)
end
@doc raw"""
    trust_regions!(M, F, ∇F, x, H; kwargs...)

evaluate the Riemannian trust-regions solver for optimization on manifolds in place of `x`.

# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F \colon \mathcal M \to \mathbb R$ to minimize
* `∇F`- the gradient $\nabla F \colon \mathcal M \to T \mathcal M$ of $F$
* `x` – an initial value $x  ∈  \mathcal M$
* `H` – the hessian $H( \mathcal M, x, \xi)$ of $F$

for more details and all options, see [`trust_regions`](@ref)
"""
function trust_regions!(
    M::Manifold,
    F::TF,
    ∇F::TdF,
    x,
    H::TH;
    retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
    preconditioner::Tprec=(M, x, ξ) -> ξ,
    stopping_criterion::StoppingCriterion=StopWhenAny(
        StopAfterIteration(1000), StopWhenGradientNormLess(10^(-6))
    ),
    Δ_bar=sqrt(manifold_dimension(M)),
    Δ=Δ_bar / 8,
    useRandom::Bool=false,
    ρ_prime::Float64=0.1,
    ρ_regularization=1000.0,
    return_options=false,
    kwargs..., #collect rest
) where {TF,TdF,TH,Tprec}
    (ρ_prime >= 0.25) && throw(
        ErrorException("ρ_prime must be strictly smaller than 0.25 but it is $ρ_prime.")
    )
    (Δ_bar <= 0) && throw(ErrorException("Δ_bar must be positive but it is $Δ_bar."))
    (Δ <= 0 || Δ > Δ_bar) && throw(
        ErrorException("Δ must be positive and smaller than Δ_bar (=$Δ_bar) but it is $Δ."),
    )
    p = HessianProblem(M, F, ∇F, H, preconditioner)
    o = TrustRegionsOptions(
        x,
        stopping_criterion,
        Δ,
        Δ_bar,
        retraction_method,
        useRandom,
        ρ_prime,
        ρ_regularization,
    )
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end

initialize_solver!(p::P, o::O) where {P<:HessianProblem,O<:TrustRegionsOptions} = nothing

function step_solver!(p::P, o::O, iter) where {P<:HessianProblem,O<:TrustRegionsOptions}
    # Determine eta0
    if o.useRand == false
        # Pick the zero vector
        eta = zero_tangent_vector(p.M, o.x)
    else
        # Random vector in T_x M (this has to be very small)
        eta = 10.0^(-6) * random_tangent(p.M, o.x)
        while norm(p.M, o.x, eta) > o.Δ
            # Must be inside trust-region
            eta = sqrt(sqrt(eps(Float64))) * eta
        end
    end
    # Solve TR subproblem approximately
    opt = truncated_conjugate_gradient_descent(
        p.M,
        p.cost,
        p.gradient,
        o.x,
        eta,
        p.hessian,
        o.Δ;
        preconditioner=p.precon,
        useRandom=o.useRand,
        #debug = [:Iteration," ",:Stop],
        return_options=true,
    )
    option = get_options(opt) # remove decorators
    η = get_solver_result(option)
    SR = get_active_stopping_criteria(option.stop)
    Hη = getHessian(p, o.x, η)
    # Initialize the cost function F und the gradient of the cost function
    # ∇F at the point x
    grad = get_gradient(p, o.x)
    fx = get_cost(p, o.x)
    norm_grad = norm(p.M, o.x, grad)
    # If using randomized approach, compare result with the Cauchy point.
    if o.useRand
        # Check the curvature,
        Hgrad = getHessian(p, o.x, grad)
        gradHgrad = inner(p.M, o.x, grad, Hgrad)
        if gradHgrad <= 0
            tau_c = 1
        else
            tau_c = min(norm_grad^3 / (o.Δ * gradHgrad), 1)
        end
        # and generate the Cauchy point.
        η_c = (-tau_c * o.Δ / norm_grad) * grad
        Hη_c = (-tau_c * o.Δ / norm_grad) * Hgrad
        # Now that we have computed the Cauchy point in addition to the
        # returned eta, we might as well keep the best of them.
        mdle = fx + inner(p.M, o.x, grad, η) + 0.5 * inner(p.M, o.x, Hη, η)
        mdlec = fx + inner(p.M, o.x, grad, η_c) + 0.5 * inner(p.M, o.x, Hη_c, η_c)
        if mdlec < mdle
            η = η_c
            Hη = Hη_c
        end
    end
    # Compute the tentative next iterate (the proposal)
    x_prop = retract(p.M, o.x, η, o.retraction_method)
    # Compute the function value of the proposal
    fx_prop = get_cost(p, x_prop)
    # Check the performance of the quadratic model against the actual cost.
    ρnum = fx - fx_prop
    ρden = -inner(p.M, o.x, η, grad) - 0.5 * inner(p.M, o.x, η, Hη)
    # Since, at convergence, both ρnum and ρden become extremely small,
    # computing ρ is numerically challenging. The break with ρnum and ρden
    # can thus lead to a large error in rho, making the
    # acceptance / rejection erratic. Meanwhile, close to convergence,
    # steps are usually trustworthy and we should transition to a Newton-
    # like method, with rho=1 consistently. The heuristic thus shifts both
    # rhonum and rhoden by a small amount such that far from convergence,
    # the shift is irrelevant and close to convergence, the ratio rho goes
    # to 1, effectively promoting acceptance of the step.
    ρ_reg = max(1, abs(fx)) * eps(Float64) * o.ρ_regularization
    ρnum = ρnum + ρ_reg
    ρden = ρden + ρ_reg
    model_decreased = (ρden >= 0)
    ρ = ρnum / ρden
    # Choose the new TR radius based on the model performance.
    # If the actual decrease is smaller than 1/4 of the predicted decrease,
    # then reduce the TR radius.
    if ρ < 1 / 4 || model_decreased == false || isnan(ρ)
        o.Δ = o.Δ / 4
    elseif ρ > 3 / 4 && any([
        typeof(s) in [StopWhenTrustRegionIsExceeded, StopWhenCurvatureIsNegative] for
        s in SR
    ])
        o.Δ = min(2 * o.Δ, o.Δ_bar)
    else
        o.Δ = o.Δ
    end
    # Choose to accept or reject the proposed step based on the model
    # performance. Note the strict inequality.
    if model_decreased && ρ > o.ρ_prime
        o.x = x_prop
    end
    return nothing
end
get_solver_result(o::O) where {O<:TrustRegionsOptions} = o.x
