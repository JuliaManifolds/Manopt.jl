
@doc raw"""
    TrustRegionsState <: AbstractHessianSolverState

describe the trust-regions solver, with


# Fields
where all but `x` are keyword arguments in the constructor

* `p` : the current iterate
* `stop` : (`StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6))
* `max_trust_region_radius` : (`sqrt(manifold_dimension(M))`) the maximum trust-region radius
* `project!` : (`copyto!`) specify a projection operation for tangent vectors
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.
* `randomize` : (`false`) indicates if the trust-region solve is to be initiated with a
        random tangent vector. If set to true, no preconditioner will be
        used. This option is set to true in some scenarios to escape saddle
        points, but is otherwise seldom activated.
* `ρ_prime` : (`0.1`) a lower bound of the performance ratio for the iterate that
        decides if the iteration will be accepted or not. If not, the
        trust-region radius will have been decreased. To ensure this,
        ρ'>= 0 must be strictly smaller than 1/4. If ρ' is negative,
        the algorithm is not guaranteed to produce monotonically decreasing
        cost values. It is strongly recommended to set ρ' > 0, to aid
        convergence.
* `ρ_regularization` : (`10000.0`) Close to convergence, evaluating the performance ratio ρ
        is numerically challenging. Meanwhile, close to convergence, the
        quadratic model should be a good fit and the steps should be
        accepted. Regularization lets ρ go to 1 as the model decrease and
        the actual decrease go to zero. Set this option to zero to disable
        regularization (not recommended). When this is not zero, it may happen
        that the iterates produced are not monotonically improving the cost
        when very close to convergence. This is because the corrected cost
        improvement could change sign if it is negative but very small.
* `trust_region_radius` : the (initial) trust-region radius

# Constructor

    TrustRegionsState(M, p)

construct a trust-regions Option with all other fields from above being
keyword arguments

# See also
[`trust_regions`](@ref)
"""
mutable struct TrustRegionsState{
    P,T,SC<:StoppingCriterion,RTR<:AbstractRetractionMethod,R<:Real,Proj
} <: AbstractHessianSolverState
    p::P
    gradient::T
    stop::SC
    trust_region_radius::R
    max_trust_region_radius::R
    retraction_method::RTR
    randomize::Bool
    project!::Proj
    ρ_prime::R
    ρ_regularization::R

    tcg_options::TruncatedConjugateGradientState{P,T,R}

    x_proposal::P
    f_proposal::R
    # Random
    Hgrad::T
    η::T
    Hη::T
    η_Cauchy::T
    Hη_Cauchy::T
    τ::R
    function TrustRegionsState{P,T,SC,RTR,R,Proj}(
        p::P,
        grad::T,
        trust_region_radius::R,
        max_trust_region_radius::R,
        ρ_prime::R,
        ρ_regularization::R,
        randomize::Bool,
        stopping_citerion::SC,
        retraction_method::RTR,
        project!::Proj=copyto!,
    ) where {P,T,SC<:StoppingCriterion,RTR<:AbstractRetractionMethod,R<:Real,Proj}
        trs = new{P,T,SC,RTR,R,Proj}()
        trs.p = p
        trs.gradient = grad
        trs.stop = stopping_citerion
        trs.retraction_method = retraction_method
        trs.trust_region_radius = trust_region_radius
        trs.max_trust_region_radius = max_trust_region_radius::R
        trs.ρ_prime = ρ_prime
        trs.ρ_regularization = ρ_regularization
        trs.randomize = randomize
        trs.project! = project!
        return trs
    end
end
function TrustRegionsState(
    M::TM,
    p::P;
    X::T=zero_vector(M, x),
    ρ_prime::R=0.1,
    ρ_regularization::R=1000.0,
    randomize::Bool=false,
    stopping_criterion::SC=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
    max_trust_region_radius::R=sqrt(manifold_dimension(M)),
    trust_region_radius::R=max_trust_region_radius / 8,
    retraction_method::RTR=default_retraction_method(M),
    project!::Proj=copyto!,
) where {
    TM<:AbstractManifold,
    P,
    T,
    R<:Real,
    SC<:StoppingCriterion,
    RTR<:AbstractRetractionMethod,
    Proj,
}
    return TrustRegionsState{P,T,SC,RTR,R,Proj}(
        p,
        X,
        trust_region_radius,
        max_trust_region_radius,
        ρ_prime,
        ρ_regularization,
        randomize,
        stopping_criterion,
        retraction_method,
        project!,
    )
end

@doc raw"""
    trust_regions(M, f, grad_f, hess_f, p)
    trust_regions(M, f, grad_f, p)

run the Riemannian trust-regions solver for optimization on manifolds to minmize `f`.

For the case that no hessian is provided, the Hessian is computed using finite difference, see
[`ApproxHessianFiniteDifference`](@ref).
For solving the the inner trust-region subproblem of finding an update-vector,
see [`truncated_conjugate_gradient_descent`](@ref).

* P.-A. Absil, C.G. Baker, K.A. Gallivan,
    Trust-region methods on Riemannian manifolds, FoCM, 2007.
    doi: [10.1007/s10208-005-0179-9](https://doi.org/10.1007/s10208-005-0179-9)
* A. R. Conn, N. I. M. Gould, P. L. Toint, Trust-region methods, SIAM,
    MPS, 2000. doi: [10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857)

# Input
* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``F : \mathcal M → ℝ`` to minimize
* `grad_f`- the gradient ``\operatorname{grad}F : \mathcal M → T \mathcal M`` of ``F``
* `Hess_f` – (optional), the hessian ``\operatorname{Hess}F(x): T_x\mathcal M → T_x\mathcal M``, ``X ↦ \operatorname{Hess}F(x)[X] = ∇_ξ\operatorname{grad}f(x)``
* `p` – an initial value ``x  ∈  \mathcal M``

# Optional
* `evaluation` – ([`AllocatingEvaluation`](@ref)) specify whether the gradient and hessian work by
   allocation (default) or [`InplaceEvaluation`](@ref) in place
* `max_trust_region_radius` – the maximum trust-region radius
* `preconditioner` – a preconditioner (a symmetric, positive definite operator
  that should approximate the inverse of the Hessian)
* `randomize` – set to true if the trust-region solve is to be initiated with a
  random tangent vector. If set to true, no preconditioner will be
  used. This option is set to true in some scenarios to escape saddle
  points, but is otherwise seldom activated.
* `project!` : (`copyto!`) specify a projection operation for tangent vectors within the TCG
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

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details

# see also
[`truncated_conjugate_gradient_descent`](@ref)
"""
function trust_regions(
    M::AbstractManifold, f::TF, grad_f::TdF, Hess_f::TH, p; kwargs...
) where {TF,TdF,TH}
    q = copy(M, p)
    return trust_regions!(M, f, grad_f, hess_f, q; kwargs...)
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
    preconditioner::Tprec=evaluation isa InplaceEvaluation ? (M, Y, p, X) -> (Y .= X) : (M, p, X) -> X,
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    stopping_criterion::StoppingCriterion=StopAfterIteration(1000) |
                                          StopWhenGradientNormLess(1e-6),
    max_trust_region_radius=sqrt(manifold_dimension(M)),
    trust_region_radius=max_trust_region_radius / 8,
    randomize::Bool=false,
    project!::Proj=copyto!,
    ρ_prime::Float64=0.1,
    ρ_regularization=1000.0,
    θ::Float64=1.0,
    κ::Float64=0.1,
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
    mho = ManifoldHessianObjective(F, gradF, hessF, preconditioner; evaluation=evaluation)
    mp = DefaultManoptProblem(M, mho)
    trs = TrustRegionsState(
        M,
        x;
        gradient=get_gradient(p, x),
        trust_region_radius=trust_region_radius,
        max_trust_region_radius=max_trust_region_radius,
        ρ_prime=ρ_prime,
        ρ_regularization=ρ_regularization,
        randomize=randomize,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        (project!)=project!,
        θ=θ,
        κ=κ,
    )
    trs = decorate_state(trs; kwargs...)
    return get_solver_return(solve!(p, trs))
end

function initialize_solver!(mp::AbstractManoptProblem, trs::TrustRegionsState)
    get_gradient!(mp, trs.gradient, trs.x)
    trs.η = zero_vector(mp.M, trs.x)
    trs.Hη = zero_vector(mp.M, trs.x)
    trs.x_proposal = deepcopy(trs.x)
    trs.f_proposal = zero(trs.trust_region_radius)

    trs.η_Cauchy = zero_vector(mp.M, trs.x)
    trs.Hη_Cauchy = zero_vector(mp.M, trs.x)
    trs.τ = zero(trs.trust_region_radius)
    trs.Hgrad = zero_vector(mp.M, trs.x)
    trs.tcg_options = TruncatedConjugateGradientState(
        mp.M,
        trs.x,
        trs.η;
        trust_region_radius=trs.trust_region_radius,
        randomize=trs.randomize,
        (project!)=trs.project!,
    )
    return trs
end

function step_solver!(mp::AbstractManoptProblem, trs::TrustRegionsState, i)
    M = get_manifold(mp)
    mho = get_objective(mp)
    # Determine eta0
    if trs.randomize
        # Random vector in T_x M (this has to be very small)
        trs.η = random_tangent(M, trs.x, 10.0^(-6))
        while norm(M, trs.x, trs.η) > trs.trust_region_radius
            # inside trust-region
            trs.η *= sqrt(sqrt(eps(Float64)))
        end
    else
        zero_vector!(M, trs.η, trs.x)
    end
    # Solve TR subproblem - update options
    trs.tcg_options.x = trs.x
    trs.tcg_options.η = trs.η
    trs.tcg_options.trust_region_radius = trs.trust_region_radius
    trs.tcg_options.stop = StopWhenAny(
        StopAfterIteration(manifold_dimension(M)),
        StopWhenAll(
            StopIfResidualIsReducedByPower(o.θ), StopIfResidualIsReducedByFactor(o.κ)
        ),
        StopWhenTrustRegionIsExceeded(),
        StopWhenCurvatureIsNegative(),
        StopWhenModelIncreased(),
    )
    solve!(mp, trs.tcg_options)
    #
    trs.η = trs.tcg_options.η
    trs.Hη = trs.tcg_options.Hη

    # Initialize the cost function F und the gradient of the cost function
    # gradF at the point x
    trs.gradient = trs.tcg_options.gradient
    fx = get_cost(mp, trs.x)
    # If using randomized approach, compare result with the Cauchy point.
    if trs.randomize
        norm_grad = norm(M, trs.x, trs.gradient)
        # Check the curvature,
        get_hessian!(mp, trs.Hgrad, trs.x, trs.gradient)
        trs.τ = inner(M, trs.x, trs.gradient, trs.Hgrad)
        trs.τ = (trs.τ <= 0) ? one(trs.τ) : min(norm_grad^3 / (trs.trust_region_radius * trs.τ), 1)
        # compare to Cauchy point and store best
        model_value = fx + inner(M, trs.x, trs.gradient, trs.η) + 0.5 * inner(M, trs.x, trs.Hη, trs.η)
        modle_value_Cauchy = fx
        -trs.τ * trs.trust_region_radius * norm_grad
        +0.5 * trs.τ^2 * trs.trust_region_radius^2 / (norm_grad^2) *
        inner(M, trs.x, trs.Hgrad, trs.gradient)
        if modle_value_Cauchy < model_value
            copyto!(M, trs.η, (-trs.τ * trs.trust_region_radius / norm_grad) * trs.gradient)
            copyto!(M, trs.Hη, (-trs.τ * trs.trust_region_radius / norm_grad) * trs.Hgrad)
        end
    end
    # Compute the tentative next iterate (the proposal)
    retract!(M, trs.x_proposal, trs.x, trs.η, trs.retraction_method)
    # Check the performance of the quadratic model against the actual cost.
    ρ_reg = max(1, abs(fx)) * eps(Float64) * trs.ρ_regularization
    ρnum = fx - get_cost(mp, trs.x_proposal)
    ρden = -inner(M, trs.x, trs.η, trs.gradient) - 0.5 * inner(M, trs.x, trs.η, trs.Hη)
    ρnum = ρnum + ρ_reg
    ρden = ρden + ρ_reg
    ρ = (abs(ρnum / fx) < sqrt(eps(Float64))) ? 1 : ρnum / ρden # stability for small absolute relative model change

    model_decreased = ρden ≥ 0
    # Update the Hessian approximation
    update_hessian!(M, mho.hessian!!, o.x, o.x_proposal, o.η)
    # Choose the new TR radius based on the model performance.
    # If the actual decrease is smaller than 1/4 of the predicted decrease,
    # then reduce the TR radius.
    if ρ < 0.1 || !model_decreased || isnan(ρ)
        trs.trust_region_radius /= 4
    elseif ρ > 3 / 4 &&
        ((trs.tcg_options.ηPη >= trs.trust_region_radius^2) || (trs.tcg_options.δHδ <= 0))
        trs.trust_region_radius = min(2 * trs.trust_region_radius, trs.max_trust_region_radius)
    end
    # Choose to accept or reject the proposed step based on the model
    # performance. Note the strict inequality.
    if model_decreased &&
        (ρ > trs.ρ_prime || (abs((ρnum) / (abs(fx) + 1)) < sqrt(eps(Float64)) && 0 < ρnum))
        copyto!(trs.x, trs.x_proposal)
        update_hessian_basis!(M, mho.hessian!!, trs.x)
    end
    return trs
end
