
@doc raw"""
    TrustRegionsState <: AbstractHessianSolverState

describe the trust-regions solver, with


# Fields
where all but `x` are keyword arguments in the constructor

* `x` : a point as starting point
* `stop` : (`StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6))
* `trust_region_radius` : the (initial) trust-region radius
* `max_trust_region_radius` : (`sqrt(manifold_dimension(M))`) the maximum trust-region radius
* `randomize` : (`false`) indicates if the trust-region solve is to be initiated with a
        random tangent vector. If set to true, no preconditioner will be
        used. This option is set to true in some scenarios to escape saddle
        points, but is otherwise seldom activated.
* `project!` : (`copyto!`) specify a projection operation for tangent vectors
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.
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

# Constructor

    TrustRegionsState(M, x)

construct a trust-regions Option with all other fields from above being
keyword arguments

# See also
[`trust_regions`](@ref)
"""
mutable struct TrustRegionsState{
    P,T,SC<:StoppingCriterion,RTR<:AbstractRetractionMethod,R<:Real,Proj
} <: AbstractHessianSolverState
    x::P
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
        x::P,
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
        o.project! = project!
        return o
    end
end
function TrustRegionsState(
    M::TM,
    x::P;
    gradient::T=zero_vector(M, x),
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
        x,
        gradient,
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
* `HessF` – the hessian ``\operatorname{Hess}F(x): T_x\mathcal M → T_x\mathcal M``, ``X ↦ \operatorname{Hess}F(x)[X] = ∇_ξ\operatorname{grad}f(x)``

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

the obtained (approximate) minimizer ``x^*``, see [`get_solver_return`](@ref) for details

# see also
[`truncated_conjugate_gradient_descent`](@ref)
"""
function trust_regions(
    M::AbstractManifold, F::TF, gradF::TdF, hessF::TH, x; kwargs...
) where {TF,TdF,TH}
    x_res = copy(M, x)
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
    p = HessianProblem(M, F, gradF, hessF, preconditioner; evaluation=evaluation)
    o = TrustRegionsState(
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
    o = decorate_state(o; kwargs...)
    return get_solver_return(solve!(p, o))
end

function initialize_solver!(p::HessianProblem, s::TrustRegionsState)
    get_gradient!(p, s.gradient, s.x)
    s.η = zero_vector(p.M, s.x)
    s.Hη = zero_vector(p.M, s.x)
    s.x_proposal = deepcopy(s.x)
    s.f_proposal = zero(s.trust_region_radius)

    s.η_Cauchy = zero_vector(p.M, s.x)
    s.Hη_Cauchy = zero_vector(p.M, s.x)
    s.τ = zero(s.trust_region_radius)
    s.Hgrad = zero_vector(p.M, s.x)
    s.tcg_options = TruncatedConjugateGradientState(
        p.M,
        s.x,
        s.η;
        trust_region_radius=s.trust_region_radius,
        randomize=s.randomize,
        (project!)=s.project!,
    )
    return s
end

function step_solver!(p::HessianProblem, s::TrustRegionsState, ::Any)
    # Determine eta0
    if s.randomize
        # Random vector in T_x M (this has to be very small)
        s.η = random_tangent(p.M, s.x, 10.0^(-6))
        while norm(p.M, s.x, s.η) > s.trust_region_radius
            # inside trust-region
            s.η *= sqrt(sqrt(eps(Float64)))
        end
    else
        zero_vector!(p.M, s.η, s.x)
    end
    # Solve TR subproblem - update options
    s.tcg_options.x = s.x
    s.tcg_options.η = s.η
    s.tcg_options.trust_region_radius = s.trust_region_radius
    s.tcg_options.stop = StopWhenAny(
        StopAfterIteration(manifold_dimension(p.M)),
        StopWhenAll(
            StopIfResidualIsReducedByPower(o.θ), StopIfResidualIsReducedByFactor(o.κ)
        ),
        StopWhenTrustRegionIsExceeded(),
        StopWhenCurvatureIsNegative(),
        StopWhenModelIncreased(),
    )
    solve!(p, s.tcg_options)
    #
    s.η = s.tcg_options.η
    s.Hη = s.tcg_options.Hη

    # Initialize the cost function F und the gradient of the cost function
    # gradF at the point x
    s.gradient = s.tcg_options.gradient
    fx = get_cost(p, s.x)
    # If using randomized approach, compare result with the Cauchy point.
    if s.randomize
        norm_grad = norm(p.M, s.x, s.gradient)
        # Check the curvature,
        get_hessian!(p, s.Hgrad, s.x, s.gradient)
        s.τ = inner(p.M, s.x, s.gradient, s.Hgrad)
        s.τ = (s.τ <= 0) ? one(s.τ) : min(norm_grad^3 / (s.trust_region_radius * s.τ), 1)
        # compare to Cauchy point and store best
        model_value =
            fx + inner(p.M, s.x, s.gradient, s.η) + 0.5 * inner(p.M, s.x, s.Hη, s.η)
        modle_value_Cauchy = fx
        -s.τ * s.trust_region_radius * norm_grad
        +0.5 * s.τ^2 * s.trust_region_radius^2 / (norm_grad^2) *
        inner(p.M, s.x, s.Hgrad, s.gradient)
        if modle_value_Cauchy < model_value
            copyto!(p.M, s.η, (-s.τ * s.trust_region_radius / norm_grad) * s.gradient)
            copyto!(p.M, s.Hη, (-s.τ * s.trust_region_radius / norm_grad) * s.Hgrad)
        end
    end
    # Compute the tentative next iterate (the proposal)
    retract!(p.M, s.x_proposal, s.x, s.η, s.retraction_method)
    # Check the performance of the quadratic model against the actual cost.
    ρ_reg = max(1, abs(fx)) * eps(Float64) * s.ρ_regularization
    ρnum = fx - get_cost(p, s.x_proposal)
    ρden = -inner(p.M, s.x, s.η, s.gradient) - 0.5 * inner(p.M, s.x, s.η, s.Hη)
    ρnum = ρnum + ρ_reg
    ρden = ρden + ρ_reg
    ρ = (abs(ρnum / fx) < sqrt(eps(Float64))) ? 1 : ρnum / ρden # stability for small absolute relative model change

    model_decreased = ρden ≥ 0
    # Update the Hessian approximation
    update_hessian!(p.M, p.hessian!!, o.x, o.x_proposal, o.η)
    # Choose the new TR radius based on the model performance.
    # If the actual decrease is smaller than 1/4 of the predicted decrease,
    # then reduce the TR radius.
    if ρ < 0.1 || !model_decreased || isnan(ρ)
        s.trust_region_radius /= 4
    elseif ρ > 3 / 4 &&
        ((s.tcg_options.ηPη >= s.trust_region_radius^2) || (s.tcg_options.δHδ <= 0))
        s.trust_region_radius = min(2 * s.trust_region_radius, s.max_trust_region_radius)
    end
    # Choose to accept or reject the proposed step based on the model
    # performance. Note the strict inequality.
    if model_decreased &&
        (ρ > s.ρ_prime || (abs((ρnum) / (abs(fx) + 1)) < sqrt(eps(Float64)) && 0 < ρnum))
        copyto!(s.x, s.x_proposal)
        update_hessian_basis!(p.M, p.hessian!!, s.x)
    end
    return s
end
