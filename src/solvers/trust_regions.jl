
@doc raw"""
    TrustRegionsState <: AbstractHessianSolverState

describe the trust-regions solver, with


# Fields
where all but `p` are keyword arguments in the constructor

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

    TrustRegionsState(M,
        p=rand(M),
        X=zero_vector(M,p),
        sub_state=TruncatedConjugateGradientState(M, p, X),
)

construct a trust-regions Option with all other fields from above being
keyword arguments

# See also

[`trust_regions`](@ref)
"""
mutable struct TrustRegionsState{
    P,
    T,
    SC<:StoppingCriterion,
    RTR<:AbstractRetractionMethod,
    R<:Real,
    Proj,
    Op<:AbstractHessianSolverState,
} <: AbstractHessianSolverState
    p::P
    X::T
    stop::SC
    trust_region_radius::R
    max_trust_region_radius::R
    retraction_method::RTR
    randomize::Bool
    project!::Proj
    ρ_prime::R
    ρ_regularization::R
    sub_state::Op
    p_proposal::P
    f_proposal::R
    # Random
    Hgrad::T
    η::T
    Hη::T
    η_Cauchy::T
    Hη_Cauchy::T
    τ::R
    reduction_threshold::R
    augmentation_threshold::R
    function TrustRegionsState{P,T,SC,RTR,R,Proj,Op}(
        p::P,
        X::T,
        trust_region_radius::R,
        max_trust_region_radius::R,
        ρ_prime::R,
        ρ_regularization::R,
        randomize::Bool,
        stopping_citerion::SC,
        retraction_method::RTR,
        reduction_threshold::R,
        augmentation_threshold::R,
        sub_state::Op,
        project!::Proj=copyto!,
    ) where {
        P,
        T,
        SC<:StoppingCriterion,
        RTR<:AbstractRetractionMethod,
        R<:Real,
        Proj,
        Op<:AbstractHessianSolverState,
    }
        trs = new{P,T,SC,RTR,R,Proj,Op}()
        trs.p = p
        trs.X = X
        trs.stop = stopping_citerion
        trs.retraction_method = retraction_method
        trs.trust_region_radius = trust_region_radius
        trs.max_trust_region_radius = max_trust_region_radius::R
        trs.ρ_prime = ρ_prime
        trs.ρ_regularization = ρ_regularization
        trs.randomize = randomize
        trs.sub_state = sub_state
        trs.reduction_threshold = reduction_threshold
        trs.augmentation_threshold = augmentation_threshold
        trs.project! = project!
        return trs
    end
end
function TrustRegionsState(
    M::TM,
    p::P=rand(M),
    X::T=zero_vector(M, p),
    sub_state::Op=TruncatedConjugateGradientState(M, p, X);
    ρ_prime::R=0.1,
    ρ_regularization::R=1000.0,
    randomize::Bool=false,
    stopping_criterion::SC=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
    max_trust_region_radius::R=sqrt(manifold_dimension(M)),
    trust_region_radius::R=max_trust_region_radius / 8,
    retraction_method::RTR=default_retraction_method(M, typeof(p)),
    reduction_threshold::R=0.1,
    augmentation_threshold::R=0.75,
    project!::Proj=copyto!,
) where {
    TM<:AbstractManifold,
    P,
    T,
    R<:Real,
    SC<:StoppingCriterion,
    RTR<:AbstractRetractionMethod,
    Proj,
    Op<:AbstractHessianSolverState,
}
    return TrustRegionsState{P,T,SC,RTR,R,Proj,Op}(
        p,
        X,
        trust_region_radius,
        max_trust_region_radius,
        ρ_prime,
        ρ_regularization,
        randomize,
        stopping_criterion,
        retraction_method,
        reduction_threshold,
        augmentation_threshold,
        sub_state,
        project!,
    )
end
get_iterate(trs::TrustRegionsState) = trs.p
function set_iterate!(trs::TrustRegionsState, M, p)
    copyto!(M, trs.p, p)
    return trs
end

function show(io::IO, trs::TrustRegionsState)
    i = get_count(trs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(trs.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Trust Region Method
    $Iter
    ## Parameters
    * augmentation threshold: $(trs.augmentation_threshold)
    * randomize: $(trs.randomize)
    * reduction threshold: $(trs.reduction_threshold)
    * retraction method: $(trs.retraction_method)
    * ρ‘: $(trs.ρ_prime)
    * ρ_regularization: $(trs.ρ_regularization)
    * trust region radius: $(trs.trust_region_radius) (max: $(trs.max_trust_region_radius))

    ## Stopping Criterion
    $(status_summary(trs.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
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
* `retraction` – (`default_retraction_method(M, typeof(p))`) approximation of the exponential map
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
* `θ` – (`1.0`) 1+θ is the superlinear convergence target rate of the tCG-method
    [`truncated_conjugate_gradient_descent`](@ref), which computes an
    approximate solution for the trust-region subproblem. The tCG-method aborts
    if the residual is less than or equal to the initial residual to the power of 1+θ.
* `κ` – (`0.1`) the linear convergence target rate of the tCG-method
    [`truncated_conjugate_gradient_descent`](@ref), which computes an
    approximate solution for the trust-region subproblem. The method aborts if the
    residual is less than or equal to κ times the initial residual.
* `reduction_threshold` – (`0.1`) Trust-region reduction threshold: if ρ (the performance ratio for
    the iterate) is less than this bound, the trust-region radius and thus the trust-regions
    decreases.
* `augmentation_threshold` – (`0.75`) Trust-region augmentation threshold: if ρ (the performance ratio for
    the iterate) is greater than this and further conditions apply, the trust-region radius and thus the trust-regions increases.

# Output

the obtained (approximate) minimizer ``p^*``, see [`get_solver_return`](@ref) for details

# see also
[`truncated_conjugate_gradient_descent`](@ref)
"""
trust_regions(M::AbstractManifold, args...; kwargs...)
function trust_regions(
    M::AbstractManifold, f, grad_f, Hess_f::TH; kwargs...
) where {TH<:Function}
    return trust_regions(M, f, grad_f, Hess_f, rand(M); kwargs...)
end
function trust_regions(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f::TH,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    preconditioner=if evaluation isa InplaceEvaluation
        (M, Y, p, X) -> (Y .= X)
    else
        (M, p, X) -> X
    end,
    kwargs...,
) where {TH<:Function}
    mho = ManifoldHessianObjective(f, grad_f, Hess_f, preconditioner; evaluation=evaluation)
    return trust_regions(M, mho, p; evaluation=evaluation, kwargs...)
end
function trust_regions(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f::TH, #we first fill a default below bwfore dispatching on p::Number
    p::Number;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    preconditioner=(M, p, X) -> X,
    kwargs...,
) where {TH<:Function}
    q = [p]
    f_(M, p) = f(M, p[])
    Hess_f_ = Hess_f
    # For now we can not update the gradient within the ApproxHessian so the filled default
    # Hessian fails here
    if evaluation isa AllocatingEvaluation
        grad_f_ = (M, p) -> [grad_f(M, p[])]
        Hess_f_ = (M, p, X) -> [Hess_f(M, p[], X[])]
        precon_ = (M, p, X) -> [preconditioner(M, p[], X[])]
    else
        grad_f_ = (M, X, p) -> (X .= [grad_f(M, p[])])
        Hess_f_ = (M, Y, p, X) -> (Y .= [Hess_f(M, p[], X[])])
        precon_ = (M, Y, p, X) -> (Y .= [preconditioner(M, p[], X[])])
    end
    rs = trust_regions(
        M, f_, grad_f_, Hess_f_, q; preconditioner=precon_, evaluation=evaluation, kwargs...
    )
    return (typeof(q) == typeof(rs)) ? rs[] : rs
end
function trust_regions(M::AbstractManifold, f, grad_f; kwargs...)
    return trust_regions(M, f, grad_f, rand(M); kwargs...)
end
function trust_regions(
    M::AbstractManifold,
    f::TF,
    grad_f::TdF,
    p;
    evaluation=AllocatingEvaluation(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
) where {TF,TdF}
    hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation=evaluation, retraction_method=retraction_method
    )
    return trust_regions(
        M,
        f,
        grad_f,
        hess_f,
        p;
        evaluation=evaluation,
        retraction_method=retraction_method,
        kwargs...,
    )
end
function trust_regions(
    M::AbstractManifold, mho::ManifoldHessianObjective, p=rand(M); kwargs...
)
    q = copy(M, p)
    return trust_regions!(M, mho, q; kwargs...)
end
# If the Hessian go autofilled already _and_ we have a p that is a number
@doc raw"""
    trust_regions!(M, f, grad_f, Hess_f, p; kwargs...)
    trust_regions!(M, f, grad_f, p; kwargs...)

evaluate the Riemannian trust-regions solver for optimization on manifolds in place of `p`.

# Input
* `M` – a manifold ``\mathcal M``
* `f` – a cost function ``F: \mathcal M → ℝ`` to minimize
* `grad_f`- the gradient ``\operatorname{grad}F: \mathcal M → T \mathcal M`` of ``F``
* `Hess_f` – (optional) the hessian ``H( \mathcal M, x, ξ)`` of ``F``
* `x` – an initial value ``x  ∈  \mathcal M``

For the case that no hessian is provided, the Hessian is computed using finite difference, see
[`ApproxHessianFiniteDifference`](@ref).

for more details and all options, see [`trust_regions`](@ref)
"""
trust_regions!(M::AbstractManifold, args...; kwargs...)
function trust_regions!(
    M::AbstractManifold,
    f,
    grad_f,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    kwargs...,
)
    hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation=evaluation, retraction_method=retraction_method
    )
    return trust_regions!(
        M,
        f,
        grad_f,
        hess_f,
        p;
        evaluation=evaluation,
        retraction_method=retraction_method,
        kwargs...,
    )
end
function trust_regions!(
    M::AbstractManifold,
    f,
    grad_f,
    Hess_f::TH,
    p;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    preconditioner=if evaluation isa InplaceEvaluation
        (M, Y, p, X) -> (Y .= X)
    else
        (M, p, X) -> X
    end,
    kwargs...,
) where {TH<:Function}
    mho = ManifoldHessianObjective(f, grad_f, Hess_f, preconditioner; evaluation=evaluation)
    return trust_regions!(M, mho, p; evaluation=evaluation, kwargs...)
end
function trust_regions!(
    M::AbstractManifold,
    mho::ManifoldHessianObjective,
    p;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
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
    reduction_threshold::Float64=0.1,
    augmentation_threshold::Float64=0.75,
    sub_state::AbstractHessianSolverState=TruncatedConjugateGradientState(
        M,
        p,
        zero_vector(M, p);
        θ=θ,
        κ=κ,
        trust_region_radius,
        randomize=randomize,
        (project!)=project!,
    ),
    kwargs..., #collect rest
) where {Proj}
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
    dmho = decorate_objective!(M, mho; kwargs...)
    mp = DefaultManoptProblem(M, dmho)
    trs = TrustRegionsState(
        M,
        p,
        get_gradient(mp, p),
        sub_state;
        trust_region_radius=trust_region_radius,
        max_trust_region_radius=max_trust_region_radius,
        ρ_prime=ρ_prime,
        ρ_regularization=ρ_regularization,
        randomize=randomize,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        reduction_threshold=reduction_threshold,
        augmentation_threshold=augmentation_threshold,
        (project!)=project!,
    )
    dtrs = decorate_state!(trs; kwargs...)
    solve!(mp, dtrs)
    return get_solver_return(get_objective(mp), dtrs)
end
function initialize_solver!(mp::AbstractManoptProblem, trs::TrustRegionsState)
    M = get_manifold(mp)
    get_gradient!(mp, trs.X, trs.p)
    trs.η = zero_vector(M, trs.p)
    trs.Hη = zero_vector(M, trs.p)
    trs.p_proposal = deepcopy(trs.p)
    trs.f_proposal = zero(trs.trust_region_radius)

    trs.η_Cauchy = zero_vector(M, trs.p)
    trs.Hη_Cauchy = zero_vector(M, trs.p)
    trs.τ = zero(trs.trust_region_radius)
    trs.Hgrad = zero_vector(M, trs.p)
    return trs
end

function step_solver!(mp::AbstractManoptProblem, trs::TrustRegionsState, i)
    M = get_manifold(mp)
    mho = get_objective(mp)
    # Determine eta0
    if trs.randomize
        # Random vector in T_x M (this has to be very small)
        trs.η = 10.0^(-6) * rand(M; vector_at=trs.p)
        while norm(M, trs.p, trs.η) > trs.trust_region_radius
            # inside trust-region
            trs.η *= sqrt(sqrt(eps(Float64)))
        end
    else
        zero_vector!(M, trs.η, trs.p)
    end
    # Solve TR subproblem - update options
    trs.sub_state.p = trs.p
    trs.sub_state.η = trs.η
    trs.sub_state.trust_region_radius = trs.trust_region_radius
    solve!(mp, trs.sub_state)
    #
    trs.η = trs.sub_state.η
    trs.Hη = trs.sub_state.Hη

    # Initialize the cost function F und the gradient of the cost function
    # gradF at the point x
    trs.X = trs.sub_state.X
    fx = get_cost(mp, trs.p)
    # If using randomized approach, compare result with the Cauchy point.
    if trs.randomize
        norm_grad = norm(M, trs.p, trs.X)
        # Check the curvature,
        get_hessian!(mp, trs.Hgrad, trs.p, trs.X)
        trs.τ = real(inner(M, trs.p, trs.X, trs.Hgrad))
        trs.τ = if (trs.τ <= 0)
            one(trs.τ)
        else
            min(norm_grad^3 / (trs.trust_region_radius * trs.τ), 1)
        end
        # compare to Cauchy point and store best
        model_value =
            fx +
            real(inner(M, trs.p, trs.X, trs.η)) +
            0.5 * real(inner(M, trs.p, trs.Hη, trs.η))
        modle_value_Cauchy = fx
        -trs.τ * trs.trust_region_radius * norm_grad
        +0.5 * trs.τ^2 * trs.trust_region_radius^2 / (norm_grad^2) *
        real(inner(M, trs.p, trs.Hgrad, trs.X))
        if modle_value_Cauchy < model_value
            copyto!(M, trs.η, (-trs.τ * trs.trust_region_radius / norm_grad) * trs.X)
            copyto!(M, trs.Hη, (-trs.τ * trs.trust_region_radius / norm_grad) * trs.Hgrad)
        end
    end
    # Compute the tentative next iterate (the proposal)
    retract!(M, trs.p_proposal, trs.p, trs.η, trs.retraction_method)
    # Check the performance of the quadratic model against the actual cost.
    ρ_reg = max(1, abs(fx)) * eps(Float64) * trs.ρ_regularization
    ρnum = fx - get_cost(mp, trs.p_proposal)
    ρden = -real(inner(M, trs.p, trs.η, trs.X)) - 0.5 * real(inner(M, trs.p, trs.η, trs.Hη))
    ρnum = ρnum + ρ_reg
    ρden = ρden + ρ_reg
    ρ = (abs(ρnum / fx) < sqrt(eps(Float64))) ? 1 : ρnum / ρden # stability for small absolute relative model change

    model_decreased = ρden ≥ 0
    # Update the Hessian approximation
    update_hessian!(M, mho.hessian!!, trs.p, trs.p_proposal, trs.η)
    # Choose the new TR radius based on the model performance.
    # If the actual decrease is smaller than reduction_threshold of the predicted decrease,
    # then reduce the TR radius.
    if ρ < trs.reduction_threshold || !model_decreased || isnan(ρ)
        trs.trust_region_radius /= 4
    elseif ρ > trs.augmentation_threshold / 4 &&
        ((trs.sub_state.ηPη >= trs.trust_region_radius^2) || (trs.sub_state.δHδ <= 0))
        trs.trust_region_radius = min(
            2 * trs.trust_region_radius, trs.max_trust_region_radius
        )
    end
    # Choose to accept or reject the proposed step based on the model
    # performance. Note the strict inequality.
    if model_decreased &&
        (ρ > trs.ρ_prime || (abs((ρnum) / (abs(fx) + 1)) < sqrt(eps(Float64)) && 0 < ρnum))
        copyto!(trs.p, trs.p_proposal)
        update_hessian_basis!(M, mho.hessian!!, trs.p)
    end
    return trs
end
