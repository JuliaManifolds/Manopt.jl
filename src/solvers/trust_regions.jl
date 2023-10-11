
@doc raw"""
    TrustRegionsState <: AbstractHessianSolverState

Store the state of the trust-regions solver.

# Fields

All the following fields (besides `p`) can be set by specifying them as keywords.

* `acceptance_rate`         – (`0.1`) a lower bound of the performance ratio for the iterate that
    decides if the iteration will be accepted or not.
* `max_trust_region_radius` – (`sqrt(manifold_dimension(M))`) the maximum trust-region radius
* `p`                       – (`rand(M)` if a manifold is provided) the current iterate
* `project!`                – (`copyto!`) specify a projection operation for tangent vectors
    for numerical stability. A function `(M, Y, p, X) -> ...` working in place of `Y`.
    per default, no projection is perfomed, set it to `project!` to activate projection.
* `stop`                    – ([`StopAfterIteration`](@ref)`(1000) | `[`StopWhenGradientNormLess`](@ref)`(1e-6)`)
* `randomize`               – (`false`) indicates if the trust-region solve is to be initiated with a
    random tangent vector. If set to true, no preconditioner will be
    used. This option is set to true in some scenarios to escape saddle
    points, but is otherwise seldom activated.
* `ρ_regularization`        – (`10000.0`) regularize the model fitness ``ρ`` to avoid division by zero
* `sub_state`               – ([`TruncatedConjugateGradientState`](@ref)`(M, p, X)`)
* `σ`                       – (`0.0` or `1e-6` depending on `randomize`) Gaussian standard deviation when creating the random initial tangent vector
* `trust_region_radius`     – (`max_trust_region_radius / 8`) the (initial) trust-region radius
* `X`                       – (`zero_vector(M,p)`) the current gradient `grad_f(p)`
  Use this default to specify the type of tangent vector to allocate also for the internal (tangent vector) fields.

# Internal Fields

* `HX`, `HY`, `HZ`          – interims storage (to avoid allocation) of ``\operatorname{Hess} f(p)[\cdot]` of `X`, `Y`, `Z`
* `Y`                       – the solution (tangent vector) of the subsolver
* `Z`                       – the Cauchy point (only used if random is activated)


# Constructor

    TrustRegionsState(M, p=rand(M))

construct a trust-regions Option with all fields from above with a default in brackets being
keyword arguments.

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
    acceptance_rate::R
    ρ_regularization::R
    sub_state::Op
    p_proposal::P
    f_proposal::R
    # Only required for Random mode Random
    HX::T
    Y::T
    HY::T
    Z::T
    HZ::T
    τ::R
    σ::R
    reduction_threshold::R
    reduction_factor::R
    augmentation_threshold::R
    augmentation_factor::R
    function TrustRegionsState{P,T,SC,RTR,R,Proj,Op}(
        p::P,
        X::T,
        trust_region_radius::R,
        max_trust_region_radius::R,
        acceptance_rate::R,
        ρ_regularization::R,
        randomize::Bool,
        stopping_citerion::SC,
        retraction_method::RTR,
        reduction_threshold::R,
        augmentation_threshold::R,
        sub_state::Op,
        project!::Proj=copyto!,
        reduction_factor=0.25,
        augmentation_factor=2.0,
        σ::R=random ? 1e-6 : 0.0,
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
        trs.acceptance_rate = acceptance_rate
        trs.ρ_regularization = ρ_regularization
        trs.randomize = randomize
        trs.sub_state = sub_state
        trs.reduction_threshold = reduction_threshold
        trs.reduction_factor = reduction_factor
        trs.augmentation_threshold = augmentation_threshold
        trs.augmentation_factor = augmentation_factor
        trs.project! = project!
        trs.σ = σ
        return trs
    end
end
function TrustRegionsState(
    M::TM,
    p::P=rand(M);
    X::T=zero_vector(M, p),
    sub_state::Op=TruncatedConjugateGradientState(M, p, X),
    ρ_prime::R=0.1, #deprecated, remove on next breaking change
    acceptance_rate=ρ_prime,
    ρ_regularization::R=1000.0,
    randomize::Bool=false,
    stopping_criterion::SC=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
    max_trust_region_radius::R=sqrt(manifold_dimension(M)),
    trust_region_radius::R=max_trust_region_radius / 8,
    retraction_method::RTR=default_retraction_method(M, typeof(p)),
    reduction_threshold::R=0.1,
    reduction_factor=0.25,
    augmentation_threshold::R=0.75,
    augmentation_factor=2.0,
    project!::Proj=copyto!,
    σ=randomize ? 1e-4 : 0.0,
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
        acceptance_rate,
        ρ_regularization,
        randomize,
        stopping_criterion,
        retraction_method,
        reduction_threshold,
        augmentation_threshold,
        sub_state,
        project!,
        reduction_factor,
        augmentation_factor,
        σ,
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
    * acceptance_rate (ρ'):   $(trs.acceptance_rate)
    * augmentation threshold: $(trs.augmentation_threshold) (factor: $(trs.augmentation_factor))
    * randomize:              $(trs.randomize)
    * reduction threshold:    $(trs.reduction_threshold) (factor: $(trs.reduction_factor))
    * retraction method:      $(trs.retraction_method)
    * ρ_regularization:       $(trs.ρ_regularization)
    * trust region radius:    $(trs.trust_region_radius) (max: $(trs.max_trust_region_radius))

    ## Stopping Criterion
    $(status_summary(trs.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

@doc raw"""
    TrustRegionTangentSpaceModelObjective{TH<:Union{Function,Nothing},O<:AbstractManifoldHessianObjective,T} <: AbstractManifoldSubObjective{O}

A trust region model of the form

```math
    \operatorname{arg\,min}_{X \in T_p\mathcal M} c + ⟨G, X⟩_p + \frac{1}(2} ⟨B(X), X⟩_p
    ,\qquad
    \lVert X \rVert ≤ Δ
```

where

* ``G`` is (a tangent vector that is an approximation of) the gradient ``\operatorname{grad} f(p)``
* ``B`` is (a bilinear form that is an approximantion of) the Hessian ``\operatorname{Hess} f(p)``
* ``c`` is the current cost ``f(p)``, but might be set to zero for simplicity, since we are only interested in the minimizer
* ``Δ`` is the current trust region radius

# Fields

* `objective` – an [`AbstractManifoldHessianObjective`](@ref) proving ``f``, its gradient and Hessian
* `c` the current cost
* `G` the current Gradient
* `H` the current bilinear form (Approximation of the Hessian)
* `Δ` the current trust region radius

If `H` is set to nothing, the hessian from the `objective` is used.
"""
struct TrustRegionTangentSpaceModelObjective{
    TH<:Union{Function,Nothing},O<:AbstractManifoldHessianObjective,T,R
} <: AbstractManifoldSubObjective{O}
    objective::O
    c::R
    G::T
    H::TH
    Δ::R
end
function TrustRegionTangentSpaceModelObjective(TpM::TangentSpace, mho, p=rand(M); kwargs...)
    return TrustRegionTangentSpaceModelObjective(base_manifold(TpM), p; kwargs...)
end
function TrustRegionTangentSpaceModelObjective(
    M::AbstractManifold,
    mho::O,
    p=rand(M);
    c::R=get_cost(M, mho, p),
    Δ::R=injectivity_radius(M) / 4,
    G::T=get_gradient(M, mho, p),
    H::TH=nothing,
) where {TH<:Union{Function,Nothing},O<:AbstractManifoldHessianObjective,T,R}
    return TrustRegionTangentSpaceModelObjective{TH,O,T,R}(mho, c, G, H, Δ)
end

get_objective(trm::TrustRegionTangentSpaceModelObjective) = trom.objective

@doc raw"""
    trust_regions(M, f, grad_f, hess_f, p=rand(M))
    trust_regions(M, f, grad_f, p=rand(M))

run the Riemannian trust-regions solver for optimization on manifolds to minimize `f`
cf. [[Absil, Baker, Gallivan, FoCM, 2006](@cite AbsilBakerGallivan:2006); [Conn, Gould, Toint, SIAM, 2000](@cite ConnGouldToint:2000)].

For the case that no hessian is provided, the Hessian is computed using finite differences,
see [`ApproxHessianFiniteDifference`](@ref).
For solving the the inner trust-region subproblem of finding an update-vector,
by default the [`truncated_conjugate_gradient_descent`](@ref) is used.

# Input
* `M`      – a manifold ``\mathcal M``
* `f`      – a cost function ``f : \mathcal M → ℝ`` to minimize
* `grad_f` – the gradient ``\operatorname{grad}F : \mathcal M → T \mathcal M`` of ``F``
* `Hess_f` – (optional), the hessian ``\operatorname{Hess}F(x): T_x\mathcal M → T_x\mathcal M``, ``X ↦ \operatorname{Hess}F(x)[X] = ∇_ξ\operatorname{grad}f(x)``
* `p`      – (`rand(M)`) an initial value ``x  ∈  \mathcal M``

# Optional
* `evaluation`              – ([`AllocatingEvaluation`](@ref)) specify whether the gradient and hessian work by
   allocation (default) or [`InplaceEvaluation`](@ref) in place
* `max_trust_region_radius` – the maximum trust-region radius
* `preconditioner`          – a preconditioner (a symmetric, positive definite operator
  that should approximate the inverse of the Hessian)
* `randomize`               – set to true if the trust-region solve is to be initiated with a
  random tangent vector and no preconditioner will be used.
* `project!`                – (`copyto!`) specify a projection operation for tangent vectors
  within the subsolver for numerical stability.
  this means we require a function `(M, Y, p, X) -> ...` working in place of `Y`.
* `retraction` – (`default_retraction_method(M, typeof(p))`) a retraction to use
* `stopping_criterion`      – ([`StopWhenAny`](@ref)([`StopAfterIteration`](@ref)`(1000)`,
  [`StopWhenGradientNormLess`](@ref)`(1e-6)`) a functor inheriting
  from [`StoppingCriterion`](@ref) indicating when to stop.
* `trust_region_radius`     – the initial trust-region radius
* `acceptance_rate`         – Accept/reject threshold: if ρ (the performance ratio for the
  iterate) is at least the acceptance rate ρ', the candidate is accepted.
  This value should  be between ``0`` and ``\frac{1}{4}``
  (formerly this was called `ρ_prime, which will be removed on the next breaking change)
* `ρ_regularization`        – (`1e3`) regularize the performance evaluation ``ρ``
  to avoid numerical inaccuracies.
* `θ`                       – (`1.0`) 1+θ is the superlinear convergence target rate of the tCG-method
    [`truncated_conjugate_gradient_descent`](@ref), and is used in a stopping crierion therein
* `κ`                       – (`0.1`) the linear convergence target rate of the tCG method
    [`truncated_conjugate_gradient_descent`](@ref), and is used in a stopping crierion therein
* `reduction_threshold`     – (`0.1`) trust-region reduction threshold: if ρ is below this threshold,
  the trust region radius is reduced by `reduction_factor`.
* `reduction_factor`        – (`0.25`) trust-region reduction factor
* `augmentation_threshold`  – (`0.75`) trust-region augmentation threshold: if ρ is above this threshold,
  we have a solution on the trust region boundary and negative curvature, we extend (augment) the radius
* `augmentation_factor`     – (`2.0`) trust-region augmentation factor

For the case that no hessian is provided, the Hessian is computed using finite difference, see
[`ApproxHessianFiniteDifference`](@ref).

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
    Hess_f::TH, #we first fill a default below before dispatching on p::Number
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
    M::AbstractManifold, mho::O, p=rand(M); kwargs...
) where {O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return trust_regions!(M, mho, q; kwargs...)
end
# If the Hessian go autofilled already _and_ we have a p that is a number
@doc raw"""
    trust_regions!(M, f, grad_f, Hess_f, p; kwargs...)
    trust_regions!(M, f, grad_f, p; kwargs...)

evaluate the Riemannian trust-regions solver in place of `p`.

# Input
* `M`      – a manifold ``\mathcal M``
* `f`      – a cost function ``f: \mathcal M → ℝ`` to minimize
* `grad_f` – the gradient ``\operatorname{grad}f: \mathcal M → T \mathcal M`` of ``F``
* `Hess_f` – (optional) the hessian ``\operatorname{Hess} f``
* `p`      – an initial value ``p  ∈  \mathcal M``

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
    mho::O,
    p;
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, typeof(p)),
    stopping_criterion::StoppingCriterion=StopAfterIteration(1000) |
                                          StopWhenGradientNormLess(1e-6),
    max_trust_region_radius::R=sqrt(manifold_dimension(M)),
    trust_region_radius::R=max_trust_region_radius / 8,
    randomize::Bool=false, # Deprecated, remove on next release (use just σ)
    project!::Proj=copyto!,
    ρ_prime::R=0.1, # Deprecated, remove on next breaking change (use acceptance_rate)
    acceptance_rate::R=ρ_prime,
    ρ_regularization=1e3,
    θ::R=1.0,
    κ::R=0.1,
    σ=randomize ? 1e-3 : 0.0,
    reduction_threshold::R=0.1,
    reduction_factor::R=0.25,
    augmentation_threshold::R=0.75,
    augmentation_factor::R=2.0,
    # ToDo – Tangent Space in Base? Implement TR Model, otherwise like below
    # sub_problem=TangentSpaceModelProblem(M, p, TrustRegionModel(mho)),
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
) where {Proj,O<:Union{ManifoldHessianObjective,AbstractDecoratedManifoldObjective},R}
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
    dmp = DefaultManoptProblem(M, dmho)
    trs = TrustRegionsState(
        M,
        p;
        X=get_gradient(dmp, p),
        sub_state=sub_state,
        trust_region_radius=trust_region_radius,
        max_trust_region_radius=max_trust_region_radius,
        acceptance_rate=acceptance_rate,
        ρ_regularization=ρ_regularization,
        randomize=randomize,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        reduction_threshold=reduction_threshold,
        reduction_factor=reduction_factor,
        augmentation_threshold=augmentation_threshold,
        augmentation_factor=augmentation_factor,
        (project!)=project!,
        σ=σ,
    )
    dtrs = decorate_state!(trs; kwargs...)
    solve!(dmp, dtrs)
    return get_solver_return(get_objective(dmp), dtrs)
end

function initialize_solver!(mp::AbstractManoptProblem, trs::TrustRegionsState)
    M = get_manifold(mp)
    get_gradient!(mp, trs.X, trs.p)
    trs.Y = zero_vector(M, trs.p)
    trs.HY = zero_vector(M, trs.p)
    trs.p_proposal = deepcopy(trs.p)
    trs.f_proposal = zero(trs.trust_region_radius)
    if trs.randomize #only init if necessary
        trs.Z = zero_vector(M, trs.p)
        trs.HZ = zero_vector(M, trs.p)
        trs.τ = zero(trs.trust_region_radius)
        trs.HX = zero_vector(M, trs.p)
    end
    return trs
end

function step_solver!(mp::AbstractManoptProblem, trs::TrustRegionsState, i)
    M = get_manifold(mp)
    mho = get_objective(mp)
    # Determine the initial tangent vector used as start point for the subsolvereta0
    if trs.σ > 0
        rand!(M, trs.Y; vector_at=trs.p, σ=trs.σ)
        nY = norm(M, trs.p, trs.Y)
        if nY > trs.trust_region_radius # move inside if outside
            trs.Y *= trs.trust_region_radius / (2 * nY)
        end
    else
        zero_vector!(M, trs.Y, trs.p)
    end
    # Update the current gradient
    get_gradient!(M, trs.X, mho, trs.p)
    # Solve TR subproblem – update options
    # TODO provide these setters for the sub problem / sub state
    # set_paramater!(trs.sub_problem, :Basepoint, trs.p)
    set_manopt_parameter!(trs.sub_state, :Basepoint, trs.p)
    set_manopt_parameter!(trs.sub_state, :Iterate, trs.Y)
    set_manopt_parameter!(trs.sub_state, :TrustRegionRadius, trs.trust_region_radius)
    solve!(mp, trs.sub_state)
    #
    copyto!(M, trs.Y, trs.p, get_solver_result(trs.sub_state))
    f = get_cost(mp, trs.p)
    if trs.σ > 0 # randomized approach: compare result with the Cauchy point.
        nX = norm(M, trs.p, trs.X)
        get_hessian!(M, trs.HY, mho, trs.p, trs.Y)
        # Check the curvature,
        get_hessian!(mp, trs.HX, trs.p, trs.X)
        trs.τ = real(inner(M, trs.p, trs.X, trs.HX))
        trs.τ = if (trs.τ <= 0)
            one(trs.τ)
        else
            min(nX^3 / (trs.trust_region_radius * trs.τ), 1)
        end
        # compare to Cauchy point and store best
        model_value =
            f +
            real(inner(M, trs.p, trs.X, trs.Y)) +
            0.5 * real(inner(M, trs.p, trs.HY, trs.Y))
        model_value_Cauchy =
            f - trs.τ * trs.trust_region_radius * nX +
            0.5 * trs.τ^2 * trs.trust_region_radius^2 / (nX^2) *
            real(inner(M, trs.p, trs.HX, trs.X))
        if model_value_Cauchy < model_value
            copyto!(M, trs.Y, (-trs.τ * trs.trust_region_radius / nX) * trs.X)
            copyto!(M, trs.HY, (-trs.τ * trs.trust_region_radius / nX) * trs.HX)
        end
    end
    # Compute the tentative next iterate (the proposal)
    retract!(M, trs.p_proposal, trs.p, trs.Y, trs.retraction_method)
    # Compute ρ_k as in (8) of ABG2007
    ρ_reg = max(1, abs(f)) * eps(Float64) * trs.ρ_regularization
    ρnum = f - get_cost(mp, trs.p_proposal)
    ρden = -real(inner(M, trs.p, trs.Y, trs.X)) - 0.5 * real(inner(M, trs.p, trs.Y, trs.HY))
    ρnum = ρnum + ρ_reg
    ρden = ρden + ρ_reg
    ρ = ρnum / ρden
    model_decreased = ρden ≥ 0
    # Update the Hessian approximation - i.e. really unwrap the original Hessian function
    # and update it if it is an approximate Hessian.
    update_hessian!(M, get_hessian_function(mho, true), trs.p, trs.p_proposal, trs.Y)
    # Choose the new TR radius based on the model performance.
    # Case (a) we performe poorly -> decrease radius
    if ρ < trs.reduction_threshold || !model_decreased || isnan(ρ)
        trs.trust_region_radius *= trs.reduction_factor
    elseif ρ > trs.augmentation_threshold &&
        get_manopt_parameter(trs.sub_state, :TrustRegionExceeded)
        # (b) We perform great and exceed/reach the trust region boundary -> increase radius
        trs.trust_region_radius = min(
            trs.augmentation_factor * trs.trust_region_radius, trs.max_trust_region_radius
        )
    end
    # (c) if we decreased and perform well enough -> accept step
    if model_decreased && (ρ > trs.acceptance_rate)
        copyto!(trs.p, trs.p_proposal)
        # If we work with approximate hessians -> update base point there
        update_hessian_basis!(M, get_hessian_function(mho, true), trs.p)
    end
    return trs
end
