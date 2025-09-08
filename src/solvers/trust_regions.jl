@doc """
    TrustRegionsState <: AbstractHessianSolverState

Store the state of the trust-regions solver.

# Fields

* `acceptance_rate`:         a lower bound of the performance ratio for the iterate
  that decides if the iteration is accepted or not.
* `HX`, `HY`, `HZ`:          interim storage (to avoid allocation) of ``$(_tex(:Hess)) f(p)[⋅]` of `X`, `Y`, `Z`
* `max_trust_region_radius`: the maximum trust-region radius
$(_var(:Field, :p; add = [:as_Iterate]))
* `project!`:                for numerical stability it is possible to project onto the tangent space after every iteration.
  the function has to work inplace of `Y`, that is `(M, Y, p, X) -> Y`, where `X` and `Y` can be the same memory.
$(_var(:Field, :stopping_criterion, "stop"))
* `randomize`:               indicate whether `X` is initialised to a random vector or not
* `ρ_regularization`:        regularize the model fitness ``ρ`` to avoid division by zero
$(_var(:Field, :sub_problem))
$(_var(:Field, :sub_state))
* `σ`:                       Gaussian standard deviation when creating the random initial tangent vector
  This field has no effect, when `randomize` is false.
* `trust_region_radius`: the trust-region radius
$(_var(:Field, :X))
* `Y`:                       the solution (tangent vector) of the subsolver
* `Z`:                       the Cauchy point (only used if random is activated)


# Constructors

    TrustRegionsState(M, mho::AbstractManifoldHessianObjective; kwargs...)
    TrustRegionsState(M, sub_problem, sub_state; kwargs...)
    TrustRegionsState(M, sub_problem; evaluation=AllocatingEvaluation(), kwargs...)

create a trust region state.
* given a [`AbstractManifoldHessianObjective`](@ref) `mho`, the default sub solver,
  a [`TruncatedConjugateGradientState`](@ref) with `mho` used to define the problem on a tangent space is created
* given a `sub_problem` and an `evaluation=` keyword, the sub problem solver is assumed to be the closed form solution,
  where `evaluation` determines how to call the sub function.

# Input

$(_var(:Argument, :M; type = true))
$(_var(:Argument, :sub_problem))
$(_var(:Argument, :sub_state))

## Keyword arguments

* `acceptance_rate=0.1`
* `max_trust_region_radius=sqrt(manifold_dimension(M))`
$(_var(:Keyword, :p; add = :as_Initial))
* `project!=copyto!`
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(1000)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-6)`"))
* `randomize=false`
* `ρ_regularization=10000.0`
* `θ=1.0`
* `trust_region_radius=max_trust_region_radius / 8`
$(_var(:Keyword, :X; add = :as_Memory))

# See also

[`trust_regions`](@ref)
"""
mutable struct TrustRegionsState{
        P,
        T,
        Pr,
        St <: AbstractManoptSolverState,
        SC <: StoppingCriterion,
        RTR <: AbstractRetractionMethod,
        R <: Real,
        Proj,
    } <: AbstractSubProblemSolverState
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
    sub_problem::Pr
    sub_state::St
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
    function TrustRegionsState{P, T, Pr, St, SC, RTR, R, Proj}(
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
            sub_problem::Pr,
            sub_state::St,
            project!::Proj = (copyto!),
            reduction_factor = 0.25,
            augmentation_factor = 2.0,
            σ::R = random ? 1.0e-6 : 0.0,
        ) where {
            P,
            T,
            Pr,
            St <: AbstractManoptSolverState,
            SC <: StoppingCriterion,
            RTR <: AbstractRetractionMethod,
            R <: Real,
            Proj,
        }
        trs = new{P, T, Pr, St, SC, RTR, R, Proj}()
        trs.p = p
        trs.X = X
        trs.stop = stopping_citerion
        trs.retraction_method = retraction_method
        trs.trust_region_radius = trust_region_radius
        trs.max_trust_region_radius = max_trust_region_radius::R
        trs.acceptance_rate = acceptance_rate
        trs.ρ_regularization = ρ_regularization
        trs.randomize = randomize
        trs.sub_problem = sub_problem
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
        M::AbstractManifold,
        sub_problem::Pr,
        sub_state::St;
        p::P = rand(M),
        X::T = zero_vector(M, p),
        acceptance_rate = 0.1,
        ρ_regularization::R = 1000.0,
        randomize::Bool = false,
        stopping_criterion::SC = StopAfterIteration(1000) | StopWhenGradientNormLess(1.0e-6),
        max_trust_region_radius::R = sqrt(manifold_dimension(M)),
        trust_region_radius::R = max_trust_region_radius / 8,
        retraction_method::RTR = default_retraction_method(M, typeof(p)),
        reduction_threshold::R = 0.1,
        reduction_factor = 0.25,
        augmentation_threshold::R = 0.75,
        augmentation_factor = 2.0,
        project!::Proj = (copyto!),
        σ = randomize ? 1.0e-4 : 0.0,
    ) where {
        P,
        T,
        Pr <: Union{AbstractManoptProblem, F} where {F},
        St <: AbstractManoptSolverState,
        R <: Real,
        SC <: StoppingCriterion,
        RTR <: AbstractRetractionMethod,
        Proj,
    }
    return TrustRegionsState{P, T, Pr, St, SC, RTR, R, Proj}(
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
        sub_problem,
        sub_state,
        project!,
        reduction_factor,
        augmentation_factor,
        σ,
    )
end
function TrustRegionsState(
        M::AbstractManifold, sub_problem; evaluation::E = AllocatingEvaluation(), kwargs...
    ) where {E <: AbstractEvaluationType}
    cfs = ClosedFormSubSolverState(; evaluation = evaluation)
    return TrustRegionsState(M, sub_problem, cfs; kwargs...)
end
function TrustRegionsState(
        M::AbstractManifold, mho::AbstractManifoldHessianObjective; p = rand(M), kwargs...
    )
    TpM = TangentSpace(M, copy(M, p))
    problem = DefaultManoptProblem(TpM, TrustRegionModelObjective(mho))
    state = TruncatedConjugateGradientState(TpM; X = get_gradient(M, mho, p))
    return TrustRegionsState(M, problem, state; p = p, kwargs...)
end
get_iterate(trs::TrustRegionsState) = trs.p
function set_iterate!(trs::TrustRegionsState, M, p)
    copyto!(M, trs.p, p)
    return trs
end
get_gradient(agst::TrustRegionsState) = agst.X
function set_gradient!(agst::TrustRegionsState, M, p, X)
    copyto!(M, agst.X, p, X)
    return agst
end

function get_message(dcs::TrustRegionsState)
    # for now only the sub solver might have messages
    return get_message(dcs.sub_state)
end
function show(io::IO, trs::TrustRegionsState)
    i = get_count(trs, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(trs.stop) ? "Yes" : "No"
    sub = repr(trs.sub_state)
    sub = replace(sub, "\n" => "\n    | ")
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
    * sub solver state     :
        | $(sub)

    ## Stopping criterion

    $(status_summary(trs.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end

_doc_TR = """
    trust_regions(M, f, grad_f, Hess_f, p=rand(M); kwargs...)
    trust_regions(M, f, grad_f, p=rand(M); kwargs...)
    trust_regions!(M, f, grad_f, Hess_f, p; kwargs...)
    trust_regions!(M, f, grad_f, p; kwargs...)

run the Riemannian trust-regions solver for optimization on manifolds to minimize `f`, see
on [AbsilBakerGallivan:2006, ConnGouldToint:2000](@cite).

For the case that no Hessian is provided, the Hessian is computed using finite differences,
see [`ApproxHessianFiniteDifference`](@ref).
For solving the inner trust-region subproblem of finding an update-vector,
by default the [`truncated_conjugate_gradient_descent`](@ref) is used.

# Input

$(_var(:Argument, :M; type = true))
$(_var(:Argument, :f))
$(_var(:Argument, :grad_f))
$(_var(:Argument, :Hess_f))
$(_var(:Argument, :p))

# Keyword arguments

* `acceptance_rate`:        accept/reject threshold: if ρ (the performance ratio for the iterate)
  is at least the acceptance rate ρ', the candidate is accepted.
  This value should  be between ``0`` and ``\frac{1}{4}``
* `augmentation_threshold=0.75`: trust-region augmentation threshold: if ρ is larger than this threshold,
  a solution is on the trust region boundary and negative curvature, and the radius is extended (augmented)
* `augmentation_factor=2.0`: trust-region augmentation factor
$(_var(:Keyword, :evaluation))
* `κ=0.1`: the linear convergence target rate of the tCG method
    [`truncated_conjugate_gradient_descent`](@ref), and is used in a stopping criterion therein
* `max_trust_region_radius`: the maximum trust-region radius
* `preconditioner`:       a preconditioner for the Hessian H.
  This is either an allocating function `(M, p, X) -> Y` or an in-place function `(M, Y, p, X) -> Y`,
  see `evaluation`, and by default set to the identity.
* `project!=copyto!`: for numerical stability it is possible to project onto the tangent space after every iteration.
  the function has to work inplace of `Y`, that is `(M, Y, p, X) -> Y`, where `X` and `Y` can be the same memory.
* `randomize=false`:      indicate whether `X` is initialised to a random vector or not.
  This disables preconditioning.
* `ρ_regularization=1e3`: regularize the performance evaluation ``ρ`` to avoid numerical inaccuracies.
* `reduction_factor=0.25`: trust-region reduction factor
* `reduction_threshold=0.1`: trust-region reduction threshold: if ρ is below this threshold,
  the trust region radius is reduced by `reduction_factor`.
$(_var(:Keyword, :retraction_method))
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(1000)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-6)`"))
$(_var(:Keyword, :sub_kwargs))
$(_var(:Keyword, :stopping_criterion, "sub_stopping_criterion"; default = "( see [`truncated_conjugate_gradient_descent`](@ref))"))
$(_var(:Keyword, :sub_problem; default = "[`DefaultManoptProblem`](@ref)`(M, `[`ConstrainedManifoldObjective`](@ref)`(subcost, subgrad; evaluation=evaluation))`"))
$(_var(:Keyword, :sub_state; default = "[`QuasiNewtonState`](@ref)", add = " where [`QuasiNewtonLimitedMemoryDirectionUpdate`](@ref) with [`InverseBFGS`](@ref) is used"))
* `θ=1.0`:                the superlinear convergence target rate of ``1+θ`` of the tCG-method
  [`truncated_conjugate_gradient_descent`](@ref), and is used in a stopping criterion therein
* `trust_region_radius=`[`injectivity_radius`](@extref `ManifoldsBase.injectivity_radius-Tuple{AbstractManifold}`)`(M) / 4`: the initial trust-region radius

For the case that no Hessian is provided, the Hessian is computed using finite difference, see
[`ApproxHessianFiniteDifference`](@ref).

$(_note(:OtherKeywords))

$(_note(:OutputSection))

# See also
[`truncated_conjugate_gradient_descent`](@ref)
"""

@doc "$(_doc_TR)"
trust_regions(M::AbstractManifold, args...; kwargs...)
# Hessian (Function) but no point
function trust_regions(
        M::AbstractManifold, f, grad_f, Hess_f::TH; kwargs...
    ) where {TH <: Function}
    return trust_regions(M, f, grad_f, Hess_f, rand(M); kwargs...)
end
# Hessian (Function) and point
function trust_regions(
        M::AbstractManifold,
        f,
        grad_f,
        Hess_f::TH,
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        preconditioner = if evaluation isa InplaceEvaluation
            (M, Y, p, X) -> (Y .= X)
        else
            (M, p, X) -> X
        end,
        kwargs...,
    ) where {TH <: Function}
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    grad_f_ = _ensure_mutating_gradient(grad_f, p, evaluation)
    Hess_f_ = _ensure_mutating_hessian(Hess_f, p, evaluation)
    preconditioner_ = _ensure_mutating_hessian(preconditioner, p, evaluation)
    mho = ManifoldHessianObjective(
        f_, grad_f_, Hess_f_, preconditioner_; evaluation = evaluation
    )
    rs = trust_regions(M, mho, p_; evaluation = evaluation, kwargs...)
    return _ensure_matching_output(p, rs)
end
# neither Hessian (Function) nor point
function trust_regions(M::AbstractManifold, f, grad_f; kwargs...)
    return trust_regions(M, f, grad_f, rand(M); kwargs...)
end
# no Hessian (Function), point (any)
function trust_regions(
        M::AbstractManifold,
        f::TF,
        grad_f::TdF,
        p;
        evaluation = AllocatingEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        kwargs...,
    ) where {TF, TdF}
    hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation = evaluation, retraction_method = retraction_method
    )
    return trust_regions(
        M,
        f,
        grad_f,
        hess_f,
        p;
        evaluation = evaluation,
        retraction_method = retraction_method,
        kwargs...,
    )
end
# Objective
function trust_regions(
        M::AbstractManifold, mho::O, p = rand(M); kwargs...
    ) where {O <: Union{ManifoldHessianObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(trust_regions; kwargs...)
    q = copy(M, p)
    return trust_regions!(M, mho, q; kwargs...)
end
calls_with_kwargs(::typeof(trust_regions)) = (trust_regions!,)

@doc "$(_doc_TR)"
trust_regions!(M::AbstractManifold, args...; kwargs...)

# No Hessian but a point (Any)
function trust_regions!(
        M::AbstractManifold,
        f,
        grad_f,
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        kwargs...,
    )
    hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation = evaluation, retraction_method = retraction_method
    )
    return trust_regions!(
        M,
        f,
        grad_f,
        hess_f,
        p;
        evaluation = evaluation,
        retraction_method = retraction_method,
        kwargs...,
    )
end
# Hessian and point
function trust_regions!(
        M::AbstractManifold,
        f,
        grad_f,
        Hess_f::TH,
        p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        preconditioner = if evaluation isa InplaceEvaluation
            (M, Y, p, X) -> (Y .= X)
        else
            (M, p, X) -> X
        end,
        kwargs...,
    ) where {TH <: Function}
    mho = ManifoldHessianObjective(f, grad_f, Hess_f, preconditioner; evaluation = evaluation)
    return trust_regions!(M, mho, p; evaluation = evaluation, kwargs...)
end
# Objective
function trust_regions!(
        M::AbstractManifold,
        mho::O,
        p;
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        stopping_criterion::StoppingCriterion = StopAfterIteration(1000) |
            StopWhenGradientNormLess(1.0e-6),
        max_trust_region_radius::R = sqrt(manifold_dimension(M)),
        trust_region_radius::R = max_trust_region_radius / 8,
        randomize::Bool = false, # Deprecated, remove on next release (use just `σ`)
        project!::Proj = (copyto!),
        ρ_prime::R = 0.1, # Deprecated, remove on next breaking change (use `acceptance_rate``)
        acceptance_rate::R = ρ_prime,
        ρ_regularization = 1.0e3,
        θ::R = 1.0,
        κ::R = 0.1,
        σ = randomize ? 1.0e-3 : 0.0,
        reduction_threshold::R = 0.1,
        reduction_factor::R = 0.25,
        augmentation_threshold::R = 0.75,
        augmentation_factor::R = 2.0,
        sub_kwargs = (;),
        sub_objective = decorate_objective!(M, TrustRegionModelObjective(mho); sub_kwargs...),
        sub_problem = DefaultManoptProblem(TangentSpace(M, p), sub_objective),
        sub_stopping_criterion::StoppingCriterion = StopAfterIteration(manifold_dimension(M)) |
            StopWhenResidualIsReducedByFactorOrPower(;
            κ = κ, θ = θ
        ) |
            StopWhenTrustRegionIsExceeded() |
            StopWhenCurvatureIsNegative() |
            StopWhenModelIncreased(),
        sub_state::AbstractManoptSolverState = decorate_state!(
            TruncatedConjugateGradientState(
                TangentSpace(M, copy(M, p));
                X = zero_vector(M, p),
                θ = θ,
                κ = κ,
                trust_region_radius,
                randomize = randomize,
                (project!) = (project!),
                stopping_criterion = sub_stopping_criterion,
                sub_kwargs...,
            );
            sub_kwargs...,
        ),
        kwargs..., #collect rest
    ) where {Proj, O <: Union{ManifoldHessianObjective, AbstractDecoratedManifoldObjective}, R}
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
    keywords_accepted(trust_regions!; kwargs...)
    dmho = decorate_objective!(M, mho; kwargs...)
    dmp = DefaultManoptProblem(M, dmho)
    trs = TrustRegionsState(
        M,
        sub_problem,
        maybe_wrap_evaluation_type(sub_state);
        p = p,
        X = get_gradient(dmp, p),
        trust_region_radius = trust_region_radius,
        max_trust_region_radius = max_trust_region_radius,
        acceptance_rate = acceptance_rate,
        ρ_regularization = ρ_regularization,
        randomize = randomize,
        stopping_criterion = stopping_criterion,
        retraction_method = retraction_method,
        reduction_threshold = reduction_threshold,
        reduction_factor = reduction_factor,
        augmentation_threshold = augmentation_threshold,
        augmentation_factor = augmentation_factor,
        (project!) = (project!),
        σ = σ,
    )
    dtrs = decorate_state!(trs; kwargs...)
    solve!(dmp, dtrs)
    return get_solver_return(get_objective(dmp), dtrs)
end
calls_with_kwargs(::typeof(trust_regions!)) = (initialize_solver!, step_solver!)

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

function step_solver!(mp::AbstractManoptProblem, trs::TrustRegionsState, k)
    M = get_manifold(mp)
    mho = get_objective(mp)
    # Determine the initial tangent vector used as start point for the subsolvereta0
    if trs.σ > 0
        rand!(M, trs.Y; vector_at = trs.p, σ = trs.σ)
        nY = norm(M, trs.p, trs.Y)
        if nY > trs.trust_region_radius # move inside if outside
            trs.Y *= trs.trust_region_radius / (2 * nY)
        end
    else
        zero_vector!(M, trs.Y, trs.p)
    end
    # Update the current gradient
    get_gradient!(M, trs.X, mho, trs.p)
    set_parameter!(trs.sub_problem, :Manifold, :Basepoint, copy(M, trs.p))
    set_parameter!(trs.sub_state, :Iterate, copy(M, trs.p, trs.Y))
    set_parameter!(trs.sub_state, :TrustRegionRadius, trs.trust_region_radius)
    solve!(trs.sub_problem, trs.sub_state)
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
    # Update the Hessian approximation, unwrap the original Hessian function
    # and update it if it is an approximate Hessian.
    update_hessian!(M, get_hessian_function(mho, true), trs.p, trs.p_proposal, trs.Y)
    # Choose the new TR radius based on the model performance.
    # Case (a) performed poorly -> decrease radius
    if ρ < trs.reduction_threshold || !model_decreased || isnan(ρ)
        trs.trust_region_radius *= trs.reduction_factor
    elseif ρ > trs.augmentation_threshold &&
            get_parameter(trs.sub_state, :TrustRegionExceeded)
        # (b) performed great and exceed/reach the trust region boundary -> increase radius
        trs.trust_region_radius = min(
            trs.augmentation_factor * trs.trust_region_radius, trs.max_trust_region_radius
        )
    end
    # (c) decreased and performed well enough -> accept step
    if model_decreased && (ρ > trs.acceptance_rate)
        copyto!(trs.p, trs.p_proposal)
        # If working with approximate Hessian -> update base point there
        update_hessian_basis!(M, get_hessian_function(mho, true), trs.p)
    end
    return trs
end
