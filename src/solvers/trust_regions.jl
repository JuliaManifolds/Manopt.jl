@doc """
    TrustRegionsState <: AbstractHessianSolverState

Store the state of the trust-regions solver.

# Fields

* `acceptance_rate`:         a lower bound of the performance ratio for the iterate
  that decides if the iteration is accepted or not.
$(_fields(:callbacks; add_properties = [:as_dict]))
* `HX`, `HY`, `HZ`:          interim storage (to avoid allocation) of ``$(_tex(:Hess)) f(p)[⋅]`` for `X`, `Y`, `Z`
* `max_trust_region_radius`: the maximum trust-region radius
$(_fields(:p; add_properties = [:as_Iterate]))
* `project!`:                for numerical stability it is possible to project onto the tangent space after every iteration.
  the function has to work inplace of `Y`, that is `(M, Y, p, X) -> Y`, where `X` and `Y` can be the same memory.
$(_fields(:stopping_criterion; name = "stop"))
* `randomize`:               indicate whether `X` is initialized to a random vector or not
* `ρ_regularization`:        regularize the model fitness ``ρ`` to avoid division by zero
$(_fields([:sub_problem, :sub_state]))
* `σ`:                       Gaussian standard deviation when creating the random initial tangent vector
  Defaults to `0` unless `randomize` is set; a value of `0` disables the randomized (Cauchy point) mode.
* `trust_region_radius`: the trust-region radius
$(_fields(:X))
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
  where `evaluation` determines how to call the sub function. It is expected to be of the form
  `(M, Y, p, Δ) -> Y` for the in-place and `(M, p, Δ) -> Y` for the allocating `evaluation`,
  that is it minimizes the model within the trust region of radius `Δ` around `p`.

# Input

$(_args([:M, :sub_problem, :sub_state]))

## Keyword arguments

* `acceptance_rate=0.1`
$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
* `max_trust_region_radius=sqrt(manifold_dimension(M))`
$(_kwargs(:p; add_properties = [:as_Initial]))
* `project!=copyto!`
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-6)"))
* `randomize=false`
* `ρ_regularization=1000.0`
* `trust_region_radius=max_trust_region_radius / 8`
$(_kwargs(:X; add_properties = [:as_Memory]))

# See also

[`trust_regions`](@ref)
"""
mutable struct TrustRegionsState{
        P, T, Pr, St <: AbstractManoptSolverState, C <: AbstractDict{Symbol},
        SC <: StoppingCriterion, RTR <: AbstractRetractionMethod, R <: Real, Proj,
    } <: AbstractSubProblemSolverState
    callbacks::C
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
    σ::R
    reduction_threshold::R
    reduction_factor::R
    augmentation_threshold::R
    augmentation_factor::R
    # Only required for Random mode Random
    HX::T
    Y::T
    HY::T
    Z::T
    HZ::T
    τ::R
    function TrustRegionsState(
            sub_problem::Pr, sub_state::St;
            callbacks::C = Dict{Symbol, Function}(),
            p::P, X::T,
            trust_region_radius::R, max_trust_region_radius::R, acceptance_rate::R,
            ρ_regularization::R, randomize::Bool,
            stopping_criterion::SC, retraction_method::RTR, reduction_threshold::R,
            augmentation_threshold::R, project!::Proj = (copyto!),
            reduction_factor::R, augmentation_factor::R, σ::R,
            #random mode ones can stay uninitielized if not provided
            HX::Union{T, Nothing} = nothing,
            Y::Union{T, Nothing} = nothing,
            HY::Union{T, Nothing} = nothing,
            Z::Union{T, Nothing} = nothing,
            HZ::Union{T, Nothing} = nothing,
            τ::Union{R, Nothing} = nothing,
        ) where {
            P, T, Pr, St <: AbstractManoptSolverState, C <: AbstractDict{Symbol},
            SC <: StoppingCriterion, RTR <: AbstractRetractionMethod, R <: Real, Proj,
        }
        trs = new{P, T, Pr, St, C, SC, RTR, R, Proj}()
        trs.callbacks = callbacks
        trs.p = p
        trs.X = X
        trs.stop = stopping_criterion
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
        !isnothing(HX) && (trs.HX = HX)
        !isnothing(Y) && (trs.Y = Y)
        !isnothing(HY) && (trs.HY = HY)
        !isnothing(Z) && (trs.Z = Z)
        !isnothing(HZ) && (trs.HZ = HZ)
        !isnothing(τ) && (trs.τ = τ)
        return trs
    end
end
TrustRegionsState(M::AbstractManifold, st::AbstractManoptSolverState; kwargs...) = error("Trust region method state can not be constructed based on $M and the sub state $st, a sub_problem is missing")
function TrustRegionsState(
        M::AbstractManifold, sub_problem::Pr, sub_state::St;
        p::P = rand(M), X::T = zero_vector(M, p),
        callbacks::C = Dict{Symbol, Function}(),
        acceptance_rate::Real = 0.1, ρ_regularization::Real = 1000.0,
        randomize::Bool = false,
        stopping_criterion::SC = StopAfterIteration(1000) | StopWhenGradientNormLess(1.0e-6),
        max_trust_region_radius::Real = sqrt(manifold_dimension(M)),
        trust_region_radius::Real = max_trust_region_radius / 8,
        retraction_method::RTR = default_retraction_method(M, typeof(p)),
        reduction_threshold::Real = 0.1, reduction_factor = 0.25,
        augmentation_threshold::Real = 0.75, augmentation_factor::Real = 2.0,
        project!::Proj = (copyto!), σ::Real = randomize ? 1.0e-4 : 0.0,
    ) where {
        P, T, Pr <: Union{AbstractManoptProblem, F} where {F}, St <: AbstractManoptSolverState,
        C <: AbstractDict{Symbol}, SC <: StoppingCriterion, RTR <: AbstractRetractionMethod, Proj,
    }
    R = promote_type(
        typeof(acceptance_rate), typeof(ρ_regularization), typeof(max_trust_region_radius),
        typeof(trust_region_radius), typeof(reduction_threshold), typeof(reduction_factor),
        typeof(augmentation_factor), typeof(augmentation_threshold), typeof(σ)
    )
    acceptance_rate = convert(R, acceptance_rate); ρ_regularization = convert(R, ρ_regularization)
    max_trust_region_radius = convert(R, max_trust_region_radius); trust_region_radius = convert(R, trust_region_radius)
    reduction_threshold = convert(R, reduction_threshold); reduction_factor = convert(R, reduction_factor)
    augmentation_factor = convert(R, augmentation_factor); augmentation_threshold = convert(R, augmentation_threshold)
    σ = convert(R, σ)

    return TrustRegionsState(
        sub_problem, sub_state;
        p = p, X = X, callbacks = callbacks,
        trust_region_radius = trust_region_radius, max_trust_region_radius = max_trust_region_radius,
        acceptance_rate = acceptance_rate, ρ_regularization = ρ_regularization,
        (project!) = project!, randomize = randomize, σ = σ,
        stopping_criterion = stopping_criterion, retraction_method = retraction_method,
        reduction_threshold = reduction_threshold, augmentation_threshold = augmentation_threshold,
        reduction_factor = reduction_factor, augmentation_factor = augmentation_factor,
    )
end
function TrustRegionsState(
        M::AbstractManifold, sub_problem, sub_state::AbstractEvaluationType; kwargs...
    )
    return TrustRegionsState(M, sub_problem; evaluation = sub_state, kwargs...)
end
function TrustRegionsState(
        M::AbstractManifold, sub_problem; evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...
    )
    sub_problem_ = maybe_wrap_function(sub_problem, evaluation)
    cfs = ClosedFormSubSolverState()
    return TrustRegionsState(M, sub_problem_, cfs; kwargs...)
end
function TrustRegionsState(
        M::AbstractManifold, mho::AbstractManifoldHessianObjective; p = rand(M), kwargs...
    )
    TpM = TangentSpace(M, copy(M, p))
    problem = DefaultManoptProblem(TpM, TrustRegionModelObjective(mho))
    state = TruncatedConjugateGradientState(TpM; X = get_gradient(M, mho, p))
    return TrustRegionsState(M, problem, state; p = p, kwargs...)
end
get_callbacks(trs::TrustRegionsState) = trs.callbacks
get_gradient(trs::TrustRegionsState) = trs.X
function get_message(trs::TrustRegionsState)
    # for now only the sub solver might have messages
    return get_message(trs.sub_state)
end
get_iterate(trs::TrustRegionsState) = trs.p
provided_callbacks(::Type{<:TrustRegionsState}) = union(_MANOPT_DEFAULT_CALLBACKS, [:Subsolver])

function set_gradient!(agst::TrustRegionsState, M, p, X)
    copyto!(M, agst.X, p, X)
    return agst
end
function set_iterate!(trs::TrustRegionsState, M, p)
    copyto!(M, trs.p, p)
    return trs
end
function Base.show(io::IO, trs::TrustRegionsState)
    print(io, "TrustRegionsState("); print(io, trs.sub_problem); print(io, ", "); print(io, trs.sub_state)
    print(io, "; ")
    print(io, "p = $(trs.p), X = $(trs.X), ")
    print(io, "callbacks = ", trs.callbacks, ", ")
    print(io, "trust_region_radius = $(trs.trust_region_radius), max_trust_region_radius = $(trs.max_trust_region_radius), ")
    print(io, "acceptance_rate = $(trs.acceptance_rate), ρ_regularization = $(trs.ρ_regularization), randomize = $(trs.randomize), ")
    print(io, "reduction_threshold = $(trs.reduction_threshold), augmentation_threshold = $(trs.augmentation_threshold), ")
    print(io, "(project!) = $(trs.project!), reduction_factor = $(trs.reduction_factor), augmentation_factor = $(trs.augmentation_factor), σ = $(trs.σ), ")
    isdefined(trs, :HX) && print(io, "HX = $(trs.HX), ")
    isdefined(trs, :Y) && print(io, "Y = $(trs.Y), ")
    isdefined(trs, :HY) && print(io, "HY = $(trs.HY), ")
    isdefined(trs, :Z) && print(io, "Z = $(trs.Z), ")
    isdefined(trs, :HZ) && print(io, "HZ = $(trs.HZ), ")
    isdefined(trs, :τ) && print(io, "τ = $(trs.τ), ")
    print(io, "stopping_criterion = $(trs.stop), retraction_method = $(trs.retraction_method)")
    return print(io, ")")
end
function status_summary(trs::TrustRegionsState; context::Symbol = :default)
    (context === :short) && return repr(trs)
    i = get_count(trs, :Iterations)
    conv_inl = (i > 0) ? (has_converged(trs.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the trust region solver$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(trs.stop) ? "Yes" : "No"
    (context === :inline) && (return "A trust regions method state – $(Iter) $(has_converged(trs) ? "(converged)" : "")")
    sub = _in_str(status_summary(trs.sub_state; context = context); indent = 1, headers = 1, indent_end = "| ")
    as = _callbacks_summary(trs)
    s = """
    # Solver state for `Manopt.jl`s Trust Region Method
    $Iter
    ## Parameters
    * acceptance_rate (ρ'):   $(trs.acceptance_rate)$(as)
    * augmentation threshold: $(trs.augmentation_threshold) (factor: $(trs.augmentation_factor))
    * randomize:              $(trs.randomize)
    * reduction threshold:    $(trs.reduction_threshold) (factor: $(trs.reduction_factor))
    * retraction method:      $(trs.retraction_method)
    * ρ_regularization:       $(trs.ρ_regularization)
    * trust region radius:    $(trs.trust_region_radius) (max: $(trs.max_trust_region_radius))
    * sub solver state:
    $(sub)

    ## Stopping criterion
    $(_in_str(status_summary(trs.stop; context = context); indent = 1, headers = 1))
    The algorithm converged: $Conv"""
    return s
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

$(_args([:M, :f, :grad_f, :Hess_f, :p]))

# Keyword arguments

* `acceptance_rate`:        accept/reject threshold: if ρ (the performance ratio for the iterate)
  is at least the acceptance rate ρ', the candidate is accepted.
  This value should be between ``0`` and ``$(_tex(:frac, "1", "4"))``
* `augmentation_threshold=0.75`: trust-region augmentation threshold: if ρ is larger than this threshold,
  a solution is on the trust region boundary and negative curvature, and the radius is extended (augmented)
* `augmentation_factor=2.0`: trust-region augmentation factor
$(_kwargs(:callbacks; add_properties = [:process_note]))
$(_kwargs(:evaluation))
* `κ=0.1`: the linear convergence target rate of the tCG method
    [`truncated_conjugate_gradient_descent`](@ref), and is used in a stopping criterion therein
* `max_trust_region_radius=sqrt(manifold_dimension(M))`: the maximum trust-region radius
* `preconditioner`:       a preconditioner for the Hessian H.
  This is either an allocating function `(M, p, X) -> Y` or an in-place function `(M, Y, p, X) -> Y`,
  see `evaluation`, and by default set to the identity.
* `project!=copyto!`: for numerical stability it is possible to project onto the tangent space after every iteration.
  the function has to work inplace of `Y`, that is `(M, Y, p, X) -> Y`, where `X` and `Y` can be the same memory.
* `randomize=false`:      indicate whether `X` is initialized to a random vector or not.
  This disables preconditioning.
* `ρ_regularization=1e3`: regularize the performance evaluation ``ρ`` to avoid numerical inaccuracies.
* `reduction_factor=0.25`: trust-region reduction factor
* `reduction_threshold=0.1`: trust-region reduction threshold: if ρ is below this threshold,
  the trust region radius is reduced by `reduction_factor`.
$(_kwargs(:retraction_method))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(1000)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-6)"))
$(_kwargs(:sub_kwargs))
$(_kwargs(:stopping_criterion; name = "sub_stopping_criterion", default = "`( see [`truncated_conjugate_gradient_descent`](@ref))` "))
* `sub_objective`: the sub objective to solve, by default the [`TrustRegionModelObjective`](@ref)`(mho)` possibly decorated with `sub_kwargs`
  Note that this keyword has no effect if you set the `sub_problem` directly.
$(_kwargs(:sub_problem; default = "`[`DefaultManoptProblem`](@ref)`(`[`TangentSpace`](@extref `ManifoldsBase.TangentSpace`)`(M,p), sub_objective)"))
$(_kwargs(:sub_state; default = "`[`TruncatedConjugateGradientState`](@ref)` "))
  , see also [`truncated_conjugate_gradient_descent`](@ref) for more details
* `θ=1.0`:                the superlinear convergence target rate of ``1+θ`` of the tCG-method
  [`truncated_conjugate_gradient_descent`](@ref), and is used in a stopping criterion therein
* `trust_region_radius=max_trust_region_radius / 8`: the initial trust-region radius

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
        M::AbstractManifold, f, grad_f, Hess_f::TH, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        preconditioner = if evaluation isa InplaceEvaluation
            (M, Y, p, X) -> (Y .= X)
        else
            (M, p, X) -> X
        end,
        kwargs...,
    ) where {TH <: Function}
    mho = ManifoldHessianObjective(
        f, grad_f, Hess_f, preconditioner; p = p, evaluation = evaluation
    )
    p_ = maybe_wrap_variable(p)
    rs = trust_regions(M, mho, p_; evaluation = evaluation, kwargs...)
    return maybe_unwrap_variable(p, rs)
end
# neither Hessian (Function) nor point
function trust_regions(M::AbstractManifold, f, grad_f; kwargs...)
    return trust_regions(M, f, grad_f, rand(M); kwargs...)
end
# no Hessian (Function), point (any)
function trust_regions(
        M::AbstractManifold, f::TF, grad_f::TdF, p;
        evaluation = AllocatingEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        kwargs...,
    ) where {TF, TdF}
    hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation = evaluation, retraction_method = retraction_method
    )
    return trust_regions(
        M, f, grad_f, hess_f, p;
        evaluation = evaluation, retraction_method = retraction_method, kwargs...,
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
        M::AbstractManifold, f, grad_f, p;
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        kwargs...,
    )
    hess_f = ApproxHessianFiniteDifference(
        M, copy(M, p), grad_f; evaluation = evaluation, retraction_method = retraction_method
    )
    return trust_regions!(
        M, f, grad_f, hess_f, p;
        evaluation = evaluation, retraction_method = retraction_method, kwargs...,
    )
end
# Hessian and point
function trust_regions!(
        M::AbstractManifold, f, grad_f, Hess_f::TH, p;
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
        M::AbstractManifold, mho::O, p;
        callbacks = Dict{Symbol, Function}(),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        stopping_criterion::StoppingCriterion = StopAfterIteration(1000) |
            StopWhenGradientNormLess(1.0e-6),
        max_trust_region_radius::Real = sqrt(manifold_dimension(M)),
        trust_region_radius::Real = max_trust_region_radius / 8,
        randomize::Bool = false,
        project!::Proj = (copyto!),
        ρ_prime::Real = 0.1, # Deprecated, remove on next breaking change (use `acceptance_rate`)
        acceptance_rate::Real = ρ_prime,
        ρ_regularization::Real = 1.0e3,
        θ::Real = 1.0,
        κ::Real = 0.1,
        σ::Real = randomize ? 1.0e-3 : 0.0,
        reduction_threshold::Real = 0.1,
        reduction_factor::Real = 0.25,
        augmentation_threshold::Real = 0.75,
        augmentation_factor::Real = 2.0,
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
                X = zero_vector(M, p), θ = θ, κ = κ,
                trust_region_radius, randomize = (σ > 0), (project!) = (project!),
                stopping_criterion = sub_stopping_criterion,
                sub_kwargs...,
            );
            sub_kwargs...,
        ),
        kwargs..., #collect rest
    ) where {Proj, O <: Union{ManifoldHessianObjective, AbstractDecoratedManifoldObjective}}
    R = float(
        promote_type(
            typeof(max_trust_region_radius), typeof(trust_region_radius),
            typeof(acceptance_rate), typeof(ρ_regularization),
            typeof(θ), typeof(κ), typeof(σ),
            typeof(reduction_threshold), typeof(reduction_factor),
            typeof(augmentation_threshold), typeof(augmentation_factor),
        ),
    )
    max_trust_region_radius = convert(R, max_trust_region_radius)
    trust_region_radius = convert(R, trust_region_radius)
    acceptance_rate = convert(R, acceptance_rate)
    ρ_regularization = convert(R, ρ_regularization)
    θ = convert(R, θ)
    κ = convert(R, κ)
    σ = convert(R, σ)
    reduction_threshold = convert(R, reduction_threshold)
    reduction_factor = convert(R, reduction_factor)
    augmentation_threshold = convert(R, augmentation_threshold)
    augmentation_factor = convert(R, augmentation_factor)
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
    # `randomize` requires a positive `σ` to have any effect, so keep the two consistent
    if randomize && (σ == 0)
        @warn "`randomize=true` has no effect for `σ=0`; the randomized (Cauchy point) mode is disabled. Pass a positive `σ` to enable it."
        randomize = false
    end
    keywords_accepted(trust_regions!; kwargs...)
    dmho = decorate_objective!(M, mho; kwargs...)
    dmp = DefaultManoptProblem(M, dmho)
    trs = TrustRegionsState(
        M, sub_problem, sub_state;
        callbacks = process_callbacks_arg(callbacks, TrustRegionsState),
        p = p, X = get_gradient(dmp, p),
        trust_region_radius = trust_region_radius,
        max_trust_region_radius = max_trust_region_radius,
        acceptance_rate = acceptance_rate,
        ρ_regularization = ρ_regularization,
        randomize = randomize,
        stopping_criterion = stopping_criterion,
        retraction_method = retraction_method,
        reduction_threshold = reduction_threshold, reduction_factor = reduction_factor,
        augmentation_threshold = augmentation_threshold, augmentation_factor = augmentation_factor,
        (project!) = (project!),
        σ = σ,
    )
    dtrs = decorate_state!(trs; kwargs...)
    solve!(dmp, dtrs)
    return get_solver_return(get_objective(dmp), dtrs)
end
calls_with_kwargs(::typeof(trust_regions!)) = (decorate_objective!, decorate_state!)

function initialize_solver!(mp::AbstractManoptProblem, trs::TrustRegionsState)
    M = get_manifold(mp)
    get_gradient!(mp, trs.X, trs.p)
    trs.Y = zero_vector(M, trs.p)
    trs.HY = zero_vector(M, trs.p)
    trs.p_proposal = deepcopy(trs.p)
    trs.f_proposal = zero(trs.trust_region_radius)
    if trs.σ > 0 #only init if necessary
        trs.Z = zero_vector(M, trs.p)
        trs.HZ = zero_vector(M, trs.p)
        trs.τ = zero(trs.trust_region_radius)
        trs.HX = zero_vector(M, trs.p)
    end
    return trs
end

# Obtain H[Y] after the sub solver ran: a tCG sub state already provides it,
# for any other sub solver it has to be computed.
#=
    Variant I: the sub task is a problem that is solved by a sub solver
=#
function _trs_solve_sub!(M, trs::TrustRegionsState, ::AbstractManoptSolverState)
    set_parameter!(trs.sub_problem, Val(:Manifold), Val(:Basepoint), copy(M, trs.p))
    set_parameter!(trs.sub_state, Val(:Iterate), copy(M, trs.p, trs.Y))
    set_parameter!(trs.sub_state, Val(:TrustRegionRadius), trs.trust_region_radius)
    solve!(trs.sub_problem, trs.sub_state)
    return copyto!(M, trs.Y, trs.p, get_solver_result(trs.sub_state))
end
#=
    Variant II: the sub task is a function providing a closed form solution
=#
function _trs_solve_sub!(M, trs::TrustRegionsState, ::ClosedFormSubSolverState)
    return trs.sub_problem(M, trs.Y, trs.p, trs.trust_region_radius)
end
function _trs_get_HY!(M, trs::TrustRegionsState, mho, ::Any)
    # for Y = 0 the model's Hessian term vanishes, no need to evaluate
    (norm(M, trs.p, trs.Y) == 0) && return zero_vector!(M, trs.HY, trs.p)
    return get_hessian!(M, trs.HY, mho, trs.p, trs.Y)
end
function _trs_get_HY!(M, trs::TrustRegionsState, mho, sub::TruncatedConjugateGradientState)
    # an approximate Hessian is not linear in `X`, so the accumulated `HY` may differ from `H[Y]`
    # so in that case we also do not use the subsolvers result but evaluate anew.
    (get_hessian_function(mho, true) isa AbstractApproximateHessianFunction) && return _trs_get_HY!(M, trs, mho, nothing)
    return copyto!(M, trs.HY, trs.p, sub.HY)
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
    _trs_solve_sub!(M, trs, trs.sub_state)
    callback(:Subsolver, mp, trs, k)
    f = get_cost(mp, trs.p)
    _trs_get_HY!(M, trs, mho, get_state(trs.sub_state))
    if trs.σ > 0 # randomized approach: compare result with the Cauchy point.
        nX = norm(M, trs.p, trs.X)
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
            (get_parameter(get_state(trs.sub_state), :TrustRegionExceeded) === true)
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
