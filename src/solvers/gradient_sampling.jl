function gradient_sampling_subsolver end
function gradient_sampling_subsolver! end

_doc_gradient_sampling_subsolver = """
    λ = gradient_sampling_subsolver(M, p, sampled_gradients)
    gradient_sampling_subsolver!(M, λ, p, sampled_gradients)

solver for the subproblem of the [`gradient_sampling`](@ref) algorithm.

Let ``Y_j``, ``j=0,…m`` denote the `sampled_gradients`
already provided transported to the tangent space at `p`

The subproblem then reads
```math
\\begin{align*}
    $(_tex(:argmin))_{λ ∈ ℝ^{m+1}} &
    $(_tex(:Bigr))\\lVert $(_tex(:sum, "j = 0", "m")) λ_j Y_{j}$(_tex(:Bigr))\\rVert^2
    \\\\
    $(_tex(:text, "s. t.")) $(_tex(:quad)) &
    $(_tex(:sum, "j = 0", "m")) λ_j = 1,
    $(_tex(:quad)) λ_j ≥ 0
    $(_tex(:quad)) $(_tex(:text, "for all ")) j =0,…,m.
\\end{align*}
```

!!! tip
    A default subsolver based on [`RipQP`.jl](https://github.com/JuliaSmoothOptimizers/RipQP.jl) and [`QuadraticModels`](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl)
    is available if these two packages are loaded.
"""

@doc "$(_doc_gradient_sampling_subsolver)"
gradient_sampling_subsolver(M::AbstractManifold, p, sampled_gradients)

@doc "$(_doc_gradient_sampling_subsolver)"
gradient_sampling_subsolver!(M::AbstractManifold, λ, p, sampled_gradients)
"""
    GradientSamplingState

A state for the [gradient sampling algorithm](@ref gradient_sampling).
The mathematical symbols are adapted from [HosseiniUschmajew:2017](@cite)

# Fields
$(_fields(:callbacks; add_properties = [:as_dict]))
* `convex_hull_coeffs<:AbstractVector{R}` store the solution vector of the sub problem, i.e. the coefficients of the result in the convex hull
$(_fields(:p; add_properties = [:as_Iterate]))
* `sampled_points<:AbstractVector{P}` memory to store the vector of sampled points
* `sampled_vectors<:AbstractVector{T}` memory to store the vector of (transported) gradients
* `sampling_radius` radius ``ϵ_k`` of the ball around the iterate to sample from
* `sampling_radius_reduction` factor ``θ_ϵ`` to reduce the sampling radius when rejecting a step
* `subgradient_norm_reduction` factor ``θ_δ`` to reduce the subgradient norm tolerance when rejecting a step
* `subgradient_norm_tolerance` bound ``δ_k`` to reject too small gradient vector solutions from the sub problem
$(_fields(:stopping_criterion; name = "stop"))
$(_fields([:stepsize, :sub_problem, :sub_state, :retraction_method, :vector_transport_method]))
$(_fields(:X; add_properties = [:as_Gradient]))
$(_fields(:X; name = "Y")) a tangent vector to assemble the solution in.

# Constructor
    GradientSamplingState(
        M::AbstractManifold, sub_problem = gradient_sampling_subsolver!, sub_state = InplaceEvaluation(); kwargs...
    )

Create a gradient sampling solver state

## Input

$(_args(:M))
$(_args(:sub_problem))
$(_args(:sub_state))

## Keyword arguments

$(_kwargs(:callbacks; show_type = false, add_properties = [:as_dict]))
$(_kwargs(:p; add_properties = [:as_Initial]))
$(_kwargs(:retraction_method))
* `sample_size = 5` set the number of sampling points. If you initialize `sampled_points`, `sampled_vectors`, and `convex_hull_coeffs` directly
  this parameter has no effect.
* `sampling_radius = 0.5`
* `sampling_radius_reduction = 0.5`
* `sampling_radius_threshold = 1.0e-2` a threshold ``ϵ_{$(_tex(:rm, "opt"))}`` to be used in the stopping criterion
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`GradientSamplingState`](@ref)`; retraction_method=retraction_method)"))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))([`StopWhenGradientNormLess`](@ref)`(subgradient_norm_threshold)`$(_sc(:All))[`StopWhenSmallerOrEqual`](@ref)`(:sampling_radius, sampling_radius_threshold))"))
* `subgradient_norm_reduction = 0.5`
* `subgradient_norm_tolerance = 0.1`
* `subgradient_norm_threshold = 1.0e-3` a threshold ``δ_{$(_tex(:rm, "opt"))}`` to be used in the stopping criterion
$(_kwargs(:vector_transport_method))
$(_kwargs(:X; add_properties = [:as_Memory]))
"""
mutable struct GradientSamplingState{
        P, T, R <: Real,
        Pr <: Union{F, AbstractManoptProblem} where {F}, St <: AbstractManoptSolverState,
        C <: AbstractDict{Symbol},
        SP <: AbstractVector{<:P}, ST <: AbstractVector{<:T}, A <: AbstractVector{<:R},
        SC <: StoppingCriterion, S <: Stepsize, RTM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod,
    } <: AbstractGradientSolverState
    callbacks::C
    convex_hull_coeffs::A
    p::P
    retraction_method::RTM
    sampled_points::SP
    sampled_vectors::ST
    sampling_radius::R # in HU17 εₗ
    sampling_radius_reduction::R # in HU17 θ_ε
    stepsize::S
    stop::SC
    subgradient_norm_reduction::R # in HU17 θ_δ
    subgradient_norm_tolerance::R # in HU17 δₗ
    sub_problem::Pr
    sub_state::St
    vector_transport_method::VTM
    X::T
    Y::T
    function GradientSamplingState(;
            callbacks::C = Dict{Symbol, Function}(),
            p::P, convex_hull_coeffs::A,
            sampled_points::SP, sampled_vectors::ST,
            sampling_radius::R, sampling_radius_reduction::R,
            subgradient_norm_reduction::R, subgradient_norm_tolerance::R,
            sub_problem::Pr, sub_state::St,
            stepsize::S, stopping_criterion::SC, retraction_method::RTM, vector_transport_method::VTM,
            X::T, Y::T,
        ) where {
            P, T, R <: Real,
            SP <: AbstractVector, ST <: AbstractVector, A <: AbstractVector,
            Pr <: Union{G, AbstractManoptProblem} where {G}, St <: AbstractManoptSolverState,
            SC <: StoppingCriterion, S <: Stepsize, RTM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod,
            C <: AbstractDict{Symbol},
        }
        return new{P, T, R, Pr, St, C, SP, ST, A, SC, S, RTM, VTM}(
            callbacks, convex_hull_coeffs,
            p, retraction_method,
            sampled_points, sampled_vectors, sampling_radius, sampling_radius_reduction,
            stepsize, stopping_criterion, subgradient_norm_reduction, subgradient_norm_tolerance,
            sub_problem, sub_state, vector_transport_method,
            X, Y,
        )
    end
end
function GradientSamplingState(
        M::AbstractManifold, sub_problem, sub_state::AbstractEvaluationType; kwargs...
    )
    return GradientSamplingState(M, sub_problem; evaluation = sub_state, kwargs...)
end
function GradientSamplingState(
        M::AbstractManifold, sub_problem = gradient_sampling_subsolver!; evaluation::AbstractEvaluationType = AllocatingEvaluation(), kwargs...
    )
    sub_problem_ = maybe_wrap_function(sub_problem, evaluation; result = :Point)
    return GradientSamplingState(M, sub_problem_, ClosedFormSubSolverState(); kwargs...)
end
function GradientSamplingState(
        M::AbstractManifold, sub_problem::Pr, sub_state::St;
        callbacks::C = Dict{Symbol, Function}(),
        p::P = rand(M),
        X::T = zero_vector(M, p),
        retraction_method::RTM = default_retraction_method(M, typeof(p)),
        sample_size::Int = 5,
        sampled_points::Vector{P} = [copy(M, p) for _ in 1:(sample_size + 1)],
        sampled_vectors::Vector{T} = [copy(M, p, X) for _ in 1:(sample_size + 1)],
        sampling_radius::R = 0.5,
        convex_hull_coeffs::Vector{R} = [zero(R) for _ in 1:(sample_size + 1)],
        sampling_radius_reduction::R = 0.5,
        sampling_radius_threshold::R = 1.0e-2,
        subgradient_norm_reduction::R = 0.5,
        subgradient_norm_tolerance::R = 0.1,
        subgradient_norm_threshold::R = 1.0e-3,
        stopping_criterion::SC = StopAfterIteration(200) | (
            StopWhenGradientNormLess(subgradient_norm_threshold) & (
                StopWhenSmallerOrEqual(:sampling_radius, sampling_radius_threshold)
            )
        ),
        stepsize::S = default_stepsize(
            M, GradientSamplingState; retraction_method = retraction_method
        ),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
    ) where {P, T, R <: Real, SC <: StoppingCriterion, S <: Stepsize, RTM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod, C <: AbstractDict{Symbol}, Pr <: Union{G, AbstractManoptProblem} where {G}, St <: AbstractManoptSolverState}
    m1 = length(sampled_points)
    m2 = length(sampled_vectors)
    m3 = length(convex_hull_coeffs)
    ((m1 != m2) || (m2 != m3)) && throw(
        ErrorException(
            """
            The temporary storage of points ($(m1)) and vectors ($(m2)) or coefficients ($(m3)) does not agree in length.
            It should be automatically generated with length $(sample_size + 1). Make sure you pass the correct amounts of temporary storages.
            """
        )
    )
    return GradientSamplingState(;
        callbacks = callbacks,
        convex_hull_coeffs = convex_hull_coeffs,
        p = p,
        sampled_points = sampled_points, sampled_vectors = sampled_vectors,
        sampling_radius = sampling_radius, sampling_radius_reduction = sampling_radius_reduction,
        subgradient_norm_tolerance = subgradient_norm_tolerance, subgradient_norm_reduction = subgradient_norm_reduction,
        sub_problem = sub_problem, sub_state = sub_state,
        stepsize = stepsize, stopping_criterion = stopping_criterion, retraction_method = retraction_method,
        vector_transport_method = vector_transport_method, X = X, Y = copy(M, p, X),
    )
end
function default_stepsize(
        M::AbstractManifold, ::Type{GradientSamplingState}; retraction_method = default_retraction_method(M),
    )
    return ArmijoLinesearchStepsize(M; retraction_method = retraction_method)
end
#
#
# Accessors
get_iterate(gss::GradientSamplingState) = gss.p
get_solver_result(gss::GradientSamplingState) = gss.p
get_gradient(gss::GradientSamplingState) = gss.X
provided_callbacks(::Type{<:GradientSamplingState}) = union(_MANOPT_DEFAULT_CALLBACKS, [:BeforeSubsolver, :Stepsize, :Subsolver])
get_callbacks(gss::GradientSamplingState) = gss.callbacks

function Base.show(io::IO, gss::GradientSamplingState)
    print(io, "GradientSamplingState(; ")
    print(io, "callbacks = ", gss.callbacks, ", ")
    print(io, "convex_hull_coeffs = ", gss.convex_hull_coeffs, ", p = ", gss.p, ", ")
    print(io, "sampled_point = ", gss.sampled_points, ", sampled_vectors = ", gss.sampled_vectors, ", ")
    print(io, "sampling_radius = ", gss.sampling_radius, ", sampling_radius_reduction = ", gss.sampling_radius_reduction, ", ")
    print(io, "subgradient_norm_reduction = ", gss.subgradient_norm_reduction, ", subgradient_norm_tolerance = ", gss.subgradient_norm_tolerance, ", ")
    print(io, "sub_problem = ", gss.sub_problem, ", sub_state = ", gss.sub_state, ", ")
    print(io, "stepsize = ", gss.stepsize, ", stopping_criterion = ", gss.stop, ", ")
    print(io, "retraction_method = ", gss.retraction_method, " vector_transport_method = ", gss.vector_transport_method, ", ")
    print(io, "X = ", gss.X, "m Y = ", gss.Y)
    return print(io, ")")
end

function status_summary(gss::GradientSamplingState; context::Symbol = :default)
    (context === :short) && return repr(gss)
    i = get_count(gss, :Iterations)
    conv_inl = (i > 0) ? (has_converged(gss.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the gradient sampling solver$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = has_converged(gss.stop) ? "Yes" : "No"
    as = _callbacks_summary(gss)
    s = """
    # Solver state for `Manopt.jl`s Gradient Sampling Algorithm
    $Iter
    ## Parameters$(as)
    * retraction method:         $(_MANOPT_INDENT)$(gss.retraction_method)
    * sampling radius:           $(_MANOPT_INDENT)$(gss.sampling_radius)
    * sampling radius reduction: $(_MANOPT_INDENT)$(gss.sampling_radius_reduction)
    * subgradient_norm_reduction:$(_MANOPT_INDENT)$(gss.subgradient_norm_reduction)
    * subgradient_norm_tolerance:$(_MANOPT_INDENT)$(gss.subgradient_norm_tolerance)
    * vector transport method:   $(_MANOPT_INDENT)$(gss.vector_transport_method)

    ## Stepsize
    $(_in_str(status_summary(gss.stepsize; context = context); indent = 1, headers = 1))

    ## Stopping criterion
    $(_in_str(status_summary(gss.stop; context = context); indent = 1, headers = 1))
    The algorithm converged: $Conv"""
    return s
end

_doc_gradient_sampling = """
    gradient_sampling(M, f, grad_f, p=rand(M); kwargs...)
    gradient_sampling(M, gradient_objective, p=rand(M); kwargs...)
    gradient_sampling!(M, f, grad_f, p; kwargs...)
    gradient_sampling!(M, gradient_objective, p; kwargs...)

perform the gradient sampling algorithm as introduced in [HosseiniUschmajew:2017](@cite).

The algorithm samples a set of `sampling_size` = ``m`` many points in a ball around the current iterate,
evaluates the gradient at these points and transports these to the current iterate.
It then builds a surrogate in the tangent space consisting of these ``m`` tangent vectors
and the gradient at the current iterate to determine a new descent direction in the convex
hull of these. See [`gradient_sampling_subsolver`](@ref) for the actual sub problem that is solved.
If this  direction exceeds, the `subgradient_norm_tolerance` the step is rejected and both
the ball radius and this tolerance are reduced.

# Input

$(_args([:M, :f, :grad_f, :p]))

$(_note(:GradientObjective))

# Keyword arguments

$(_kwargs(:callbacks; add_properties = [:process_note]))
$(_kwargs(:differential))
$(_kwargs(:evaluation; add_properties = [:GradientExample]))
$(_kwargs(:retraction_method))
* `sample_size = `$(_link(:manifold_dimension))`+1` set the number of sampling points. If you initialize `sampled_points`, `sampled_vectors`, and `convex_hull_coeffs` directly
  this parameter has no effect.
* `sampling_radius = 0.5`
* `sampling_radius_reduction = 0.5`
* `sampling_radius_threshold = 1.0e-2` a threshold ``ϵ_{$(_tex(:rm, "opt"))}`` to be used in the stopping criterion
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`GradientSamplingState`](@ref)`; retraction_method=retraction_method)"))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(100)`$(_sc(:Any))([`StopWhenGradientNormLess`](@ref)`(subgradient_norm_threshold)`$(_sc(:All))[`StopWhenSmallerOrEqual`](@ref)`(:sampling_radius, sampling_radius_threshold))"))
* `subgradient_norm_reduction = 0.5`
* `subgradient_norm_tolerance = 0.1`
* `subgradient_norm_threshold = 1.0e-3` a threshold ``δ_{$(_tex(:rm, "opt"))}`` to be used in the stopping criterion
$(_kwargs(:vector_transport_method))
$(_kwargs(:X; add_properties = [:as_Gradient]))
"""

@doc "$(_doc_gradient_sampling)"
gradient_sampling(M::AbstractManifold, args...; kwargs...)

function gradient_sampling(
        M::AbstractManifold, f, grad_f, p = rand(M);
        differential = missing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    p_ = maybe_wrap_variable(p)
    mgo = ManifoldGradientObjective(f, grad_f; evaluation = evaluation, differential = differential)
    rs = gradient_sampling(M, mgo, p_; kwargs...)
    return maybe_unwrap_variable(p, rs)
end
function gradient_sampling(
        M::AbstractManifold, mgo::O, p = rand(M); kwargs...
    ) where {O <: Union{AbstractManifoldFirstOrderObjective, AbstractDecoratedManifoldObjective}}
    q = copy(M, p)
    return gradient_sampling!(M, mgo, q; kwargs...)
end
calls_with_kwargs(::typeof(gradient_sampling)) = (gradient_sampling!,)

"$(_doc_gradient_sampling)"
gradient_sampling!(M::AbstractManifold, args...; kwargs...)

function gradient_sampling!(
        M::AbstractManifold, f, grad_f, p;
        differential = missing, evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    keywords_accepted(gradient_sampling; kwargs...)
    mgo = ManifoldGradientObjective(
        f, grad_f; differential = differential, evaluation = evaluation
    )
    return gradient_sampling!(M, mgo, p; kwargs...)
end
function gradient_sampling!(
        M::AbstractManifold, mgo::O, p;
        callbacks = Dict{Symbol, Function}(),
        X = zero_vector(M, p),
        sample_size::Int = manifold_dimension(M) + 1,
        convex_hull_coeffs::AbstractVector = [0.0 for _ in 1:(sample_size + 1)],
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        sampled_points::AbstractVector = [copy(M, p) for _ in 1:(sample_size + 1)],
        sampled_vectors::AbstractVector = [copy(M, p, X) for _ in 1:(sample_size + 1)],
        sampling_radius::Real = 0.5,
        sub_problem = gradient_sampling_subsolver!,
        sub_state = InplaceEvaluation(),
        sampling_radius_reduction::Real = 0.5, sampling_radius_threshold::Real = 1.0e-2,
        subgradient_norm_reduction::Real = 0.5, subgradient_norm_tolerance::Real = 0.1, subgradient_norm_threshold::Real = 1.0e-3,
        stopping_criterion::StoppingCriterion = StopAfterIteration(100) | (
            StopWhenGradientNormLess(subgradient_norm_threshold) & (
                StopWhenSmallerOrEqual(:sampling_radius, sampling_radius_threshold)
            )
        ),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = default_stepsize(
            M, GradientSamplingState; retraction_method = retraction_method
        ),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(M, typeof(p)),
        kwargs..., #collect rest
    ) where {O <: Union{AbstractManifoldFirstOrderObjective, AbstractDecoratedManifoldObjective}}
    # all explicit others from above are anyways accepted here, so we only have to pass kwargs in
    keywords_accepted(gradient_sampling!; kwargs...)
    dmgo = decorate_objective!(M, mgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    s = GradientSamplingState(
        M, sub_problem, sub_state;
        callbacks = process_callbacks_arg(callbacks, GradientSamplingState),
        p = p,
        sample_size = sample_size,
        convex_hull_coeffs = convex_hull_coeffs,
        retraction_method = retraction_method,
        sampled_points = sampled_points, sampled_vectors = sampled_vectors,
        sampling_radius = sampling_radius,
        sampling_radius_reduction = sampling_radius_reduction,
        subgradient_norm_reduction = subgradient_norm_reduction, subgradient_norm_tolerance = subgradient_norm_tolerance,
        stepsize = _produce_type(stepsize, M, p),
        stopping_criterion = stopping_criterion,
        vector_transport_method = vector_transport_method,
        X = X,
    )
    ds = decorate_state!(s; kwargs...)
    solve!(dmp, ds)
    return get_solver_return(get_objective(dmp), ds)
end
calls_with_kwargs(::typeof(gradient_sampling!)) = (decorate_objective!, decorate_state!)

#
#
# Solver implementation
function initialize_solver!(mp::AbstractManoptProblem, gss::GradientSamplingState)
    get_gradient!(mp, gss.X, gss.p)
    return gss
end

function step_solver!(
        mp::AbstractManoptProblem, gss::GradientSamplingState, i
    )
    M = get_manifold(mp)
    # resample on TpM, map to manifold and make sure they are within radius
    for (j, (pj, Xj)) in enumerate(zip(gss.sampled_points, gss.sampled_vectors))
        if j == 1 # add gradient as first element - > here we just copy the iterate over
            copyto!(M, pj, gss.p)
        else
            rand!(M, Xj; vector_at = gss.p, σ = gss.sampling_radius / 2)
            retract!(M, pj, gss.p, Xj, gss.retraction_method)
            while distance(M, gss.p, pj) > gss.sampling_radius
                Xj ./= 2
                retract!(M, pj, gss.p, Xj, gss.retraction_method)
            end
        end
    end
    # re-use the tangent vector memory to evaluate the gradients
    # and transport them to the current iterate
    for (i, (pj, Xj)) in enumerate(zip(gss.sampled_points, gss.sampled_vectors))
        get_gradient!(mp, Xj, pj) # we only have to transport the elements 2,3,...:
        (i > 1) && vector_transport_to!(M, Xj, pj, Xj, gss.p)
    end
    # solve sub problem in convex_hull_coeffs
    callback(:BeforeSubsolver, mp, gss, i)
    _gradient_sampling_subsolver(M, gss)
    callback(:Subsolver, mp, gss, i)
    # reconstruct tangent vector from the coefficients (w_l in HU17) in Y
    zero_vector!(M, gss.Y, gss.p)
    for (λj, Xj) in zip(gss.convex_hull_coeffs, gss.sampled_vectors)
        gss.Y .+= λj * Xj
    end
    # Decide whether to accept the step or update radius
    if norm(M, gss.p, gss.Y) < gss.subgradient_norm_tolerance
        # do not accept this step but decrease radius and tolerance
        gss.sampling_radius *= gss.sampling_radius_reduction
        gss.subgradient_norm_tolerance *= gss.subgradient_norm_reduction
    else
        # We already have the gradient in the sampled vectors[1]
        # and set normed -Y as search direction
        step = get_stepsize(mp, gss, i, -gss.Y / norm(M, gss.p, gss.Y); gradient = gss.sampled_vectors[1])
        callback(:Stepsize, mp, gss, i)
        ManifoldsBase.retract_fused!(M, gss.p, gss.p, -gss.Y / norm(M, gss.p, gss.Y), step, gss.retraction_method)
        get_gradient!(mp, gss.X, gss.p)
    end
    return gss
end
# closed form in-place
function _gradient_sampling_subsolver(
        M, gss::GradientSamplingState{P, T, R, F, ClosedFormSubSolverState}
    ) where {P, T, R, F}
    gss.sub_problem(M, gss.convex_hull_coeffs, gss.p, gss.sampled_vectors)
    return gss
end
# (c) (not yet needed / implemented) an actual sub solver call
