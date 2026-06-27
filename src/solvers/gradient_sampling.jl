# TODO Document all fields and refer to their names in the paper as well
#
"""
    GradientSamplingState

A state for the [gradient sampling algorithm](@ref gradient_sampling).

# Fields
$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:stopping_criterion; name = "stop"))
$(_fields([:stepsize, :retraction_method, :vector_transport_method]))
$(_fields([:sub_problem, :sub_state]))
$(_fields(:X; add_properties = [:as_Gradient]))

# Constructor

"""
mutable struct GradientSamplingState{
        P, T, R <: Real,
        SP <: AbstractVector{<:P}, ST <: AbstractVector{<:T},
        Pr <: Union{F, AbstractManoptProblem} where {F}, St <: AbstractManoptSolverState,
        SC <: StoppingCriterion, S <: Stepsize, RTM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod,
    }
    p::P
    sampled_points::SP
    sampled_vectors::ST
    sampling_radius::R # (In paper: εₗ)
    sampling_radius_reduction::R # in HU17 θ_ε
    sampling_radius_threshold::R # in HU17 εₒₚₜ)
    subgradient_norm_reduction::R # in HU17 θ_δ
    subgradient_norm_threshold::R # in HU17 δₒₚₜ)
    subgradient_norm_tolerance::R # in HU17 δₗ)
    sub_problem::Pr
    sub_state::St
    stepsize::S
    stop::SC
    retraction_method::RTM
    vector_transport_method::VTM
    X::T
    function GradientSamplingState(;
            p::P, X::T,
            sampled_point::SP, sampled_vectors::ST,
            sampling_radius::R, sampling_radius_reduction::R, sampling_radius_threshold::R,
            subgradient_norm_reduction::R, subgradient_norm_threshold::R, subgradient_norm_tolerance::R,
            sub_problem::Pr, sub_state::St,
            stepsize::S, stopping_criterion::SC, retraction_method::RTM, vector_transport_method::VTM
        ) where {
            P, T, R <: Real,
            SP <: AbstractVector, ST <: AbstractVector,
            Pr <: Union{G, AbstractManoptProblem} where {G}, St <: AbstractManoptSolverState,
            SC <: StoppingCriterion, S <: Stepsize, RTM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod,
        }
        return new{P, T, R, SP, ST, Pr, St, SC, S, RTM, VTM}(
            p, X, sampled_point, sampled_vectors,
            sampling_radius, sampling_radius_reduction, sampling_radius_threshold,
            subgradient_norm_reduction, subgradient_norm_threshold, subgradient_norm_tolerance,
            sub_problem, sub_state, stepsize, stopping_criterion, retraction_method, vector_transport_method
        )
    end
end

function GradientSamplingState(
        M::AbstractManifold;
        p::P = rand(M),
        X::T = zero_vector(M, p),
        retraction_method::RTM = default_retraction_method(M, typeof(p)),
        sample_size::Int = 5,
        sampled_points::Vector{P} = [copy(M, p) for _ in 1:sample_size],
        sampled_vectors::Vector{T} = [copy(M, p, X) for _ in 1:sample_size],
        sampling_radius::R = 0.5,
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
        # TODO: Can we maybe adapt / reuse the QP solver from convex / prox bundle?
        sub_problem = [],
        sub_state = [],
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p)),
    ) where {P, T, R <: Real, SC <: StoppingCriterion, S <: Stepsize, RTM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod}
    m1 = length(sampled_points)
    m2 = length(sampled_vectors)
    (m1 != m1) && throw(
        ErrorException(
            """
            The temporary storage of points ($(m1)) and vectors ($(m2)) does not agree in length.
            It should be automatically generated with length $(sample_size). Make sure you pass the correct amounts of temporary storages.
            """
        )
    )
    return GradientSamplingState(;
        p = p, X = X, sampled_points = sampled_points, sampled_vectors = sampled_vectors,
        sampling_radius = sampling_radius, sampling_radius_reduction = sampling_radius_reduction, sampling_radius_threshold = sampling_radius_threshold,
        subgradient_norm_tolerance = subgradient_norm_tolerance, subgradient_norm_threshold = subgradient_norm_threshold, subgradient_norm_reduction = subgradient_norm_reduction,
        sub_problem = sub_problem, sub_state = sub_state,
        stepsize = stepsize, stopping_criterion = stopping_criterion, retraction_method = retraction_method,
        vector_transport_method = vector_transport_method,
    )
end

function default_stepsize(
        M::AbstractManifold, ::Type{GradientSamplingState}; retraction_method = default_retraction_method(M),
    )
    return ArmijoLinesearchStepsize(M; retraction_method = retraction_method, initial_stepsize = 1.0)
end
#
#
# Accessors
get_iterate(gss::GradientSamplingState) = gss.p
get_solver_result(gss::GradientSamplingState) = gss.p
get_gradient(gss::GradientSamplingState) = gss.X

initialize_solver!(::AbstractManoptProblem, gss::GradientSamplingState) = gss

function Base.show(io::IO, gss::GradientSamplingState)
    print(io, "GradientSamplingState(;")
    error("TODO")
    return print(io, ")")
end

function status_summary(gss::GradientSamplingState; context::Symbol = :default)
    (context === :short) && return repr(gss)
    i = get_count(gss, :Iterations)
    conv_inl = (i > 0) ? (indicates_convergence(gss.stop) ? " (converged" : " (stopped") * " after $i iterations)" : ""
    (context === :inline) && return "A solver state for the gradient sampling solver$(conv_inl)"
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(gss.stop) ? "Yes" : "No"
    error("TODO")
    s = """
    # Solver state for `Manopt.jl`s Gradient Sampling Algorithm
    $Iter
    ## Parameters
    * retraction method: $(gss.retraction_method)

    ## Stepsize
    $(_in_str(status_summary(gss.stepsize; context = context); indent = 0, headers = 1))

    ## Stopping criterion
    $(_in_str(status_summary(gss.stop; context = context); indent = 0, headers = 1))
    This indicates convergence: $Conv"""
    return s
end

# TODO:
_doc_gradient_sampling = """
    gradient_sampling(M, f, grad_f, p=rand(M); kwargs...)
    gradient_sampling(M, gradient_objective, p=rand(M); kwargs...)
    gradient_sampling!(M, f, grad_f, p; kwargs...)
    gradient_sampling!(M, gradient_objective, p; kwargs...)

perform the gradient sampling algorithm as introduced in [HosseiniUschmajew:2017](@cite).

# Input

$(_args([:M, :f, :grad_f, :p]))

$(_note(:GradientObjective))

# Keyword arguments

$(_kwargs(:differential))
$(_kwargs(:evaluation; add_properties = [:GradientExample]))
$(_kwargs(:retraction_method))
$(_kwargs(:stepsize; default = "`[`default_stepsize`](@ref)`(M, `[`GradientSamplingState`](@ref)`; retraction_method=retraction_method)"))
$(_kwargs(:stopping_criterion; default = "`[`StopAfterIteration`](@ref)`(200)`$(_sc(:Any))[`StopWhenGradientNormLess`](@ref)`(1e-8)"))
$(_kwargs(:X; add_properties = [:as_Gradient]))
"""

@doc "$(_doc_gradient_sampling)"
gradient_sampling(M::AbstractManifold, args...; kwargs...)

function gradient_sampling(
        M::AbstractManifold, f, grad_f, p = rand(M);
        differential = nothing,
        evaluation::AbstractEvaluationType = AllocatingEvaluation(),
        kwargs...,
    )
    p_ = _ensure_mutating_variable(p)
    f_ = _ensure_mutating_cost(f, p)
    grad_f_ = _ensure_mutating_gradient(grad_f, p, evaluation)
    mgo = ManifoldGradientObjective(
        f_, grad_f_; evaluation = evaluation, differential = differential
    )
    rs = gradient_sampling(M, mgo, p_; kwargs...)
    return _ensure_matching_output(p, rs)
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
        differential = nothing, evaluation::AbstractEvaluationType = AllocatingEvaluation(),
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
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, typeof(p)),
        stepsize::Union{Stepsize, ManifoldDefaultsFactory} = default_stepsize(
            M, GradientSamplingState; retraction_method = retraction_method
        ),
        X = zero_vector(M, p),
        # TODO: all other kwargs from the state here
        kwargs..., #collect rest
    ) where {O <: Union{AbstractManifoldFirstOrderObjective, AbstractDecoratedManifoldObjective}}
    # all explicit others others from above are anyways accepted here, so we only have to pass kwargs in
    keywords_accepted(gradient_sampling!; kwargs...)
    dmgo = decorate_objective!(M, mgo; kwargs...)
    dmp = DefaultManoptProblem(M, dmgo)
    s = GradientSamplingState(
        M;
        p = p,
        stepsize = _produce_type(stepsize, M, p),
        retraction_method = retraction_method,
        X = X,
    )
    ds = decorate_state!(s; debug = debug, kwargs...)
    solve!(dmp, ds)
    return get_solver_return(get_objective(dmp), ds)
end
calls_with_kwargs(::typeof(gradient_sampling!)) = (decorate_objective!, decorate_state!)

#
#
# Solver implementation

function step_solver!(
        mp::AbstractManoptProblem, gss::GradientSamplingState, i
    )
    M = get_manifold(mp)
    # resample on TpM, map to manifold and make sure they are within radius
    for (ps, Xs) in zip(gss.sampled_points, gss.sampled_vectors)
        rand!(M, Xs; vector_at = gss.p, σ = gss.sampling_radius / 2)
        retract!(M, ps, Xs, gss.retraction_method)
        while distance(M, gss.p, ps) > gss.sampling_radius
            Xs ./= 2
            retract!(M, ps, Xs, gss.retraction_method)
        end
    end
    # re-use the tangent vector memory to evaluate the gradients
    # and transport them to the current iterate
    for (ps, Xs) in zip(gss.sampled_points, gss.sampled_vectors)
        get_gradient!(mp, Xs, ps)
        vector_transport_to!(M, Xs, ps, Xs, gss.p)
    end
    # TODO: solve sub problem in some Y
    Y = zero_vector(M, p)
    # Decide whether to accept the step or update radius
    if norm(M, gss.p, Y) < gss.subgradient_norm_threshold
        gss.sampling_radius *= gss.sampling_radius_reduction
        gss.subgradient_norm_tolerance *= gss.subgradient_norm_reduction
    else
        copyto!(M, gss.X, gss.p, -Y)
        step = get_stepsize(mp, gss, i)
        ManifoldsBase.retract_fused!(M, gss.p, gss.p, gss.X, step, gss.retraction_method)
    end
    # TODO
    # the remaining rtol & atol from Ole Gunnars step! are part of the sub_solver?
    # check RipQP as a solver instead of JuMP, maybe also the other new QP?
    return gss
end
