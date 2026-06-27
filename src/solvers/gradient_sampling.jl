# TODO Document all fields and refer to their names in the paper as well
#
"""
    GradientSamplingState

A state for the [gradient sampling algorithm](@ref gradient_sampling).

# Fields
$(_fields(:p; add_properties = [:as_Iterate]))
$(_fields(:X; add_properties = [:as_Gradient]))
$(_fields(:stopping_criterion; name = "stop"))
$(_fields([:stepsize, :retraction_method, :vector_transport_method]))

# Constructor

"""
mutable struct GradientSamplingState{
        P, T, R <: Real, SC <: StoppingCriterion, S <: Stepsize, RTM <: AbstractRetractionMethod,
        VTM <: AbstractVectorTransportMethod,
    }
    p::P # (In paper: xₗ)
    X::T # (In paper: wₗ)
    sampling_radius::R # (In paper: εₗ)
    subgradient_norm_tolerance::R # (In paper: δₗ)
    # maybe call these two tolerances: ?
    optimal_subgradient_norm::R # (In paper: δₒₚₜ)
    optimal_sampling_radius::R # (In paper: εₒₚₜ)
    # Missing
    # number of samples – m in the paper
    # how to shrink sampling radius –  θ_ε in the paper
    # how to shrink subgradient norm tolerancen θ_δ in the paper
    # temporary storage for m points and m tangent vectors (first to rand to generate pi then to evaluate gradients)
    # subproblem, substate
    stepsize::S
    stop::SC
    retraction_method::RTM
    vector_transport_method::VTM
    function GradientSamplingState(;
            p::P, X::T, norm_X::R,
            sampling_radius::R, subgradient_norm_tolerance::R, optimal_subgradient_norm::R,
            optimal_sampling_radius::R,
            stepsize::S, stopping_criterion::SC, retraction_method::RTM, vector_transport_method::VTM
        ) where {P, T, R <: Real, SC <: StoppingCriterion, S <: Stepsize, RTM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod}
        return new{P, T, R, SC, S, RTM, VTM}(
            p, X, norm_X, sampling_radius, subgradient_norm_tolerance, optimal_subgradient_norm,
            optimal_sampling_radius, stepsize, stopping_criterion, retraction_method, vector_transport_method
        )
    end
end

function GradientSamplingState(
        M::AbstractManifold;
        p::P = rand(M),
        X::T = zero_vector(M, p),
        sampling_radius::R = 0.5,
        subgradient_norm_tolerance::R = 0.1,
        optimal_subgradient_norm::R = 1.0e-3,
        optimal_sampling_radius::R = 1.0e-2,
        retraction_method::RTM = default_retraction_method(M, typeof(p)),
        stopping_criterion::SC = StopAfterIteration(200) | (
            StopWhenGradientNormLess(optimal_subgradient_norm) & (
                StopWhenSmallerOrEqual(:sampling_radius, optimal_sampling_radius)
            )
        ),
        stepsize::S = default_stepsize(
            M, GradientSamplingState; retraction_method = retraction_method
        ),
        vector_transport_method::VTM = default_vector_transport_method(M, typeof(p))
    ) where {P, T, R <: Real, SC <: StoppingCriterion, S <: Stepsize, RTM <: AbstractRetractionMethod, VTM <: AbstractVectorTransportMethod}
    return GradientSamplingState(;
        p = p, X = X, sampling_radius = sampling_radius,
        subgradient_norm_tolerance = subgradient_norm_tolerance, optimal_subgradient_norm = optimal_subgradient_norm,
        optimal_sampling_radius = optimal_sampling_radius, stepsize = stepsize, stopping_criterion = stopping_criterion, retraction_method = retraction_method,
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

perform the gradient sampling algorithm as introduced in [HosseiniUschmajew:2017](@cite)
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
    # TODO
    # Take the kwargs of this function in the notebook and turn them into
    # parameters in the state - see above.
    # the remaining rtol & atol are part of the subsolver

    #
    # use subproblem / substate for the subsolver
    # check RipQP as a solver instead of JuMP, maybe also the other new QP?

    return gss
end
