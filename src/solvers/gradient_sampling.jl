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
        vector_transport:method::VTM = default_vector_transport_method(M, typeof(p))
    ) where {P, T, R <: Real, SC <: StoppingCriterion, S <: Stepsize, RTM <: AbstractRetractionMethod}
    return GradientSamplingState(;
        p = p, X = X, sampling_radius = sampling_radius,
        subgradient_norm_tolerance = subgradient_norm_tolerance, optimal_subgradient_norm = optimal_subgradient_norm,
        optimal_sampling_radius = optimal_sampling_radius, stepsize = stepsize, stopping_criterion = stopping_criterion, retraction_method = retraction_method,
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

function step_solver!(
        mp::AbstractManoptProblem, gss::GradientSamplingState, i
    )
    # TODO
    # Take the kwargs of this function in the notebook and turn them into
    # parameters in the state - see above.
    # the remaining rtol & atol are part of the subsolver

    #
    # use subproblem / substate for the subsolver
    # check RipQP as a solver instead of JuMP, maybe also the other new QP?

    return gss
end
