#
# Options
#
@doc raw"""
    ParticleSwarmOptions{P,T} <: Options

Describes a particle swarm optimizing algorithm, with

# Fields
a default value is given in brackets if a parameter can be left out in initialization.

* `x0` – a set of points (of type `AbstractVector{P}`) on a manifold as initial particle positions
* `velocity` – a set of tangent vectors (of type `AbstractVector{T}`) representing the velocities of the particles
* `inertia` – (`0.65`) the inertia of the patricles
* `social_weight` – (`1.4`) a social weight factor
* `cognitive_weight` – (`1.4`) a cognitive weight factor
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(500)`, [`StopWhenChangeLess`](@ref)`(10^{-4})))`
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `retraction_method` – (`ExponentialRetraction`) the rectraction to use
* `inverse_retraction_method` - (`LogarithmicInverseRetraction`) an inverse retraction to use.

# Constructor

    ParticleSwarmOptions(x0, velocity, inertia, social_weight, cognitive_weight, stopping_criterion[, retraction_method=ExponentialRetraction(), inverse_retraction_method=LogarithmicInverseRetraction()])

construct a particle swarm Option with the fields and defaults as above.

# See also
[`particle_swarm`](@ref)
"""
mutable struct ParticleSwarmOptions{
    TX<:AbstractVector,
    TG,
    TVelocity<:AbstractVector,
    TParams<:Real,
    TStopping<:StoppingCriterion,
    TRetraction<:AbstractRetractionMethod,
    TInvRetraction<:AbstractInverseRetractionMethod,
    TVTM<:AbstractVectorTransportMethod,
} <: Options
    x::TX
    p::TX
    g::TG
    velocity::TVelocity
    inertia::TParams
    social_weight::TParams
    cognitive_weight::TParams
    stop::TStopping
    retraction_method::TRetraction
    inverse_retraction_method::TInvRetraction
    vector_transport_method::TVTM
    function ParticleSwarmOptions(
        x0::AbstractVector,
        velocity::AbstractVector,
        inertia=0.65,
        social_weight=1.4,
        cognitive_weight=1.4,
        stopping_criterion::StoppingCriterion=StopWhenAny(
            StopAfterIteration(500), StopWhenChangeLess(10.0^(-4))
        ),
        retraction_method::AbstractRetractionMethod=ExponentialRetraction(),
        inverse_retraction_method::AbstractInverseRetractionMethod=LogarithmicInverseRetraction(),
        vector_transport_method::AbstractVectorTransportMethod=ParallelTransport(),
    )
        o = new{
            typeof(x0),
            eltype(x0),
            typeof(velocity),
            typeof(inertia + social_weight + cognitive_weight),
            typeof(stopping_criterion),
            typeof(retraction_method),
            typeof(inverse_retraction_method),
            typeof(vector_transport_method),
        }()
        o.x = x0
        o.p = deepcopy(x0)
        o.velocity = velocity
        o.inertia = inertia
        o.social_weight = social_weight
        o.cognitive_weight = cognitive_weight
        o.stop = stopping_criterion
        o.retraction_method = retraction_method
        o.inverse_retraction_method = inverse_retraction_method
        o.vector_transport_method = vector_transport_method
        return o
    end
end
