#
# Options
#
@doc raw"""
    ParticleSwarmOptions{P,T} <: Options

Describes a particle swarm optimizing algorithm, with

# Fields

* `x` – a set of points (of type `AbstractVector{P}`) on a manifold as initial particle positions
* `velocity` – a set of tangent vectors (of type `AbstractVector{T}`) representing the velocities of the particles
* `inertia` – (`0.65`) the inertia of the patricles
* `social_weight` – (`1.4`) a social weight factor
* `cognitive_weight` – (`1.4`) a cognitive weight factor
* `stopping_criterion` – (`[`StopAfterIteration`](@ref)`(500) | `[`StopWhenChangeLess`](@ref)`(1e-4)`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `retraction_method` – (`default_retraction_method(M)`) the rectraction to use
* `inverse_retraction_method` - (`default_inverse_retraction_method(M)`) an inverse retraction to use.
* `vector_transport_method` - (`default_vector_transport_method(M)`) a vector transport to use

# Constructor

    ParticleSwarmOptions(M, x0, velocity; kawrgs...)

construct a particle swarm Option for the manifold `M` starting at initial population `x0` with velocities `x0`,
where the manifold is used within the defaults of the other fields mentioned above,
which are keyword arguments here.

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
    @deprecate ParticleSwarmOptions(
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
    ) ParticleSwarmOptions(
        DefaultManifold(2),
        x0,
        velocity;
        inertia=inertia,
        social_weight=social_weight,
        cognitive_weight=cognitive_weight,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        inverse_retraction_method=inverse_retraction_method,
        vector_transport_method=vector_transport_method,
    )
    function ParticleSwarmOptions(
        M::AbstractManifold,
        x0::AbstractVector,
        velocity::AbstractVector;
        inertia=0.65,
        social_weight=1.4,
        cognitive_weight=1.4,
        stopping_criterion::StoppingCriterion=StopAfterIteration(500) |
                                              StopWhenChangeLess(1e-4),
        retraction_method::AbstractRetractionMethod=default_retraction_method(M),
        inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
            M
        ),
        vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
            M
        ),
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
get_iterate(O::ParticleSwarmOptions) = O.x
