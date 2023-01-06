#
# State
#
@doc raw"""
    ParticleSwarmState{P,T} <: AbstractManoptSolverState

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

    ParticleSwarmState(M, x0, velocity; kawrgs...)

construct a particle swarm Option for the manifold `M` starting at initial population `x0` with velocities `x0`,
where the manifold is used within the defaults of the other fields mentioned above,
which are keyword arguments here.

# See also

[`particle_swarm`](@ref)
"""
mutable struct ParticleSwarmState{
    TX<:AbstractVector,
    TG,
    TVelocity<:AbstractVector,
    TParams<:Real,
    TStopping<:StoppingCriterion,
    TRetraction<:AbstractRetractionMethod,
    TInvRetraction<:AbstractInverseRetractionMethod,
    TVTM<:AbstractVectorTransportMethod,
} <: AbstractManoptSolverState
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

    function ParticleSwarmState(
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
#
# Accessors
#
get_iterate(O::ParticleSwarmState) = O.x
function set_iterate!(O::ParticleSwarmState, p)
    O.x = p
    return O
end
#
# Constructors
#
@doc raw"""
    patricle_swarm(M, F)

perform the particle swarm optimization algorithm (PSO), starting with the initial particle positions ``x_0``[^Borckmans2010].
The aim of PSO is to find the particle position ``g`` on the `Manifold M` that solves
```math
\min_{x ∈\mathcal{M}} F(x).
```
To this end, a swarm of particles is moved around the `Manifold M` in the following manner.
For every particle ``k`` we compute the new particle velocities ``v_k^{(i)}`` in every step ``i`` of the algorithm by

```math
v_k^{(i)} = ω \, \operatorname{T}_{x_k^{(i)}\gets x_k^{(i-1)}}v_k^{(i-1)} + c \,  r_1  \operatorname{retr}_{x_k^{(i)}}^{-1}(p_k^{(i)}) + s \,  r_2 \operatorname{retr}_{x_k^{(i)}}^{-1}(g),
```

where ``x_k^{(i)}`` is the current particle position, ``ω`` denotes the inertia,
``c`` and ``s`` are a cognitive and a social weight, respectively,
``r_j``, ``j=1,2`` are random factors which are computed new for each particle and step,
``\operatorname{retr}^{-1}`` denotes an inverse retraction on the `Manifold` `M`, and
``\operatorname{T}`` is a vector transport.

Then the position of the particle is updated as

```math
x_k^{(i+1)} = \operatorname{retr}_{x_k^{(i)}}(v_k^{(i)}),
```

where ``\operatorname{retr}`` denotes a retraction on the `Manifold` `M`. At the end of each step for every particle, we set

```math
p_k^{(i+1)} = \begin{cases}
x_k^{(i+1)},  & \text{if } F(x_k^{(i+1)})<F(p_{k}^{(i)}),\\
p_{k}^{(i)}, & \text{else,}
\end{cases}

```
and
```math
g_k^{(i+1)} =\begin{cases}
p_k^{(i+1)},  & \text{if } F(p_k^{(i+1)})<F(g_{k}^{(i)}),\\
g_{k}^{(i)}, & \text{else,}
\end{cases}
```
i.e. ``p_k^{(i)}`` is the best known position for the particle ``k`` and ``g^{(i)}`` is the global best known position ever visited up to step ``i``.

[^Borckmans2010]:
    > P. B. Borckmans, M. Ishteva, P.-A. Absil, __A Modified Particle Swarm Optimization Algorithm for the Best Low Multilinear Rank Approximation of Higher-Order Tensors__,
    > In: Dorigo M. et al. (eds) Swarm Intelligence. ANTS 2010. Lecture Notes in Computer Science, vol 6234. Springer, Berlin, Heidelberg,
    > doi [10.1007/978-3-642-15461-4_2](https://doi.org/10.1007/978-3-642-15461-4_2)

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize

# Optional
* `n` - (`100`) number of random initial positions of x0
* `x0` – the initial positions of each particle in the swarm ``x_k^{(0)} ∈ \mathcal M`` for ``k = 1, \dots, n``, per default these are n random points on `M`
* `velocity` – a set of tangent vectors (of type `AbstractVector{T}`) representing the velocities of the particles, per default a random tangent vector per inital position
* `inertia` – (`0.65`) the inertia of the patricles
* `social_weight` – (`1.4`) a social weight factor
* `cognitive_weight` – (`1.4`) a cognitive weight factor
* `retraction_method` – (`default_retraction_method(M)`) a `retraction(M,x,ξ)` to use.
* `inverse_retraction_method` - (`default_inverse_retraction_method(M)`) an `inverse_retraction(M,x,y)` to use.
* `vector_transport_mthod` - (`default_vector_transport_method(M)`) a vector transport method to use.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(500)`, [`StopWhenChangeLess`](@ref)`(10^{-4})))`
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.

...
and the ones that are passed to [`decorate_state!`](@ref) for decorators.

# Output

the obtained (approximate) minimizer ``g``, see [`get_solver_return`](@ref) for details
"""
function particle_swarm(
    M::AbstractManifold,
    F::TF;
    n::Int=100,
    x0::AbstractVector=[rand(M) for i in 1:n],
    kwargs...,
) where {TF}
    x_res = copy.(Ref(M), x0)
    return particle_swarm!(M, F; n=n, x0=x_res, kwargs...)
end

@doc raw"""
    patricle_swarm!(M, F; n=100, x0::AbstractVector=[rand(M) for i in 1:n], kwargs...)

perform the particle swarm optimization algorithm (PSO), starting with the initial particle positions ``x_0``[^Borckmans2010]
in place of `x0`.

# Input
* `M` – a manifold ``\mathcal M``
* `F` – a cost function ``F:\mathcal M→ℝ`` to minimize

# Optional
* `n` - (`100`) number of random initial positions of x0
* `x0` – the initial positions of each particle in the swarm ``x_k^{(0)} ∈ \mathcal M`` for ``k = 1, \dots, n``, per default these are n random points on `M`s

for more optional arguments, see [`particle_swarm`](@ref).
"""
function particle_swarm!(
    M::AbstractManifold,
    f::TF;
    n::Int=100,
    x0::AbstractVector=[rand(M) for i in 1:n],
    velocity::AbstractVector=[rand(M; vector_at=y) for y in x0],
    inertia::Real=0.65,
    social_weight::Real=1.4,
    cognitive_weight::Real=1.4,
    stopping_criterion::StoppingCriterion=StopAfterIteration(500) |
                                          StopWhenChangeLess(1e-4),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M),
    inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M
    ),
    vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
        M
    ),
    kwargs..., #collect rest
) where {TF}
    dmco = decorate_objective!(M, ManifoldCostObjective(f); kwargs...)
    mp = DefaultManoptProblem(M, dmco)
    o = ParticleSwarmState(
        M,
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
    o = decorate_state!(o; kwargs...)
    return get_solver_return(solve!(mp, o))
end

#
# Solver functions
#
function initialize_solver!(mp::AbstractManoptProblem, s::ParticleSwarmState)
    j = argmin([get_cost(mp, p) for p in s.x])
    return s.g = deepcopy(s.x[j])
end
function step_solver!(mp::AbstractManoptProblem, s::ParticleSwarmState, ::Any)
    M = get_manifold(mp)
    for i in 1:length(s.x)
        s.velocity[i] =
            s.inertia .* s.velocity[i] +
            s.cognitive_weight * rand(1) .*
            inverse_retract(M, s.x[i], s.p[i], s.inverse_retraction_method) +
            s.social_weight * rand(1) .*
            inverse_retract(M, s.x[i], s.g, s.inverse_retraction_method)
        xOld = s.x[i]
        s.x[i] = retract(M, s.x[i], s.velocity[i], s.retraction_method)
        s.velocity[i] = vector_transport_to(
            M, xOld, s.velocity[i], s.x[i], s.vector_transport_method
        )
        if get_cost(mp, s.x[i]) < get_cost(mp, s.p[i])
            s.p[i] = s.x[i]
            if get_cost(mp, s.p[i]) < get_cost(mp, s.g)
                s.g = s.p[i]
            end
        end
    end
end
get_solver_result(s::ParticleSwarmState) = s.g
#
# Change not only refers to different iterates (x=the whole population)
# but also lives in the power manifold on M, so we have to adapt StopWhenChangeless
#
function (c::StopWhenChangeLess)(mp::AbstractManoptProblem, s::ParticleSwarmState, i)
    if has_storage(c.storage, :Iterate)
        x_old = get_storage(c.storage, :Iterate)
        n = length(s.x)
        d = distance(
            PowerManifold(get_manifold(mp), NestedPowerRepresentation(), n), s.x, x_old
        )
        if d < c.threshold && i > 0
            c.reason = "The algorithm performed a step with a change ($d in the population) less than $(c.threshold).\n"
            c.storage(mp, s, i)
            return true
        end
    end
    c.storage(mp, s, i)
    return false
end
