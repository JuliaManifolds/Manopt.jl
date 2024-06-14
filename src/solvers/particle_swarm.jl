#
# State
#
@doc raw"""
    ParticleSwarmState{P,T} <: AbstractManoptSolverState

Describes a particle swarm optimizing algorithm, with

# Fields

* `cognitive_weight`:          (`1.4`) a cognitive weight factor
* `inertia`:                   (`0.65`) the inertia of the particles
* `inverse_retraction_method`: (`default_inverse_retraction_method(M, eltype(swarm))`) an inverse retraction to use.
* `retraction_method`:         (`default_retraction_method(M, eltype(swarm))`) the retraction to use
* `social_weight`:             (`1.4`) a social weight factor
* `stopping_criterion`:        (`[`StopAfterIteration`](@ref)`(500) | `[`StopWhenChangeLess`](@ref)`(1e-4)`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `vector_transport_method`:  (`default_vector_transport_method(M, eltype(swarm))`) a vector transport to use
* `velocity`:                 a set of tangent vectors (of type `AbstractVector{T}`) representing the velocities of the particles

# Internal and temporary fields

* `cognitive_vector`: temporary storage for a tangent vector related to `cognitive_weight`
* `p`:                storage for the best point ``p`` visited by all particles.
* `positional_best`:  storing the best position ``p_i`` every single swarm participant visited
* `q`:                temporary storage for a point to avoid allocations during a step of the algorithm
* `social_vec`:       temporary storage for a tangent vector related to `social_weight`
* `swarm`:            a set of points (of type `AbstractVector{P}`) on a manifold ``\{s_i\}_{i=1}^N``

# Constructor

    ParticleSwarmState(M, initial_swarm, velocity; kawrgs...)

construct a particle swarm solver state for the manifold `M` starting at initial population `x0` with `velocities`,
where the manifold is used within the defaults specified previously. All fields with defaults are keyword arguments here.

# See also

[`particle_swarm`](@ref)
"""
mutable struct ParticleSwarmState{
    P,
    T,
    TX<:AbstractVector{P},
    TVelocity<:AbstractVector{T},
    TParams<:Real,
    TStopping<:StoppingCriterion,
    TRetraction<:AbstractRetractionMethod,
    TInvRetraction<:AbstractInverseRetractionMethod,
    TVTM<:AbstractVectorTransportMethod,
} <: AbstractManoptSolverState
    swarm::TX
    positional_best::TX
    p::P
    velocity::TVelocity
    inertia::TParams
    social_weight::TParams
    cognitive_weight::TParams
    q::P
    social_vector::T
    cognitive_vector::T
    stop::TStopping
    retraction_method::TRetraction
    inverse_retraction_method::TInvRetraction
    vector_transport_method::TVTM

    function ParticleSwarmState(
        M::AbstractManifold,
        swarm::VP,
        velocity::VT;
        inertia=0.65,
        social_weight=1.4,
        cognitive_weight=1.4,
        stopping_criterion::SCT=StopAfterIteration(500) | StopWhenChangeLess(1e-4),
        retraction_method::RTM=default_retraction_method(M, eltype(swarm)),
        inverse_retraction_method::IRM=default_inverse_retraction_method(M, eltype(swarm)),
        vector_transport_method::VTM=default_vector_transport_method(M, eltype(swarm)),
    ) where {
        P,
        T,
        VP<:AbstractVector{<:P},
        VT<:AbstractVector{<:T},
        RTM<:AbstractRetractionMethod,
        SCT<:StoppingCriterion,
        IRM<:AbstractInverseRetractionMethod,
        VTM<:AbstractVectorTransportMethod,
    }
        s = new{
            P,T,VP,VT,typeof(inertia + social_weight + cognitive_weight),SCT,RTM,IRM,VTM
        }()
        s.swarm = swarm
        s.positional_best = copy.(Ref(M), swarm)
        s.q = copy(M, first(swarm))
        s.p = copy(M, first(swarm))
        s.social_vector = zero_vector(M, s.q)
        s.cognitive_vector = zero_vector(M, s.q)
        s.velocity = velocity
        s.inertia = inertia
        s.social_weight = social_weight
        s.cognitive_weight = cognitive_weight
        s.stop = stopping_criterion
        s.retraction_method = retraction_method
        s.inverse_retraction_method = inverse_retraction_method
        s.vector_transport_method = vector_transport_method
        return s
    end
end
function show(io::IO, pss::ParticleSwarmState)
    i = get_count(pss, :Iterations)
    Iter = (i > 0) ? "After $i iterations\n" : ""
    Conv = indicates_convergence(pss.stop) ? "Yes" : "No"
    s = """
    # Solver state for `Manopt.jl`s Particle Swarm Optimization Algorithm
    $Iter
    ## Parameters
    * inertia:          $(pss.inertia)
    * social_weight:    $(pss.social_weight)
    * cognitive_weight: $(pss.cognitive_weight)
    * inverse retraction method: $(pss.inverse_retraction_method)
    * retraction method:         $(pss.retraction_method)
    * vector transport method:   $(pss.vector_transport_method)

    ## Stopping criterion

    $(status_summary(pss.stop))
    This indicates convergence: $Conv"""
    return print(io, s)
end
#
# Access functions
#
get_iterate(pss::ParticleSwarmState) = pss.p
function set_iterate!(pss::ParticleSwarmState, p)
    pss.p = p
    return pss
end
function set_manopt_parameter!(pss::ParticleSwarmState, ::Val{:Population}, swarm)
    return pss.swarm = swarm
end
function get_manopt_parameter(pss::ParticleSwarmState, ::Val{:Population})
    return pss.swarm
end

#
# Constructors
#
_doc_swarm = raw"``S = \{s_1, \ldots, s_n\}``"
_doc_velocities = raw"""
```math
  X_k^{(i)} = ω \, \operatorname{T}_{s_k^{(i)}\gets s_k^{(i-1)}}X_k^{(i-1)} + c r_1  \operatorname{retr}_{s_k^{(i)}}^{-1}(p_k^{(i)}) + s r_2 \operatorname{retr}_{s_k^{(i)}}^{-1}(p),
```
"""
_doc_particle_update = raw"""
```math
s_k^{(i+1)} = \operatorname{retr}_{s_k^{(i)}}(X_k^{(i)}),
```
"""
_doc_swarm_best = raw"""
```math
p_k^{(i+1)} = \begin{cases}
s_k^{(i+1)},  & \text{if } F(s_k^{(i+1)})<F(p_{k}^{(i)}),\\
p_{k}^{(i)}, & \text{else,}
\end{cases}
```
"""
_doc_swarm_global_best = raw"""
```math
g^{(i+1)} = \begin{cases}
p_k^{(i+1)},  & \text{if } F(p_k^{(i+1)})<F(g_{k}^{(i)}),\\
g_{k}^{(i)}, & \text{else,}
\end{cases}
```
"""

@doc """
    patricle_swarm(M, f; kwargs...)
    patricle_swarm(M, f, swarm; kwargs...)
    patricle_swarm(M, mco::AbstractManifoldCostObjective; kwargs..)
    patricle_swarm(M, mco::AbstractManifoldCostObjective, swarm; kwargs..)

perform the particle swarm optimization algorithm (PSO) to solve
$_problem_default
PSO starts with an initial `swarm` [BorckmansIshtevaAbsil:2010](@cite) of points
on the manifold.
If no `swarm` is provided, `swarm_size` many random points are used.

To this end, a swarm $_doc_swarm of particles is moved around the manifold `M` in the following manner.
For every particle ``s_k^{(i)}`` the new particle velocities ``X_k^{(i)}`` are computed in every step ``i`` of the algorithm by
$_doc_velocities

where
* ``s_k^{(i)}`` is the current particle position,
* ``ω`` denotes the inertia,
* ``c`` and ``s`` are a cognitive and a social weight, respectively,
* ``r_j``, ``j=1,2`` are random factors which are computed new for each particle and step
* we further have $_math_VT and $_math_inv_retr

Then the position of the particle is updated as

$_doc_particle_update

where we use $_math_retr on the `Manifold` `M`.
Then the single particles best entries ``p_k^{(i)}`` are updated as

$_doc_swarm_best

and the global best position

$_doc_swarm_global_best

# Input

$_arg_M
$_arg_f
* `swarm = [rand(M) for _ in 1:swarm_size]`: an initial swarm of points.

Instead of a cost function `f` you can also provide an [`AbstractManifoldCostObjective`](@ref) `mco`.

# Optional

* `cognitive_weight`:          (`1.4`) a cognitive weight factor
* `inertia`:                   (`0.65`) the inertia of the particles
* `inverse_retraction_method`: (`default_inverse_retraction_method(M, eltype(swarm))`) an inverse retraction to use.
* `swarm_size`:                (`100`) swarm size, if it should be generated randomly
* `retraction_method`:         (`default_retraction_method(M, eltype(swarm))`) a retraction to use.
* `social_weight`:             (`1.4`) a social weight factor
* `stopping_criterion`:        ([`StopAfterIteration`](@ref)`(500) | `[`StopWhenChangeLess`](@ref)`(1e-4)`)
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `vector_transport_method`:   (`default_vector_transport_method(M, eltype(swarm))`) a vector transport method to use.
* `velocity`:                  a set of tangent vectors (of type `AbstractVector{T}`) representing the velocities of the particles, per default a random tangent vector per initial position

All other keyword arguments are passed to [`decorate_state!`](@ref) for decorators or
[`decorate_objective!`](@ref), respectively.
If you provide the [`ManifoldGradientObjective`](@ref) directly, these decorations can still be specified

# Output

the obtained (approximate) minimizer ``g``, see [`get_solver_return`](@ref) for details
"""
function particle_swarm(
    M::AbstractManifold,
    f;
    n=nothing,
    swarm_size=isnothing(n) ? 100 : n,
    x0=nothing,
    kwargs...,
)
    !isnothing(n) && (@warn "The keyword `n` is deprecated, use `swarm_size` instead")
    !isnothing(x0) &&
        (@warn "The keyword `x0` is deprecated, use `particle_swarm(M, f, x0)` instead")
    return particle_swarm(
        M, f, isnothing(x0) ? [rand(M) for _ in 1:swarm_size] : x0; kwargs...
    )
end
function particle_swarm(M::AbstractManifold, f, swarm::AbstractVector; kwargs...)
    mco = ManifoldCostObjective(f)
    return particle_swarm(M, mco, swarm; kwargs...)
end
function particle_swarm(
    M::AbstractManifold,
    f,
    swarm::AbstractVector{T};
    velocity::AbstractVector=[rand(M; vector_at=y) for y in swarm],
    kwargs...,
) where {T<:Number}
    f_(M, p) = f(M, p[])
    swarm_ = [[s] for s in swarm]
    velocity_ = [[v] for v in velocity]
    rs = particle_swarm(M, f_, swarm_; velocity=velocity_, kwargs...)
    #return just a number if  the return type is the same as the type of q
    return (typeof(swarm_[1]) == typeof(rs)) ? rs[] : rs
end

function particle_swarm(
    M::AbstractManifold, mco::O, swarm::AbstractVector; kwargs...
) where {O<:Union{AbstractManifoldCostObjective,AbstractDecoratedManifoldObjective}}
    new_swarm = [copy(M, xi) for xi in swarm]
    return particle_swarm!(M, mco, new_swarm; kwargs...)
end

@doc raw"""
    particle_swarm!(M, f, swarm; kwargs...)
    particle_swarm!(M, mco::AbstractManifoldCostObjective, swarm; kwargs..)

perform the particle swarm optimization algorithm (PSO), starting with the initial `swarm` which is then modified in place.

# Input

* `M`:     a manifold ``\mathcal M``
* `f`:     a cost function ``f:\mathcal M→ℝ`` to minimize
* `swarm`: (`[rand(M) for _ in 1:swarm_size]`) an initial swarm of points.

Instead of a cost function `f` you can also provide an [`AbstractManifoldCostObjective`](@ref) `mco`.

For more details and optional arguments, see [`particle_swarm`](@ref).
"""
function particle_swarm!(M::AbstractManifold, f, swarm::AbstractVector; kwargs...)
    mco = ManifoldCostObjective(f)
    return particle_swarm!(M, mco, swarm; kwargs...)
end
function particle_swarm!(
    M::AbstractManifold,
    mco::O,
    swarm::AbstractVector;
    velocity::AbstractVector=[rand(M; vector_at=y) for y in swarm],
    inertia::Real=0.65,
    social_weight::Real=1.4,
    cognitive_weight::Real=1.4,
    stopping_criterion::StoppingCriterion=StopAfterIteration(500) |
                                          StopWhenSwarmVelocityLess(1e-4),
    retraction_method::AbstractRetractionMethod=default_retraction_method(M, eltype(swarm)),
    inverse_retraction_method::AbstractInverseRetractionMethod=default_inverse_retraction_method(
        M, eltype(swarm)
    ),
    vector_transport_method::AbstractVectorTransportMethod=default_vector_transport_method(
        M, eltype(swarm)
    ),
    kwargs..., #collect rest
) where {O<:Union{AbstractManifoldCostObjective,AbstractDecoratedManifoldObjective}}
    dmco = decorate_objective!(M, mco; kwargs...)
    mp = DefaultManoptProblem(M, dmco)
    pss = ParticleSwarmState(
        M,
        swarm,
        velocity;
        inertia=inertia,
        social_weight=social_weight,
        cognitive_weight=cognitive_weight,
        stopping_criterion=stopping_criterion,
        retraction_method=retraction_method,
        inverse_retraction_method=inverse_retraction_method,
        vector_transport_method=vector_transport_method,
    )
    dpss = decorate_state!(pss; kwargs...)
    solve!(mp, dpss)
    return get_solver_return(get_objective(mp), dpss)
end

#
# Solver functions
#
function initialize_solver!(mp::AbstractManoptProblem, s::ParticleSwarmState)
    M = get_manifold(mp)
    j = argmin([get_cost(mp, p) for p in s.swarm])
    copyto!(M, s.p, s.swarm[j])
    return s
end
function step_solver!(mp::AbstractManoptProblem, s::ParticleSwarmState, ::Any)
    M = get_manifold(mp)
    for i in 1:length(s.swarm)
        inverse_retract!(
            M,
            s.cognitive_vector,
            s.swarm[i],
            s.positional_best[i],
            s.inverse_retraction_method,
        )
        inverse_retract!(M, s.social_vector, s.swarm[i], s.p, s.inverse_retraction_method)
        s.velocity[i] .=
            s.inertia .* s.velocity[i] .+
            s.cognitive_weight .* rand(1) .* s.cognitive_vector .+
            s.social_weight .* rand(1) .* s.social_vector
        copyto!(M, s.q, s.swarm[i])
        retract!(M, s.swarm[i], s.swarm[i], s.velocity[i], s.retraction_method)
        vector_transport_to!(
            M, s.velocity[i], s.q, s.velocity[i], s.swarm[i], s.vector_transport_method
        )
        if get_cost(mp, s.swarm[i]) < get_cost(mp, s.positional_best[i])
            copyto!(M, s.positional_best[i], s.swarm[i])
            if get_cost(mp, s.positional_best[i]) < get_cost(mp, s.p)
                copyto!(M, s.p, s.positional_best[i])
            end
        end
    end
end

#
# Stopping Criteria
#
@doc raw"""
    StopWhenSwarmVelocityLess <: StoppingCriterion

Stopping criterion for [`particle_swarm`](@ref), when the velocity of the swarm
is less than a threshold.

# Fields
* `threshold`:      the threshold
* `at_iteration`:   store the iteration the stopping criterion was (last) fulfilled
* `reason`:         store the reason why the stopping criterion was fulfilled, see [`get_reason`](@ref)
* `velocity_norms`: interim vector to store the norms of the velocities before computing its norm

# Constructor

    StopWhenSwarmVelocityLess(tolerance::Float64)

initialize the stopping criterion to a certain `tolerance`.
"""
mutable struct StopWhenSwarmVelocityLess{F<:Real} <: StoppingCriterion
    threshold::F
    at_iteration::Int
    velocity_norms::Vector{F}
    StopWhenSwarmVelocityLess(tolerance::F) where {F} = new{F}(tolerance, -1, F[])
end
# It just indicates loss of velocity, not convergence to a minimizer
indicates_convergence(c::StopWhenSwarmVelocityLess) = false
function (c::StopWhenSwarmVelocityLess)(
    mp::AbstractManoptProblem, pss::ParticleSwarmState, i::Int
)
    if i == 0 # reset on init
        c.at_iteration = -1
        # init to correct length
        c.velocity_norms = zeros(eltype(c.velocity_norms), length(pss.swarm))
        return false
    end
    M = get_manifold(mp)
    c.velocity_norms .= [norm(M, p, X) for (p, X) in zip(pss.swarm, pss.velocity)]
    if i > 0 && norm(c.velocity_norms) < c.threshold
        c.at_iteration = i
        return true
    end
    return false
end
function get_reason(c::StopWhenSwarmVelocityLess)
    if (c.at_iteration >= 0) && (norm(c.velocity_norms) < c.threshold)
        return "At iteration $(c.at_iteration) the algorithm reached a velocity of the swarm ($(norm(c.velocity_norms))) less than the threshold ($(c.threshold)).\n"
    end
    return ""
end
function status_summary(c::StopWhenSwarmVelocityLess)
    has_stopped = (c.at_iteration >= 0) && (norm(c.velocity_norms) < c.threshold)
    s = has_stopped ? "reached" : "not reached"
    return "swarm velocity norm < $(c.threshold):\t$s"
end
function show(io::IO, c::StopWhenSwarmVelocityLess)
    return print(io, "StopWhenSwarmVelocityLess($(c.threshold))\n    $(status_summary(c))")
end
