#
# State
#
@doc """
    ParticleSwarmState{P,T} <: AbstractManoptSolverState

Describes a particle swarm optimizing algorithm, with

# Fields

* `cognitive_weight`: a cognitive weight factor
* `inertia`:          the inertia of the particles
$(_var(:Field, :inverse_retraction_method))
$(_var(:Field, :retraction_method))
* `social_weight`:    a social weight factor
$(_var(:Field, :stopping_criterion, "stop"))
$(_var(:Field, :vector_transport_method))
* `velocity`:         a set of tangent vectors (of type `AbstractVector{T}`) representing the velocities of the particles

# Internal and temporary fields

* `cognitive_vector`: temporary storage for a tangent vector related to `cognitive_weight`
$(_var(:Field, :p; add = " storing the best point visited by all particles"))
* `positional_best`:  storing the best position ``p_i`` every single swarm participant visited
$(_var(:Field, :p, "q"; add = " serving as temporary storage for interims results; avoids allocations"))
* `social_vec`:       temporary storage for a tangent vector related to `social_weight`
* `swarm`:            a set of points (of type `AbstractVector{P}`) on a manifold ``$(_math(:Sequence, "a", "i", "1", "N"))``

# Constructor

    ParticleSwarmState(M, initial_swarm, velocity; kawrgs...)

construct a particle swarm solver state for the manifold `M` starting with the initial population `initial_swarm` with `velocities`.
The `p` used in the following defaults is the type of one point from the swarm.

# Keyword arguments

* `cognitive_weight=1.4`
* `inertia=0.65`
$(_var(:Keyword, :inverse_retraction_method))
$(_var(:Keyword, :retraction_method))
* `social_weight=1.4`
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(500)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-4)`"))
$(_var(:Keyword, :vector_transport_method))

# See also

[`particle_swarm`](@ref)
"""
mutable struct ParticleSwarmState{
        P,
        T,
        TX <: AbstractVector{P},
        TVelocity <: AbstractVector{T},
        TParams <: Real,
        TStopping <: StoppingCriterion,
        TRetraction <: AbstractRetractionMethod,
        TInvRetraction <: AbstractInverseRetractionMethod,
        TVTM <: AbstractVectorTransportMethod,
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
            inertia = 0.65,
            social_weight = 1.4,
            cognitive_weight = 1.4,
            stopping_criterion::SCT = StopAfterIteration(500) | StopWhenChangeLess(M, 1.0e-4),
            retraction_method::RTM = default_retraction_method(M, eltype(swarm)),
            inverse_retraction_method::IRM = default_inverse_retraction_method(M, eltype(swarm)),
            vector_transport_method::VTM = default_vector_transport_method(M, eltype(swarm)),
        ) where {
            P,
            T,
            VP <: AbstractVector{<:P},
            VT <: AbstractVector{<:T},
            RTM <: AbstractRetractionMethod,
            SCT <: StoppingCriterion,
            IRM <: AbstractInverseRetractionMethod,
            VTM <: AbstractVectorTransportMethod,
        }
        s = new{
            P, T, VP, VT, typeof(inertia + social_weight + cognitive_weight), SCT, RTM, IRM, VTM,
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
function set_parameter!(pss::ParticleSwarmState, ::Val{:Population}, swarm)
    return pss.swarm = swarm
end
function get_parameter(pss::ParticleSwarmState, ::Val{:Population})
    return pss.swarm
end

#
# Constructors
#
_doc_swarm = raw"``S = \{s_1,…, s_n\}``"

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

_doc_PSO = """
    patricle_swarm(M, f; kwargs...)
    patricle_swarm(M, f, swarm; kwargs...)
    patricle_swarm(M, mco::AbstractManifoldCostObjective; kwargs..)
    patricle_swarm(M, mco::AbstractManifoldCostObjective, swarm; kwargs..)
    particle_swarm!(M, f, swarm; kwargs...)
    particle_swarm!(M, mco::AbstractManifoldCostObjective, swarm; kwargs..)

perform the particle swarm optimization algorithm (PSO) to solve

$(_problem(:Default))

PSO starts with an initial `swarm` [BorckmansIshtevaAbsil:2010](@cite) of points
on the manifold.
If no `swarm` is provided, the `swarm_size` keyword is used to generate random points.
The computation can be performed in-place of `swarm`.

To this end, a swarm $_doc_swarm of particles is moved around the manifold `M` in the following manner.
For every particle ``s_k^{(i)}`` the new particle velocities ``X_k^{(i)}`` are computed in every step ``i`` of the algorithm by

```math
X_k^{(i)} = ω $(_math(:vector_transport, :symbol, "s_k^{(i)", "s_k^{(i-1)}")) X_k^{(i-1)} + c r_1  $(_tex(:invretr))_{s_k^{(i)}}(p_k^{(i)}) + s r_2 $(_tex(:invretr))_{s_k^{(i)}}(p),
```


where
* ``s_k^{(i)}`` is the current particle position,
* ``ω`` denotes the inertia,
* ``c`` and ``s`` are a cognitive and a social weight, respectively,
* ``r_j``, ``j=1,2`` are random factors which are computed new for each particle and step
* $(_math(:vector_transport, :symbol)) is a vector transport, and
* $(_tex(:invretr)) is an inverse retraction

Then the position of the particle is updated as

$_doc_particle_update

Then the single particles best entries ``p_k^{(i)}`` are updated as

$_doc_swarm_best

and the global best position

$_doc_swarm_global_best

# Input

$(_var(:Argument, :M; type = true))
$(_var(:Argument, :f))
* `swarm = [rand(M) for _ in 1:swarm_size]`: an initial swarm of points.

Instead of a cost function `f` you can also provide an [`AbstractManifoldCostObjective`](@ref) `mco`.

# Keyword Arguments

* `cognitive_weight=1.4`: a cognitive weight factor
* `inertia=0.65`: the inertia of the particles
$(_var(:Keyword, :inverse_retraction_method))
$(_var(:Keyword, :retraction_method))
* `social_weight=1.4`: a social weight factor
* `swarm_size=100`: swarm size, if it should be generated randomly
$(_var(:Keyword, :stopping_criterion; default = "[`StopAfterIteration`](@ref)`(500)`$(_sc(:Any))[`StopWhenChangeLess`](@ref)`(1e-4)`"))
$(_var(:Keyword, :vector_transport_method))
* `velocity`:                  a set of tangent vectors (of type `AbstractVector{T}`) representing the velocities of the particles, per default a random tangent vector per initial position

$(_note(:OtherKeywords))
If you provide the objective directly, these decorations can still be specified

$(_note(:OutputSection))
"""
@doc "$(_doc_PSO)"
function particle_swarm(M::AbstractManifold, f; swarm_size = 100, kwargs...)
    return particle_swarm(M, f, [rand(M) for _ in 1:swarm_size]; kwargs...)
end
function particle_swarm(
        M::AbstractManifold,
        f,
        swarm::AbstractVector;
        velocity::AbstractVector = [rand(M; vector_at = y) for y in swarm],
        kwargs...,
    )
    f_ = _ensure_mutating_cost(f, first(swarm))
    swarm_ = [_ensure_mutating_variable(s) for s in swarm]
    velocity_ = [_ensure_mutating_variable(v) for v in velocity]
    mco = ManifoldCostObjective(f_)
    rs = particle_swarm(M, mco, swarm_; velocity = velocity_, kwargs...)
    return _ensure_matching_output(first(swarm), rs)
end
function particle_swarm(
        M::AbstractManifold, mco::O, swarm::AbstractVector; kwargs...
    ) where {O <: Union{AbstractManifoldCostObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(particle_swarm; kwargs...)
    new_swarm = [copy(M, xi) for xi in swarm]
    return particle_swarm!(M, mco, new_swarm; kwargs...)
end
calls_with_kwargs(::typeof(particle_swarm)) = (particle_swarm!,)

@doc "$(_doc_PSO)"
function particle_swarm!(M::AbstractManifold, f, swarm::AbstractVector; kwargs...)
    mco = ManifoldCostObjective(f)
    return particle_swarm!(M, mco, swarm; kwargs...)
end
function particle_swarm!(
        M::AbstractManifold,
        mco::O,
        swarm::AbstractVector;
        velocity::AbstractVector = [rand(M; vector_at = y) for y in swarm],
        inertia::Real = 0.65,
        social_weight::Real = 1.4,
        cognitive_weight::Real = 1.4,
        stopping_criterion::StoppingCriterion = StopAfterIteration(500) |
            StopWhenSwarmVelocityLess(1.0e-4),
        retraction_method::AbstractRetractionMethod = default_retraction_method(M, eltype(swarm)),
        inverse_retraction_method::AbstractInverseRetractionMethod = default_inverse_retraction_method(
            M, eltype(swarm)
        ),
        vector_transport_method::AbstractVectorTransportMethod = default_vector_transport_method(
            M, eltype(swarm)
        ),
        kwargs..., #collect rest
    ) where {O <: Union{AbstractManifoldCostObjective, AbstractDecoratedManifoldObjective}}
    keywords_accepted(particle_swarm!; kwargs...)
    dmco = decorate_objective!(M, mco; kwargs...)
    mp = DefaultManoptProblem(M, dmco)
    pss = ParticleSwarmState(
        M,
        swarm,
        velocity;
        inertia = inertia,
        social_weight = social_weight,
        cognitive_weight = cognitive_weight,
        stopping_criterion = stopping_criterion,
        retraction_method = retraction_method,
        inverse_retraction_method = inverse_retraction_method,
        vector_transport_method = vector_transport_method,
    )
    dpss = decorate_state!(pss; kwargs...)
    solve!(mp, dpss)
    return get_solver_return(get_objective(mp), dpss)
end
calls_with_kwargs(::typeof(particle_swarm!)) = (decorate_objective!, decorate_state!)

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
    return
end

#
# Stopping Criteria
#
@doc """
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
mutable struct StopWhenSwarmVelocityLess{F <: Real} <: StoppingCriterion
    threshold::F
    at_iteration::Int
    velocity_norms::Vector{F}
    StopWhenSwarmVelocityLess(tolerance::F) where {F} = new{F}(tolerance, -1, F[])
end
# It just indicates loss of velocity, not convergence to a minimizer
indicates_convergence(c::StopWhenSwarmVelocityLess) = false
function (c::StopWhenSwarmVelocityLess)(
        mp::AbstractManoptProblem, pss::ParticleSwarmState, k::Int
    )
    if k == 0 # reset on init
        c.at_iteration = -1
        # init to correct length
        c.velocity_norms = zeros(eltype(c.velocity_norms), length(pss.swarm))
        return false
    end
    M = get_manifold(mp)
    c.velocity_norms .= [norm(M, p, X) for (p, X) in zip(pss.swarm, pss.velocity)]
    if k > 0 && norm(c.velocity_norms) < c.threshold
        c.at_iteration = k
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
