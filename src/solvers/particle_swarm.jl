@doc raw"""
    patricle_swarm(M, F)

perform the particle swarm optimization algorithm (PSO), starting with the initial particle positions $x_0$[^Borckmans2010].
The aim of PSO is to find the particle position $g$ on the `Manifold M` that solves
```math
\min_{x \in \mathcal{M}} F(x).
```
To this end, a swarm of particles is moved around the `Manifold M` in the following manner.
For every particle $k$ we compute the new particle velocities $v_k^{(i)}$ in every step $i$ of the algorithm by
```math
v_k^{(i)} = \omega \, \operatorname{T}_{x_k^{(i)}\gets x_k^{(i-1)}}v_k^{(i-1)} + c \,  r_1  \operatorname{retr}_{x_k^{(i)}}^{-1}(p_k^{(i)}) + s \,  r_2 \operatorname{retr}_{x_k^{(i)}}^{-1}(g),
```
where $x_k^{(i)}$ is the current particle position, $\omega$ denotes the inertia,
$c$ and $s$ are a cognitive and a social weight, respectively,
$r_j$, $~j=1,2$ are random factors which are computed new for each particle and step,
$\operatorname{retr}^{-1}$ denotes an inverse retraction on the `Manifold` `M`, and
$\operatorname{T}$ is a vector transport.

Then the position of the particle is updated as
```math
x_k^{(i+1)} = \operatorname{retr}_{x_k^{(i)}}(v_k^{(i)}),
```
where $\operatorname{retr}$ denotes a retraction on the `Manifold` `M`. At the end of each step for every particle, we set
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
i.e. $p_k^{(i)}$ is the best known position for the particle $k$ and $g^{(i)}$ is the global best known position ever visited up to step $i$.


[^Borckmans2010]:
    > P. B. Borckmans, M. Ishteva, P.-A. Absil, __A Modified Particle Swarm Optimization Algorithm for the Best Low Multilinear Rank Approximation of Higher-Order Tensors__,
    > In: Dorigo M. et al. (eds) Swarm Intelligence. ANTS 2010. Lecture Notes in Computer Science, vol 6234. Springer, Berlin, Heidelberg,
    > doi [10.1007/978-3-642-15461-4_2](https://doi.org/10.1007/978-3-642-15461-4_2)


# Input
* `M` – a manifold $\mathcal M$
* `F` – a cost function $F\colon\mathcal M\to\mathbb R$ to minimize

# Optional
* `n` - (`100`) number of random initial positions of x0
* `x0` – the initial positions of each particle in the swarm $x_k^{(0)} ∈ \mathcal M$ for $k = 1, \dots, n$, per default these are n [`random_point`](@ref)s
* `velocity` – a set of tangent vectors (of type `AbstractVector{T}`) representing the velocities of the particles, per default a [`random_tangent`](@ref) per inital position
* `inertia` – (`0.65`) the inertia of the patricles
* `social_weight` – (`1.4`) a social weight factor
* `cognitive_weight` – (`1.4`) a cognitive weight factor
* `retraction_method` – `ExponentialRetraction` a `retraction(M,x,ξ)` to use.
* `inverse_retraction_method` - `LogarithmicInverseRetraction` an `inverse_retraction(M,x,y)` to use.
* `vector_transport_mthod` - `ParallelTransport` a vector transport method to use.
* `stopping_criterion` – ([`StopWhenAny`](@ref)`(`[`StopAfterIteration`](@ref)`(500)`, [`StopWhenChangeLess`](@ref)`(10^{-4})))`
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
* `return_options` – (`false`) – if activated, the extended result, i.e. the
    complete [`Options`](@ref) are returned. This can be used to access recorded values.
    If set to false (default) just the optimal value `xOpt` if returned

...
and the ones that are passed to [`decorate_options`](@ref) for decorators.

# Output
* `g` – the resulting point of PSO
OR
* `options` - the options returned by the solver (see `return_options`)
"""
function particle_swarm(
    M::Manifold,
    F::TF;
    n::Int = 100,
    x0::AbstractVector = [random_point(M) for i in 1:n],
    velocity::AbstractVector = [random_tangent(M, y) for y in x0],
    inertia::Real = 0.65,
    social_weight::Real = 1.4,
    cognitive_weight::Real = 1.4,
    stopping_criterion::StoppingCriterion = StopWhenAny(
        StopAfterIteration(200),
        StopWhenChangeLess(10.0^-4),
    ),
    retraction_method::AbstractRetractionMethod = ExponentialRetraction(),
    inverse_retraction_method::AbstractInverseRetractionMethod = LogarithmicInverseRetraction(),
    vector_transport_method::AbstractVectorTransportMethod = ParallelTransport(),
    return_options = false,
    kwargs..., #collect rest
) where {TF}
    p = CostProblem(M, F)
    o = ParticleSwarmOptions(
        x0,
        velocity,
        inertia,
        social_weight,
        cognitive_weight,
        stopping_criterion,
        retraction_method,
        inverse_retraction_method,
        vector_transport_method,
    )
    o = decorate_options(o; kwargs...)
    resultO = solve(p, o)
    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end

#
# Solver functions
#
function initialize_solver!(p::CostProblem, o::ParticleSwarmOptions)
    j = argmin([p.cost(y) for y in o.x])
    return o.g = deepcopy(o.x[j])
end
function step_solver!(p::CostProblem, o::ParticleSwarmOptions, iter)
    for i in 1:length(o.x)
        o.velocity[i] =
            o.inertia .* o.velocity[i] +
            o.cognitive_weight * rand(1) .*
            inverse_retract(p.M, o.x[i], o.p[i], o.inverse_retraction_method) +
            o.social_weight * rand(1) .*
            inverse_retract(p.M, o.x[i], o.g, o.inverse_retraction_method)
        xOld = o.x[i]
        o.x[i] = retract(p.M, o.x[i], o.velocity[i], o.retraction_method)
        o.velocity[i] =
            vector_transport_to(p.M, xOld, o.velocity[i], o.x[i], o.vector_transport_method)
        if p.cost(o.x[i]) < p.cost(o.p[i])
            o.p[i] = o.x[i]
            if p.cost(o.p[i]) < p.cost(o.g)
                o.g = o.p[i]
            end
        end
    end
end
get_solver_result(o::ParticleSwarmOptions) = o.g
