# [Particle swarm optimization](@id ParticleSwarmSolver)

```@meta
CurrentModule = Manopt
```

```@docs
  particle_swarm
  particle_swarm!
```

## State

```@docs
ParticleSwarmState
```

## [Technical details](@id sec-arc-technical-details)

The [`particle_swarm`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* An [`inverse_retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `inverse_retraction_method=` does not have to be specified.
* A [`vector_transport_to!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/vector_transports/)`M, Y, p, X, q)`; it is recommended to set the [`default_vector_transport_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/vector_transports/#ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `vector_transport_method=` does not have to be specified.
* By default the stopping criterion uses the [`norm`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#LinearAlgebra.norm-Tuple{AbstractManifold,%20Any,%20Any}) as well, to stop when the norm of the gradient is small, but if you implemented `inner` from the last point, the norm is provided already.
* Tangent vectors storing the social and cognitive vectors are initialized calling [`zero_vector`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.zero_vector-Tuple{AbstractManifold,%20Any})`(M,p)`.
* A [`copyto!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#Base.copyto!-Tuple{AbstractManifold,%20Any,%20Any})`(M, q, p)` and [`copy`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#Base.copy-Tuple{AbstractManifold,%20Any})`(M,p)` for points.
* The [`distance`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.distance-Tuple{AbstractManifold,%20Any,%20Any})`(M, p, q)` when using the default stopping criterion, which uses [`StopWhenChangeLess`](@ref).

## Literature

```@bibliography
Pages = ["particle_swarm.md"]
Canonical=false
```
