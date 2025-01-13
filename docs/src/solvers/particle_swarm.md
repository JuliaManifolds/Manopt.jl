# Particle swarm optimization

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

## Stopping criteria

```@docs
StopWhenSwarmVelocityLess
```

## [Technical details](@id sec-arc-technical-details)

The [`particle_swarm`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* An [`inverse_retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `inverse_retraction_method=` does not have to be specified.
* A [`vector_transport_to!`](@extref ManifoldsBase :doc:`vector_transports`)`M, Y, p, X, q)`; it is recommended to set the [`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `vector_transport_method=` does not have to be specified.
* By default the stopping criterion uses the [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`) as well, to stop when the norm of the gradient is small, but if you implemented `inner`, the norm is provided already.
* Tangent vectors storing the social and cognitive vectors are initialized calling [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.
* The [`distance`](@extref `ManifoldsBase.distance-Tuple{AbstractManifold, Any, Any}`)`(M, p, q)` when using the default stopping criterion, which uses [`StopWhenChangeLess`](@ref).

## Literature

```@bibliography
Pages = ["particle_swarm.md"]
Canonical=false
```
