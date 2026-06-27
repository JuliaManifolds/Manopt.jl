# Gradient Sampling Algorithm

```@meta
CurrentModule = Manopt
```

```@docs
gradient_sampling
gradient_sampling!
```

## State

```@docs
GradientSamplingState
```

## [Technical details](@id sec-gradient-sampling-technical-details)

The [`gradient_sampling`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* By default gradient sampling uses [`ArmijoLinesearch`](@ref) which requires [`max_stepsize`](@ref)`(M)` to be set and an implementation of [`inner`](@extref `ManifoldsBase.inner-Tuple{AbstractManifold, Any, Any, Any}`)`(M, p, X)`.
* By default the stopping criterion uses the [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`) as well, to stop when the norm of the gradient is small, but if you implemented [`inner`](@extref `ManifoldsBase.inner-Tuple{AbstractManifold, Any, Any, Any}`)`(M, p, X)`, the norm is provided already.
* By default the tangent vector storing the gradient is initialized calling [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.
* A [`vector_transport_to!`](@extref ManifoldsBase :doc:`vector_transports`)`M, Y, p, X, q)`; it is recommended to set the [`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `vector_transport_method=` does not have to be specified.

## Literature

```@bibliography
Pages = ["gradient_sampling.md"]
Canonical=false
```
