# Augmented Lagrangian method

```@meta
CurrentModule = Manopt
```

```@docs
  augmented_Lagrangian_method
  augmented_Lagrangian_method!
```

## State

```@docs
AugmentedLagrangianMethodState
```

## [Technical details](@id sec-alm-technical-details)

The [`augmented_Lagrangian_method`](@ref) solver requires the following functions of a manifold to be available

* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.
* Everything the sub solver requires, which by default is the [`quasi_Newton`](@ref) method
* A [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.
* The [`distance`](@extref `ManifoldsBase.distance-Tuple{AbstractManifold, Any, Any}`)`(M, p, q)` with respect to the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`); it measures the length of the step the sub solver took and is hence also required by the default stopping criteria [`StopWhenChangeLess`](@ref) and [`StopWhenStepsizeLess`](@ref).


## Literature

```@bibliography
Pages = ["augmented_Lagrangian_method.md"]
Canonical=false
```
