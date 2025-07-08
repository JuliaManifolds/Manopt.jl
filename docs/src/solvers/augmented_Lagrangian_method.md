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

## Helping functions

```@docs
AugmentedLagrangianCost
AugmentedLagrangianGrad
```

## [Technical details](@id sec-agd-technical-details)

The [`augmented_Lagrangian_method`](@ref) solver requires the following functions of a manifold to be available

* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.
* Everything the subsolver requires, which by default is the [`quasi_Newton`](@ref) method
* A [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.


## Literature

```@bibliography
Pages = ["augmented_Lagrangian_method.md"]
Canonical=false
```
