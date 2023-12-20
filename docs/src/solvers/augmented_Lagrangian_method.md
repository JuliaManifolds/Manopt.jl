# [Augmented Lagrangian method](@id AugmentedLagrangianSolver)

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

* A [`copyto!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#Base.copyto!-Tuple{AbstractManifold,%20Any,%20Any})`(M, q, p)` and [`copy`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#Base.copy-Tuple{AbstractManifold,%20Any})`(M,p)` for points.
* Everything the subsolver requires, which by default is the [`quasi_Newton`](@ref) method
* A [`zero_vector`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.zero_vector-Tuple{AbstractManifold,%20Any})`(M,p)`.


## Literature

```@bibliography
Pages = ["augmented_Lagrangian_method.md"]
Canonical=false
```
