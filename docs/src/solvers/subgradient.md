# [Subgradient method](@id sec-subgradient-method)

```@docs
subgradient_method
subgradient_method!
```

## State

```@docs
SubGradientMethodState
```

For [`DebugAction`](@ref)s and [`RecordAction`](@ref)s to record (sub)gradient,
its norm and the step sizes, see the [gradient descent](gradient_descent.md)
actions.


## [Technical details](@id sec-sgm-technical-details)

The [`subgradient_method`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.

## Literature

```@bibliography
Pages = ["subgradient.md"]
Canonical=false
```