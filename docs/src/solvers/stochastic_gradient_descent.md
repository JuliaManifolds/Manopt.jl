# [Stochastic gradient descent](@id StochasticGradientDescentSolver)

```@meta
CurrentModule = Manopt
```

```@docs
stochastic_gradient_descent
stochastic_gradient_descent!
```

## State

```@docs
StochasticGradientDescentState
```

Additionally, the options share a [`DirectionUpdateRule`](@ref),
so you can also apply [`MomentumGradient`](@ref) and [`AverageGradient`](@ref) here.
The most inner one should always be.

```@docs
AbstractGradientGroupProcessor
StochasticGradient
```

## [Technical details](@id sec-sgd-technical-details)

The [`stochastic_gradient_descent`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
