# Stochastic gradient descent

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
Manopt.default_stepsize(::AbstractManifold, ::Type{StochasticGradientDescentState})
```

Additionally, the options share a [`DirectionUpdateRule`](@ref),
so you can also apply [`MomentumGradient`](@ref) and [`AverageGradient`](@ref) here.
The most inner one should always be.

```@docs
StochasticGradient
```

which internally uses

```@docs
AbstractGradientGroupDirectionRule
StochasticGradientRule
```

## [Technical details](@id sec-sgd-technical-details)

The [`stochastic_gradient_descent`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
