# [Gradient descent](@id GradientDescentSolver)

```@meta
CurrentModule = Manopt
```

```@docs
  gradient_descent
  gradient_descent!
```

## State

```@docs
GradientDescentState
```

## Direction update rules

A field of the options is the `direction`, a [`DirectionUpdateRule`](@ref), which by default [`IdentityUpdateRule`](@ref) just evaluates the gradient but can be enhanced for example to

```@docs
DirectionUpdateRule
IdentityUpdateRule
MomentumGradient
AverageGradient
Nesterov
```

## Debug actions

```@docs
DebugGradient
DebugGradientNorm
DebugStepsize
```

## Record actions

```@docs
RecordGradient
RecordGradientNorm
RecordStepsize
```

## [Technical details](@id sec-gradient-descent-technical-details)

The [`gradient_descent`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* By default gradient descent uses [`ArmijoLinesearch`](@ref) which requires [`max_stepsize`](@ref)`(M)` to be set and an implementation of [`inner`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.inner-Tuple%7BAbstractManifold,%20Any,%20Any,%20Any%7D)`(M, p, X)`.
* By default the stopping criterion uses the [`norm`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#LinearAlgebra.norm-Tuple{AbstractManifold,%20Any,%20Any}) as well, to stop when the norm of the gradient is small, but if you implemented `inner` from the last point, the norm is provided already.
* By default the tangent vector storing the gradient is initialized calling [`zero_vector`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.zero_vector-Tuple{AbstractManifold,%20Any})`(M,p)`.

## Literature

```@bibliography
Pages = ["gradient_descent.md"]
Canonical=false

Luenberger:1972
```
