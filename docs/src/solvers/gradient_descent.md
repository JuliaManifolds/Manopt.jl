# [Gradient Descent](@id GradientDescentSolver)

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

## Direction Update Rules

A field of the options is the `direction`, a [`DirectionUpdateRule`](@ref), which by default [`IdentityUpdateRule`](@ref) just evaluates the gradient but can be enhanced for example to

```@docs
DirectionUpdateRule
IdentityUpdateRule
MomentumGradient
AverageGradient
Nesterov
```

## Debug Actions

```@docs
DebugGradient
DebugGradientNorm
DebugStepsize
```

## Record Actions

```@docs
RecordGradient
RecordGradientNorm
RecordStepsize
```

## [Technical Details](@id GradientDescent-Technical-Details)

The [`gradient_descent`](@ref) solver requires the following functions of your manifold to be available

* A retraction; if you do not want to specify them directly, [`default_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}) should be implemented as well.
* By default gradient descent uses [`ArmijoLinesearch`](@ref) which requires [`max_stepsize`](@ref)`(M)` to be set and an implementation of [`norm`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#LinearAlgebra.norm-Tuple{AbstractManifold,%20Any,%20Any})`(M, p, X)`.
* By default the stopping criterion uses the [`norm`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#LinearAlgebra.norm-Tuple{AbstractManifold,%20Any,%20Any}) as well, to check for a small gradient

## Literature

```@bibliography
Pages = ["gradient_descent.md"]
Canonical=false

Luenberger:1972
```
