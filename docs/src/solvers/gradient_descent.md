# Gradient descent

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
AdaptiveDirection
AverageGradient
DirectionUpdateRule
IdentityUpdateRule
MomentumGradient
Nesterov
PreconditionedDirection
```

where the [`AdaptiveDirection`](@ref) can be configured with different adaptive rules

```@docs
BasicDirection
AdamDirection
```

Internally the direction rules use the [`ManifoldDefaultsFactory`](@ref) and produce the (not exported) actual rules

```@docs
Manopt.AdaptiveDirectionRule
Manopt.AverageGradientRule
Manopt.ConjugateDescentCoefficientRule
Manopt.MomentumGradientRule
Manopt.NesterovRule
Manopt.PreconditionedDirectionRule
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

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* By default gradient descent uses [`ArmijoLinesearch`](@ref) which requires [`max_stepsize`](@ref)`(M)` to be set and an implementation of [`inner`](@extref `ManifoldsBase.inner-Tuple{AbstractManifold, Any, Any, Any}`)`(M, p, X)`.
* By default the stopping criterion uses the [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`) as well, to stop when the norm of the gradient is small, but if you implemented `inner`, the norm is provided already.
* By default the tangent vector storing the gradient is initialized calling [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.

## Literature

```@bibliography
Pages = ["gradient_descent.md"]
Canonical=false

Luenberger:1972
```
