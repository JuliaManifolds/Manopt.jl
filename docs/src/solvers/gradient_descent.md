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

## Literature

```@bibliography
Pages = ["solvers/gradient_descent.md"]
Canonical=false

Luenberger:1972
```
