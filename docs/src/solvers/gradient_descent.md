# [Gradient Descent](@id GradientDescentSolver)

```@meta
CurrentModule = Manopt
```

```@docs
  gradient_descent
```

## Options

```@docs
AbstractGradientDescentOptions
GradientDescentOptions
```

## Direction Update Rules

A field of the options is the `direction`, a [`DirectionUpdateRule`](@ref), which by default [`Gradient`](@ref) just evaluates the gradient but can be enhanced for example to

```@docs
DirectionUpdateRule
Gradient
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
