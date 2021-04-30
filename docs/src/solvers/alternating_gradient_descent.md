# [Alternating Gradient Descent](@id AlternatingGradientDescentSolver)

```@meta
CurrentModule = Manopt
```

```@docs
alternating_gradient_descent
alternating_gradient_descent!
```

## Problem

```@docs
AlternatingGradientProblem
```

## Options

```@docs
AlternatingGradientDescentOptions
```

Additionally, the options share a [`DirectionUpdateRule`](@ref),
which chooses the current component, so they can be decorated further;
The most inner one should always be the following one though.

```@docs
AlternatingGradient
```
