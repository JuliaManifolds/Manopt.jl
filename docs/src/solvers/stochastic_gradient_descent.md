# [Stochastic Gradient Descent](@id StochasticGradientDescentSolver)

```@meta
CurrentModule = Manopt
```

```@docs
stochastic_gradient_descent
stochastic_gradient_descent!
```

## Options

```@docs
StochasticGradientDescentOptions
```

Additionally, the options share a [`DirectionUpdateRule`](@ref),
so you can also apply [`MomentumGradient`](@ref) and [`AverageGradient`](@ref) here.
The most inner one should always be.

```@docs
AbstractStochasticGradientProcessor
StochasticGradient
```
