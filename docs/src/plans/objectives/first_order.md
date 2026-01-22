# [First order objectives](@id first_order_objectives)

```@meta
CurrentModule = Manopt
```

## Gradient objectives

```@docs
AbstractManifoldFirstOrderObjective
ManifoldFirstOrderObjective
ManifoldAlternatingGradientObjective
ManifoldStochasticGradientObjective
NonlinearLeastSquaresObjective
```

While the [`ManifoldFirstOrderObjective`](@ref) allows to provide different
first order information, there are also its shortcuts, mainly for historical reasons,
but also since these are the most commonly used ones.

```@docs
ManifoldGradientObjective
ManifoldCostGradientObjective
```

### Access functions

```@docs
get_gradient
get_gradients
get_differential
get_residuals
get_residuals!
```

and internally

```@docs
get_differential_function
get_gradient_function
```

#### Robustifiers

Inside the [`NonlinearLeastSquaresObjective`](@ref) one can use robustifiers. The following ones are provided

```@docs
SoftL1Robustifier
AbstractRobustifierFunction
CauchyRobustifier
TolerantRobustifier
TukeyRobustifier
ComposedRobustifierFunction
ArctanRobustifier
ScaledRobustifierFunction
RobustifierFunction
IdentityRobustifier
HuberRobustifier
get_robustifier_values
```

## Subgradient objectives

```@docs
ManifoldSubgradientObjective
```

#### Access functions

```@docs
get_subgradient
```


and internally

```@docs
get_subgradient_function
```

## Proximal map objectives

```@docs
ManifoldProximalMapObjective
```

### Access functions

```@docs
get_proximal_map
```