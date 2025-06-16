# Proximal Gradient Method

```@docs
proximal_gradient_method
proximal_gradient_method!
```

```@docs
Manopt.ProximalGradientMethodAcceleration
```

## State

```@docs
ProximalGradientMethodState
```

## Stopping criteria

```@docs
StopWhenGradientMappingNormLess
```

## Stepsize

```@docs
ProximalGradientMethodBacktracking
Manopt.ProximalGradientMethodBacktrackingStepsize
```

## Helpers and internal functions

```@docs
Manopt.get_cost_smooth
Manopt.default_stepsize(::AbstractManifold, ::Type{<:ProximalGradientMethodState})
```

## Literature

```@bibliography
Pages = ["proximal_gradient_method.md"]
Canonical=false
```