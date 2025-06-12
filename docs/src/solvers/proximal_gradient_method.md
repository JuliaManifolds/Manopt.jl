# Proximal Gradient Method

```@docs
proximal_gradient_method
proximal_gradient_method!
```

```@docs
Manopt.ProxGradAcceleration
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
```

## Helpers and internal functions

```@docs
Manopt.get_cost_h
Manopt.get_cost_g
Manopt.default_stepsize(::AbstractManifold, ::Type{<:ProximalGradientMethodState})
```

## Literature

```@bibliography
Pages = ["proximal_gradient_method.md"]
Canonical=false
```