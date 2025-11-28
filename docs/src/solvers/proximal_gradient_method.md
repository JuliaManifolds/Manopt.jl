# Proximal gradient method

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

## Helping functions

```@docs
ProximalGradientNonsmoothSubgradient
ProximalGradientNonsmoothCost
```

## Stopping criteria

```@docs
StopWhenGradientMappingNormLess
```

## [Stepsize](@id Sec-ProxGrad-Stepsize)

```@docs
ProximalGradientMethodBacktracking
Manopt.ProximalGradientMethodBacktrackingStepsize
```

## Internal functions

```@docs
Manopt.get_cost_smooth
Manopt.default_stepsize(::AbstractManifold, ::Type{<:ProximalGradientMethodState})
```

## Literature

```@bibliography
Pages = ["proximal_gradient_method.md"]
Canonical=false
```