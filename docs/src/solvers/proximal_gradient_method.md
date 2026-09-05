# Proximal gradient method

```@meta
CurrentModule = Manopt
```

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

## Helpers

```@docs
ProximalGradientNonsmoothSubgradient
ProximalGradientNonsmoothCost
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