# Vector Bundle Newton Method

```@meta
CurrentModule = Manopt
```

```@docs
  vectorbundle_newton
  vectorbundle_newton!
```

## Problem

```@docs
VectorBundleManoptProblem
Manopt.get_manifold(::VectorBundleManoptProblem)
```

## State

```@docs
VectorBundleNewtonState
```

## [Step size](@id Sec-VectorBundleNewton-Stepsize)

```@docs
AffineCovariantStepsize
```

## Internal Functions

```@autodocs
Modules = [Manopt]
Pages = ["vectorbundle_newton.jl"]
Order = [:function]
Public=false
Private=true
```

## Literature

```@bibliography
Pages = ["vectorbundle_newton.md"]
Canonical=false
```