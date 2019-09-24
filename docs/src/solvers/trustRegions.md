# [The Riemannian Trust-Regions Solver](@id trustRegions)

The aim is to solve an optimization problem on a manifold

```math
min_{x \in \mathcal{M}} f(x)
```

## Initialization

Initialize $x_0 = x$ if an initial point $x$ is given by the caller or
$x_0 = \operatorname{randomMPoint}(\mathcal{M})$

## Iteration

Repeat until a convergence criterion is reached

## Result

The result is given by the last computed $x_k$.

## Interface

```@docs
trustRegions
```

## Options

```@docs
TrustRegionOptions
```
