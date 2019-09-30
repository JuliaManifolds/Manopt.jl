# [The Riemannian Trust-Regions Solver](@id trustRegions)

The aim is to solve an optimization problem on a manifold

```math
min_{x \in \mathcal{M}} f(x)
```

by using the Riemannian trust-regions solver.

## Initialization

Initialize $x_0 = x$ if an initial point $x$ is given by the caller or set
$x_0 = \operatorname{randomMPoint}(\mathcal{M})$.

## Iteration

Repeat until a convergence criterion is reached

1. If the initial point $x_0$ was chosen randomly, set
    $\eta = \operatorname{randomTVector}(\mathcal{M}, x)$ and multiply it by
    $\sqrt{\sqrt{\operatorname{eps}(Float64)}}$ as long as its norm is greater than
    the current trust-regions radius $\Delta$. If the initial point $x_0$ is given
    by the caller, set $\eta = \operatorname{zeroTVector}(\mathcal{M}, x)$.
2. Obtain $\eta_k$ by (approximately) solving the trust-regions subproblem.
    The problem as well as the solution method is described in the
    [`truncatedConjugateGradient`](@ref).
3. 

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
