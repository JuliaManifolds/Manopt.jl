# [Cyclic Proximal Point](@id CPPSolver)

The Cyclic Proximal Point (CPP) algorithm aims to minimize

```math
F(x) = \sum_{i=1}^c f_i(x)
```

assuming that the [proximal maps](@ref proximalMapFunctions) $\operatorname{prox}_{λ f_i}(x)$
are given in closed form or can be computed efficiently (at least approximately).

The algorithm then cycles through these proximal maps, where the type of cycle
might differ and the proximal parameter $λ_k$ changes after each cycle $k$.

For a convergence result on
[Hadamard manifolds](https://en.wikipedia.org/wiki/Hadamard_manifold)
see [Bačák, SIAM J. Optim., 2014](@cite Bacak:2014).

```@docs
cyclic_proximal_point
cyclic_proximal_point!
```

## State

```@docs
CyclicProximalPointState
```

## Debug Functions

```@docs
DebugProximalParameter
```

## Record Functions

```@docs
RecordProximalParameter
```

## Literature

```@bibliography
Pages = ["solvers/cyclic_proximal_point.md"]
Canonical=false
```
