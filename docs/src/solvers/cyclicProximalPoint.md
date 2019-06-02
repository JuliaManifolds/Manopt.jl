# [Cyclic Proximal Point](@id CPPSolver)

The Cyclic Proximal Point (CPP) algorithm is a [Proximal Problem](@ref ProximalProblem).

It aims to minimize

```math
F(x) = \sum_{i=1}^c f_i(x)
```

assuming that the [proximal maps](@ref proximalMapFunctions) $\operatorname{prox}_{\lambda f_i}(x)$
are given in closed form or can be computed efficiently (at least approximately).

The algorithm then cycles through these proximal maps, where the type of cycle
might differ and the proximal parameter $\lambda_k$ changes after each cycle $k$.

For a convergence result on
[Hadamard manifolds](https://en.wikipedia.org/wiki/Hadamard_manifold)
see [[Bačák, 2014](#Bačák2014)].

```@docs
cyclicProximalPoint
```

```@docs
CyclicProximalPointOptions
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

```@raw html
<ul>
<li id="Bačák2014">[<a>Bačák, 2014</a>]
  Bačák, M: <emph>Computing Medians and Means in Hadamard Spaces.</emph>,
  SIAM Journal on Optimization, Volume 24, Number 3, pp. 1542–1566,
  doi: <a href="https://doi.org/10.1137/140953393">10.1137/140953393</a>,
  arxiv: <a href="https://arxiv.org/abs/1210.2145">1210.2145</a>.
  </li>
</ul>
```
