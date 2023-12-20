# [Cyclic proximal point](@id CPPSolver)

The Cyclic Proximal Point (CPP) algorithm aims to minimize

```math
F(x) = \sum_{i=1}^c f_i(x)
```

assuming that the proximal maps ``\operatorname{prox}_{λ f_i}(x)``
are given in closed form or can be computed efficiently (at least approximately).

The algorithm then cycles through these proximal maps, where the type of cycle
might differ and the proximal parameter ``λ_k`` changes after each cycle ``k``.

For a convergence result on
[Hadamard manifolds](https://en.wikipedia.org/wiki/Hadamard_manifold)
see [Bacak:2014](@citet*).

```@docs
cyclic_proximal_point
cyclic_proximal_point!
```

## [Technical details](@id sec-cppa-technical-details)

The [`cyclic_proximal_point`](@ref) solver requires no additional functions to be available for your manifold, besides the ones you use in the proximal maps.

By default, one of the stopping criteria is [`StopWhenChangeLess`](@ref),
which either requires

* An [`inverse_retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `inverse_retraction_method=` or `inverse_retraction_method_dual=` (for ``\mathcal N``) does not have to be specified or the [`distance`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.distance-Tuple{AbstractManifold,%20Any,%20Any})`(M, p, q)` for said default inverse retraction.

## State

```@docs
CyclicProximalPointState
```

## Debug functions

```@docs
DebugProximalParameter
```

## Record functions

```@docs
RecordProximalParameter
```

## Literature

```@bibliography
Pages = ["cyclic_proximal_point.md"]
Canonical=false
```
