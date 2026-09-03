# Cyclic proximal point

```@meta
CurrentModule = Manopt
```

The Cyclic Proximal Point (CPP) algorithm aims to minimize

```math
f(p) = \sum_{i=1}^c f_i(p)
```

assuming that the proximal maps ``\operatorname{prox}_{λ f_i}(p)``
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
which requires

* An [`inverse_retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`) to a favorite inverse retraction. If this default is set, an `inverse_retraction_method=` does not have to be specified. Alternatively, the [`distance`](@extref `ManifoldsBase.distance-Tuple{AbstractManifold, Any, Any}`)`(M, p, q)` can be used.

## State

```@docs
CyclicProximalPointState
```

## Literature

```@bibliography
Pages = ["cyclic_proximal_point.md"]
Canonical=false
```
