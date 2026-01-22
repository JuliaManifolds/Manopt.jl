# [Decorators for Objectives](@id decorators_for_objectives)

```@meta
CurrentModule = Manopt
```

```@docs
AbstractDecoratedManifoldObjective
```

An objective can be decorated using the following trait and function to initialize

```@docs
dispatch_objective_decorator
is_objective_decorator
decorate_objective!
```

## [Embedded objectives](@id subsection-embedded-objectives)

```@docs
EmbeddedManifoldObjective
```

## [Scaled objectives](@id subsection-scaled-objectives)

```@docs
ScaledManifoldObjective
```

## [Cache objective](@id subsection-cache-objective)

Since single function calls, for example to the cost or the gradient, might be expensive,
a simple cache objective exists as a decorator, that caches one cost value or gradient.

It can be activated/used with the `cache=` keyword argument available for every solver.

```@docs
Manopt.reset_counters!
Manopt.objective_cache_factory
```

### A simple cache

A first generic cache is always available, but it only caches one gradient and one cost function evaluation (for the same point).

```@docs
SimpleManifoldCachedObjective
```

### A generic cache

For the more advanced cache, you need to implement some type of cache yourself, that provides a `get!`
and implement [`init_caches`](@ref).
This is for example provided if you load [`LRUCache.jl`](https://github.com/JuliaCollections/LRUCache.jl). Then you obtain

```@docs
ManifoldCachedObjective
init_caches
```

## [Count objective](@id subsection-count-objective)

```@docs
ManifoldCountObjective
```

### Internal decorators and functions

```@docs
ReturnManifoldObjective
```
