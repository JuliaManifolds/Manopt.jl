# [Nelder Mead method](@id NelderMeadSolver)

```@meta
CurrentModule = Manopt
```

```@docs
    NelderMead
    NelderMead!
```

## State

```@docs
    NelderMeadState
```

## Simplex

```@docs
NelderMeadSimplex
```

## Additional stopping criteria

```@docs
StopWhenPopulationConcentrated
```

## [Technical details](@id sec-NelderMead-technical-details)

The [`NelderMead`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* An [`inverse_retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `inverse_retraction_method=` does not have to be specified.
* The [`distance`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.distance-Tuple{AbstractManifold,%20Any,%20Any})`(M, p, q)` when using the default stopping criterion, which includes [`StopWhenPopulationConcentrated`](@ref).
* Within the default initialization [`rand`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#Base.rand-Tuple{AbstractManifold})`(M)` is used to generate the initial population
* A [`mean`](https://juliamanifolds.github.io/Manifolds.jl/stable/features/statistics.html#Statistics.mean-Tuple{AbstractManifold,%20AbstractVector,%20AbstractVector,%20ExtrinsicEstimation})`(M, population)` has to be available, for example by loading [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/stable/) and its [statistics](https://juliamanifolds.github.io/Manifolds.jl/stable/features/statistics.html) tools