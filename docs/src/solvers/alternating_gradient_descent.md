# [Alternating gradient descent](@id title-agds)

```@meta
CurrentModule = Manopt
```

```@docs
alternating_gradient_descent
alternating_gradient_descent!
```

## State

```@docs
AlternatingGradientDescentState
```

Additionally, the options share a [`DirectionUpdateRule`](@ref),
which chooses the current component, so they can be decorated further;
The most inner one should always be the following one though.

```@docs
AlternatingGradient
```


## [Technical details](@id sec-agd-technical-details)

The [`alternating_gradient_descent`](@ref) solver requires the following functions of a manifold to be available

* The problem has to be phrased on a [`ProductManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/metamanifolds/#ProductManifold), to be able to
alternate between parts of the input.
* A [retract!](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)ion; it is recommended to set the [`default_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* By default alternating gradient descent uses [`ArmijoLinesearch`](@ref) which requires [`max_stepsize`](@ref)`(M)` to be set and an implementation of [`inner`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.inner-Tuple%7BAbstractManifold,%20Any,%20Any,%20Any%7D)`(M, p, X)`.
* By default the tangent vector storing the gradient is initialized calling [`zero_vector`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.zero_vector-Tuple{AbstractManifold,%20Any})`(M,p)`.
