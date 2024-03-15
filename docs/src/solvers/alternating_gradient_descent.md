# [Alternating gradient descent](@id solver-alternating-gradient-descent)

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

* The problem has to be phrased on a [`ProductManifold`](@extref ManifoldsBase ProductManifold), to be able to
alternate between parts of the input.
* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* By default alternating gradient descent uses [`ArmijoLinesearch`](@ref) which requires [`max_stepsize`](@ref)`(M)` to be set and an implementation of [`inner`](@extref `ManifoldsBase.inner-Tuple{AbstractManifold, Any, Any, Any}`)`(M, p, X)`.
* By default the tangent vector storing the gradient is initialized calling [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.
