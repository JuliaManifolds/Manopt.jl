# [Adaptive regularization with cubics](@id ARSSection)



```@meta
CurrentModule = Manopt
```

```@docs
adaptive_regularization_with_cubics
adaptive_regularization_with_cubics!
```

## State

```@docs
AdaptiveRegularizationState
```

## Sub solvers

There are several ways to approach the subsolver. The default is the first one.

## Lanczos iteration

```@docs
Manopt.LanczosState
```

## (Conjugate) gradient descent

There is a generic objective, that implements the sub problem

```@docs
AdaptiveRagularizationWithCubicsModelObjective
```

Since the sub problem is given on the tangent space, you have to provide

```
arc_obj = AdaptiveRagularizationWithCubicsModelObjective(mho, σ)
sub_problem = DefaultProblem(TangentSpaceAt(M,p), arc_obj)
```

where `mho` is the Hessian objective of `f` to solve.
Then use this for the `sub_problem` keyword
and use your favourite gradient based solver for the `sub_state` keyword, for example a
[`ConjugateGradientDescentState`](@ref)

## Additional stopping criteria

```@docs
StopWhenAllLanczosVectorsUsed
StopWhenFirstOrderProgress
```

## [Technical details](@id sec-arc-technical-details)

The [`adaptive_regularization_with_cubics`](@ref) requires the following functions
of a manifolds to be available

* A [retract!](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)ion; it is recommended to set the [`default_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* if you do not provide an initial regularization parameter `σ`, a [`manifold_dimension`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}) is required.
* By default the tangent vector storing the gradient is initialized calling [`zero_vector`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.zero_vector-Tuple{AbstractManifold,%20Any})`(M,p)`.
* [`inner`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.inner-Tuple{AbstractManifold,%20Any,%20Any,%20Any})`(M, p, X, Y)` is used within the algorithm step

Furthermore, within the Lanczos subsolver, generating a random vector (at `p`) using [`rand!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#Base.rand-Tuple{AbstractManifold})`(M, X; vector_at=p)` in place of `X` is required

## Literature

```@bibliography
Pages = ["adaptive-regularization-with-cubics.md"]
Canonical=false
```