# Adaptive regularization with cubics

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

There are several ways to approach the sub solver. The default is the first one.

## [Lanczos iteration](@id arc-Lanczos)

```@docs
Manopt.LanczosState
```

## (Conjugate) gradient descent

There is a generic objective, that implements the sub problem

```@docs
AdaptiveRegularizationWithCubicsModelObjective
get_cost(::TangentSpace, ::AdaptiveRegularizationWithCubicsModelObjective, X)
get_gradient(::TangentSpace,::AdaptiveRegularizationWithCubicsModelObjective, X)
```

Since the sub problem is given on the tangent space, you have to provide

```julia
arc_obj = AdaptiveRegularizationWithCubicsModelObjective(mho, σ)
sub_problem = DefaultManoptProblem(TangentSpace(M, p), arc_obj)
```

where `mho` is the Hessian objective of `f` to solve.
Then use this for the `sub_problem` keyword
and use your favorite gradient based solver for the `sub_state` keyword, for example a
[`ConjugateGradientDescentState`](@ref)

## Additional stopping criteria

```@docs
StopWhenAllLanczosVectorsUsed
StopWhenFirstOrderProgress
```

## [Technical details](@id sec-arc-technical-details)

The [`adaptive_regularization_with_cubics`](@ref) requires the following functions
of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favorite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* if you do not provide an initial regularization parameter `σ`, a [`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`) is required.
* By default the tangent vector storing the gradient is initialized calling [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.
* [`inner`](@extref `ManifoldsBase.inner-Tuple{AbstractManifold, Any, Any, Any}`)`(M, p, X, Y)` is used within the algorithm step

Furthermore, within the Lanczos sub solver, generating a random vector (at `p`) using [`rand!`](@extref Base.rand-Tuple{AbstractManifold})`(M, X; vector_at=p)` in place of `X` is required

## Literature

```@bibliography
Pages = ["adaptive_regularization_with_cubics.md"]
Canonical=false
```