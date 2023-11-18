
# [Conjugate gradient descent](@id CGSolver)

```@meta
CurrentModule = Manopt
```

```@docs
conjugate_gradient_descent
conjugate_gradient_descent!
```

## State

```@docs
ConjugateGradientDescentState
```

## [Available coefficients](@id cg-coeffs)

The update rules act as [`DirectionUpdateRule`](@ref), which internally always first evaluate the gradient itself.

```@docs
ConjugateGradientBealeRestart
ConjugateDescentCoefficient
DaiYuanCoefficient
FletcherReevesCoefficient
HagerZhangCoefficient
HestenesStiefelCoefficient
LiuStoreyCoefficient
PolakRibiereCoefficient
SteepestDirectionUpdateRule
```

## [Technical details](@id sec-cgd-technical-details)

The [`conjugate_gradient_descent`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* A [`vector_transport_to!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/vector_transports/)`M, Y, p, X, q)`; it is recommended to set the [`default_vector_transport_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/vector_transports/#ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `vector_transport_method=` or `vector_transport_method_dual=` (for ``\mathcal N``) does not have to be specified.
* By default gradient descent uses [`ArmijoLinesearch`](@ref) which requires [`max_stepsize`](@ref)`(M)` to be set and an implementation of [`inner`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.inner-Tuple%7BAbstractManifold,%20Any,%20Any,%20Any%7D)`(M, p, X)`.
* By default the stopping criterion uses the [`norm`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#LinearAlgebra.norm-Tuple{AbstractManifold,%20Any,%20Any}) as well, to stop when the norm of the gradient is small, but if you implemented `inner` from the last point, the norm is provided already.
* By default the tangent vector storing the gradient is initialized calling [`zero_vector`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.zero_vector-Tuple{AbstractManifold,%20Any})`(M,p)`.

# Literature

```@bibliography
Pages = ["conjugate_gradient_descent.md"]
Canonical=false
```
