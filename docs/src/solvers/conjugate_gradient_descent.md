
# Conjugate gradient descent

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
ConjugateDescentCoefficient
ConjugateGradientBealeRestart
DaiYuanCoefficient
FletcherReevesCoefficient
HagerZhangCoefficient
HestenesStiefelCoefficient
HybridCoefficient
LiuStoreyCoefficient
PolakRibiereCoefficient
SteepestDescentCoefficient
```

## [Restart rules](@id cg-restart)

The update rules might produce update steps that are not a descent direction, or at least
be only approximately one. In these cases the following restart rules can be specified.

```@docs
AbstractRestartCondition
NeverRestart
RestartOnNonDescent
RestartOnNonSufficientDescent
```

## Internal rules for coefficients

```@docs
Manopt.ConjugateGradientBealeRestartRule
Manopt.DaiYuanCoefficientRule
Manopt.FletcherReevesCoefficientRule
Manopt.HagerZhangCoefficientRule
Manopt.HestenesStiefelCoefficientRule
Manopt.HybridCoefficientRule
Manopt.LiuStoreyCoefficientRule
Manopt.PolakRibiereCoefficientRule
Manopt.SteepestDescentCoefficientRule
```

## [Technical details](@id sec-cgd-technical-details)

The [`conjugate_gradient_descent`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* A [`vector_transport_to!`](@extref ManifoldsBase :doc:`vector_transports`)`M, Y, p, X, q)`; it is recommended to set the [`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `vector_transport_method=` or `vector_transport_method_dual=` (for ``\mathcal N``) does not have to be specified.
* By default gradient descent uses [`ArmijoLinesearch`](@ref) which requires [`max_stepsize`](@ref)`(M)` to be set and an implementation of [`inner`](@extref `ManifoldsBase.inner-Tuple{AbstractManifold, Any, Any, Any}`)`(M, p, X)`.
* By default the stopping criterion uses the [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`) as well, to stop when the norm of the gradient is small, but if you implemented `inner`, the norm is provided already.
* By default the tangent vector storing the gradient is initialized calling [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.

# Literature

```@bibliography
Pages = ["conjugate_gradient_descent.md"]
Canonical=false
```
