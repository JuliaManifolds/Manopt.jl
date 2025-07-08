# Difference of convex

```@meta
CurrentModule = Manopt
```

## [Difference of convex algorithm](@id solver-difference-of-convex)

```@docs
difference_of_convex_algorithm
difference_of_convex_algorithm!
```

## [Difference of convex proximal point](@id solver-difference-of-convex-proximal-point)

```@docs
difference_of_convex_proximal_point
difference_of_convex_proximal_point!
```

## Solver states

```@docs
DifferenceOfConvexState
DifferenceOfConvexProximalState
```

## The difference of convex objective

```@docs
ManifoldDifferenceOfConvexObjective
```

as well as for the corresponding sub problem

```@docs
LinearizedDCCost
LinearizedDCGrad
```

```@docs
ManifoldDifferenceOfConvexProximalObjective
```

as well as for the corresponding sub problems

```@docs
ProximalDCCost
ProximalDCGrad
```

## Helper functions

```@docs
get_subtrahend_gradient
```

## [Technical details](@id sec-cp-technical-details)

The [`difference_of_convex_algorithm`](@ref) and [`difference_of_convex_proximal_point`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` or `retraction_method_dual=` (for ``\mathcal N``) does not have to be specified.
* An [`inverse_retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `inverse_retraction_method=` or `inverse_retraction_method_dual=` (for ``\mathcal N``) does not have to be specified.

By default, one of the stopping criteria is [`StopWhenChangeLess`](@ref),
which either requires

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` or `retraction_method_dual=` (for ``\mathcal N``) does not have to be specified.
* An [`inverse_retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `inverse_retraction_method=` or `inverse_retraction_method_dual=` (for ``\mathcal N``) does not have to be specified or the [`distance`](@extref `ManifoldsBase.distance-Tuple{AbstractManifold, Any, Any}`)`(M, p, q)` for said default inverse retraction.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.
* By default the tangent vector storing the gradient is initialized calling [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.
* everything the subsolver requires, which by default is the [`trust_regions`](@ref) or if you do not provide a Hessian [`gradient_descent`](@ref).

## Literature

```@bibliography
Pages = ["difference_of_convex.md"]
Canonical=false
```