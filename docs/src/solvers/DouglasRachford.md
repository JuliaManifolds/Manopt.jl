# Douglas—Rachford algorithm

The (Parallel) Douglas—Rachford ((P)DR) algorithm was generalized to Hadamard
manifolds in [BergmannPerschSteidl:2016](@cite).

The aim is to minimize the sum

```math
f(p) = g(p) + h(p)
```

on a manifold, where the two summands have proximal maps
``\operatorname{prox}_{λ g}, \operatorname{prox}_{λ h}`` that are easy
to evaluate (maybe in closed form, or not too costly to approximate).
Further, define the reflection operator at the proximal map as

```math
\operatorname{refl}_{λ g}(p) = \operatorname{retr}_{\operatorname{prox}_{λ g}(p)} \bigl( -\operatorname{retr}^{-1}_{\operatorname{prox}_{λ g}(p)} p \bigr).
```

Let ``\alpha_k ∈  [0,1]`` with ``\sum_{k ∈ ℕ} \alpha_k(1-\alpha_k) =  \infty``
and ``λ > 0`` (which might depend on iteration ``k`` as well) be given.

Then the (P)DRA algorithm for initial data ``p^{(0)} ∈ \mathcal M`` as

## Initialization

Initialize ``q^{(0)} = p^{(0)}`` and ``k=0``

## Iteration

Repeat until a convergence criterion is reached

1. Compute ``r^{(k)} = \operatorname{refl}_{λ g}\operatorname{refl}_{λ h}(q^{(k)})``
2. Within that operation, store ``p^{(k+1)} = \operatorname{prox}_{λ h}(q^{(k)})`` which is the prox the inner reflection reflects at.
3. Compute ``q^{(k+1)} = g(\alpha_k; q^{(k)}, r^{(k)})``, where ``g`` is a curve approximating the shortest geodesic, provided by a retraction and its inverse
4. Set ``k = k+1``

## Result

The result is given by the last computed ``p^{(K)}`` at the last iterate ``K``.

For the parallel version, the first proximal map is a vectorial version where
in each component one prox is applied to the corresponding copy of ``t_k`` and
the second proximal map corresponds to the indicator function of the set,
where all copies are equal (in ``\mathcal M^n``, where ``n`` is the number of copies),
leading to the second prox being the Riemannian mean.

## Interface

```@docs
  DouglasRachford
  DouglasRachford!
```

## State

```@docs
  DouglasRachfordState
```

For specific [`DebugAction`](@ref)s and [`RecordAction`](@ref)s see also
[Cyclic Proximal Point](cyclic_proximal_point.md).

Furthermore, this solver has a short hand notation for the involved [`reflect`](@ref)ion.

```@docs
reflect
```

## [Technical details](@id sec-dr-technical-details)

The [`DouglasRachford`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* An [`inverse_retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `inverse_retraction_method=` does not have to be specified.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.

By default, one of the stopping criteria is [`StopWhenChangeLess`](@ref),
which requires

* An [`inverse_retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `inverse_retraction_method=` or `inverse_retraction_method_dual=` (for ``\mathcal N``) does not have to be specified or the [`distance`](@extref `ManifoldsBase.distance-Tuple{AbstractManifold, Any, Any}`)`(M, p, q)` for said default inverse retraction.

## Literature

```@bibliography
Pages = ["DouglasRachford.md"]
```
