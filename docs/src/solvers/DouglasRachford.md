# [Douglas—Rachford algorithm](@id DRSolver)

The (Parallel) Douglas—Rachford ((P)DR) Algorithm was generalized to Hadamard
manifolds in [BergmannPerschSteidl:2016](@cite).

The aim is to minimize the sum

```math
F(p) = f(p) + g(p)
```

on a manifold, where the two summands have proximal maps
``\operatorname{prox}_{λ f}, \operatorname{prox}_{λ g}`` that are easy
to evaluate (maybe in closed form, or not too costly to approximate).
Further, define the reflection operator at the proximal map as

```math
\operatorname{refl}_{λ f}(p) = \operatorname{retr}_{\operatorname{prox}_{λ f}(p)} \bigl( -\operatorname{retr}^{-1}_{\operatorname{prox}_{λ f}(p)} p \bigr).
```

Let ``\alpha_k ∈  [0,1]`` with ``\sum_{k ∈ \mathbb N} \alpha_k(1-\alpha_k) =  \infty``
and ``λ > 0`` (which might depend on iteration ``k`` as well) be given.

Then the (P)DRA algorithm for initial data ``x_0 ∈ \mathcal H`` as

## Initialization

Initialize ``q_0 = p_0`` and ``k=0``

## Iteration

Repeat until a convergence criterion is reached

1. Compute ``s_k = \operatorname{refl}_{λ f}\operatorname{refl}_{λ g}(q_k)``
2. Within that operation, store ``p_{k+1} = \operatorname{prox}_{λ g}(t_k)`` which is the prox the inner reflection reflects at.
3. Compute ``q_{k+1} = g(\alpha_k; q_k, s)``, where ``g`` is a curve approximating the shortest geodesic, provided by a retraction and its inverse
4. Set ``k = k+1``

until a stopping criterion is met.

## Acceleration and Inertia

Before computing the first step, one can apply _inertia_: Given some ``θ_k \in (0,1)``
we can perform a step before, namely

```math
t = \operatorname{retr}_{q_k}
  \bigl( -θ_k\operatorname{retr}^{-1}_{q_k}(q_{k-1}) \bigr),
```

that is adding inertia from the last two results computed in step 3
and use `t_k` as the argument of the double-reflection in step 1.

Instead of just computing step 4, one can also add an acceleration.
Let `T` denote the double-reflection from the first step.
Then, given a number ``n`` called _acceleration_ step 3 is replaced with

```math
q_{k+1} = T^n\bigl( g(\alpha_k; q_k, s) \bigr)
```

These both methods can also be combined by specifying inertia and acceleration

## Result

The result is given by the last computed ``p_K``.

For the parallel version, the first proximal map is a vectorial version where
in each component one prox is applied to the corresponding copy of ``t_k`` and
the second proximal map corresponds to the indicator function of the set,
where all copies are equal (in ``\mathcal H^n``, where ``n`` is the number of copies),
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
[Cyclic Proximal Point](@ref CPPSolver).

Furthermore, this solver has a short hand notation for the involved [`reflect`](@ref)ion.

```@docs
reflect
```

## [Technical details](@id sec-dr-technical-details)

The [`DouglasRachford`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* An [`inverse_retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `inverse_retraction_method=` does not have to be specified.
* A [`copyto!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#Base.copyto!-Tuple{AbstractManifold,%20Any,%20Any})`(M, q, p)` and [`copy`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#Base.copy-Tuple{AbstractManifold,%20Any})`(M,p)` for points.

By default, one of the stopping criteria is [`StopWhenChangeLess`](@ref),
which requires

* An [`inverse_retract!`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/retractions/#ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}) to a favourite retraction. If this default is set, a `inverse_retraction_method=` or `inverse_retraction_method_dual=` (for ``\mathcal N``) does not have to be specified or the [`distance`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions/#ManifoldsBase.distance-Tuple{AbstractManifold,%20Any,%20Any})`(M, p, q)` for said default inverse retraction.

## Literature

```@bibliography
Pages = ["DouglasRachford.md"]
```
