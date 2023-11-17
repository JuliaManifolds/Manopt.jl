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

Initialize ``t_0 = x_0`` and ``k=0``

## Iteration

Repeat until a convergence criterion is reached

1. Compute ``s_k = \operatorname{refl}_{λ f}\operatorname{refl}_{λ g}(t_k)``
2. Within that operation, store ``p_{k+1} = \operatorname{prox}_{λ g}(t_k)`` which is the prox the inner reflection reflects at.
3. Compute ``t_{k+1} = g(\alpha_k; t_k, s_k)``, where ``g`` is a curve approximating the shortest geodesic, provided by a retraction and its inverse
4. Set ``k = k+1``

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

## Literature

```@bibliography
Pages = ["DouglasRachford.md"]
```
