# Exact penalty method

```@meta
CurrentModule = Manopt
```

```@docs
  exact_penalty_method
  exact_penalty_method!
```

## State

```@docs
ExactPenaltyMethodState
```

## Helping functions

```@docs
ExactPenaltyCost
ExactPenaltyGrad
SmoothingTechnique
LinearQuadraticHuber
LogarithmicSumOfExponentials
```

## [Technical details](@id sec-dr-technical-details)

The [`exact_penalty_method`](@ref) solver requires the following functions of a manifold to be available


* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.
* Everything the subsolver requires, which by default is the [`quasi_Newton`](@ref) method
* A [`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`.


The stopping criteria involves [`StopWhenChangeLess`](@ref) and [`StopWhenGradientNormLess`](@ref)
which require

* An [`inverse_retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, X, p, q)`; it is recommended to set the [`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `inverse_retraction_method=` or `inverse_retraction_method_dual=` (for ``\mathcal N``) does not have to be specified or the [`distance`](@extref `ManifoldsBase.distance-Tuple{AbstractManifold, Any, Any}`)`(M, p, q)` for said default inverse retraction.
* the [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`) as well, to stop when the norm of the gradient is small, but if you implemented `inner`, the norm is provided already.

## Literature

```@bibliography
Pages = ["exact_penalty_method.md"]
Canonical=false
```
