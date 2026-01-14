# Levenberg-Marquardt

```@meta
CurrentModule = Manopt
```

```@docs
LevenbergMarquardt
LevenbergMarquardt!
```

## Options

```@docs
LevenbergMarquardtState
```

## Sub-problem

```@docs
LevenbergMarquardtSurrogatePenaltyObjective
LevenbergMarquardtSurrogateConstrainedObjective
```

## [Technical details](@id sec-lm-technical-details)

The [`LevenbergMarquardt`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* the [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`) as well, to stop when the norm of the gradient is small, but if you implemented `inner`, the norm is provided already.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.

## Internals

```@docs
Manopt.default_lm_lin_solve!
```

## Literature

```@bibliography
Pages = ["LevenbergMarquardt.md"]
Canonical=false
```
