# Levenberg-Marquardt

```@meta
CurrentModule = Manopt
```

```@docs
LevenbergMarquardt
LevenbergMarquardt!
```

## State

```@docs
LevenbergMarquardtState
```

## Solver internals

### Internal functions

Internally within the sub solvers both a linear operator, sometimes as a full matrix, and a vector as right hand side of a linear system have to be constructed. The following functions accompany this.

```@docs
Manopt.default_lm_lin_solve!
Manopt.add_normal_vector_field!
Manopt.add_normal_linear_operator!
Manopt.add_linear_operator_coord!
```

## [Technical details](@id sec-lm-technical-details)

The [`LevenbergMarquardt`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favorite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* The [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`), to stop when the norm of the gradient is small; if you implemented `inner`, the norm is provided already.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.

## Literature

```@bibliography
Pages = ["LevenbergMarquardt.md"]
Canonical=false
```
