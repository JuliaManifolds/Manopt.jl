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
Manopt.AbstractLinearSurrogateObjective
Manopt.NormalEquationsObjective
Manopt.LevenbergMarquardtLinearSurrogateObjective
Manopt.LevenbergMarquardtLinearSurrogateCoordinatesObjective
Manopt.CoordinatesNormalSystemState
```

## Internals

```@docs
Manopt.get_LevenbergMarquardt_scaling
Manopt.get_linear_operator!
Manopt.residuals_count
```

## [Technical details](@id sec-lm-technical-details)

The [`LevenbergMarquardt`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* the [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`) as well, to stop when the norm of the gradient is small, but if you implemented `inner`, the norm is provided already.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.


### TEMP

This is a temporary area before sorting them correctly to first get the docs to rendfer

```@docs
Manopt.default_lm_lin_solve!
Manopt.BlockNonzeroVector
Manopt.BlockNonzeroMatrix
Manopt.add_normal_vector_field_coord!
Manopt.add_normal_vector_field!
Manopt.ZeroTangentVector
Manopt.add_normal_linear_operator!
Manopt.add_normal_linear_operator_coord!
```

## Literature

```@bibliography
Pages = ["LevenbergMarquardt.md"]
Canonical=false
```
