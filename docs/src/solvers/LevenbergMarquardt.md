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

A main ingredient of the Levenberg-Marquardt solver is the linear surrogate that is generated in every iteration and then solved, for
example by considering the linear system of its optimality conditions.
The following cases are available.

```@docs
Manopt.AbstractLinearSurrogateObjective
Manopt.AbstractLevenbergMarquardtLinearSurrogateObjective
Manopt.LevenbergMarquardtLinearSurrogateObjective
Manopt.LevenbergMarquardtLinearSurrogateCoordinatesObjective
Manopt.NormalEquationsObjective
Manopt.CoordinatesNormalSystemState
```

Within these especially the scaling parameter ``α`` is important.
Its computation and numerical stability aspects are documented as follows.

```@docs
Manopt.get_LevenbergMarquardt_scaling
```

## Solver Internals

### Internal functions

Internally within the sub solvers both a linear operator, sometimes as a full matrix, and a vector as right hand side of a linear system have to be constructucted. The following functions accompany this.

```@docs
Manopt.default_lm_lin_solve!
Manopt.add_normal_vector_field!
Manopt.add_normal_linear_operator!
Manopt.add_linear_operator_coord!
```

### Internal structures

In several places, especially when the Jacobian matrices or tangent vectors involved are spares, the following structures help to avoid allocating unnecessary zero matrices or vectors.

```@docs
Manopt.BlockNonzeroVector
Manopt.BlockNonzeroMatrix
Manopt.ZeroTangentVector
```

## [Technical details](@id sec-lm-technical-details)

The [`LevenbergMarquardt`](@ref) solver requires the following functions of a manifold to be available

* A [`retract!`](@extref ManifoldsBase :doc:`retractions`)`(M, q, p, X)`; it is recommended to set the [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`) to a favourite retraction. If this default is set, a `retraction_method=` does not have to be specified.
* the [`norm`](@extref `LinearAlgebra.norm-Tuple{AbstractManifold, Any, Any}`) as well, to stop when the norm of the gradient is small, but if you implemented `inner`, the norm is provided already.
* A [`copyto!`](@extref `Base.copyto!-Tuple{AbstractManifold, Any, Any}`)`(M, q, p)` and [`copy`](@extref `Base.copy-Tuple{AbstractManifold, Any}`)`(M,p)` for points.

## Literature

```@bibliography
Pages = ["LevenbergMarquardt.md"]
Canonical=false
```
