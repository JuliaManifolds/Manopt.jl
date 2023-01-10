# [A Manifold Objective](@id ObjectiveSection)

```@meta
CurrentModule = Manopt
```

The Objective describes that actual cost function and all its properties.

```@docs
AbstractManifoldObjective
decorate_objective!
```

Which has two main different possibilities for its containing functions concerning the evaluation mode – not necessarily the cost, but for example gradient in an [`AbstractManifoldGradientObjective`](@ref).

```@docs
AbstractEvaluationType
AllocatingEvaluation
InplaceEvaluation
evaluation_type
```

It sometimes might be nice to set certain parameters within

## Cost Objective

```@docs
AbstractManifoldCostObjective
ManifoldCostObjective
```

### Access functions

```@docs
get_cost
get_cost_function
```

## Gradient Objectives

```@docs
AbstractManifoldGradientObjective
ManifoldGradientObjective
ManifoldAlternatingGradientObjective
ManifoldStochasticGradientObjective
NonlinearLeastSquaresObjective
```

There is also a second variant, if just one function is responsible for computing the cost _and_ the gradient

```@docs
ManifoldCostGradientObjective
```

### Access functions

```@docs
get_gradient
get_gradients
get_gradient_function
```

## Subgradient Objective

```@docs
ManifoldSubgradientObjective
```

### Access Functions

```@docs
get_subgradient
```

## Proximal Map Objective

```@docs
ManifoldProximalMapObjective
```

### Access Functions

```@docs
get_proximal_map
```


## Hessian Objective

```@docs
ManifoldHessianObjective
```

### Access functions

```@docs
get_hessian
get_preconditioner
```

## Primal-Dual based Objetives

```@docs
AbstractPrimalDualManifoldObjective
PrimalDualManifoldObjective
PrimalDualManifoldSemismoothNewtonObjective
```

### Access functions

```@docs
adjoint_linearized_operator
forward_operator
get_differential_dual_prox
get_differential_primal_prox
get_dual_prox
get_primal_prox
linearized_forward_operator
```

## Constrained Objective

Besides the [`AbstractEvaluationType`](@ref) there is one further property to
distinguish among constraint functions, especially the gradients of the constraints.

```@docs
ConstraintType
FunctionConstraint
VectorConstraint
```

The [`ConstraintType`](@ref) is a parameter of the corresponding Objective.

```@docs
ConstrainedManifoldObjective
```

### Access functions

```@docs
get_constraints
get_equality_constraint
get_equality_constraints
get_inequality_constraint
get_inequality_constraints
get_grad_equality_constraint
get_grad_equality_constraints
get_grad_equality_constraints!
get_grad_equality_constraint!
get_grad_inequality_constraint
get_grad_inequality_constraint!
get_grad_inequality_constraints
get_grad_inequality_constraints!
```

## [Cache Objective](@id CacheSection)

Since single function calls, e.g. to the cost or the gradient, might be expensive,
a simple cache objective exists as a decorator, that caches one cost value or gradient.

_This feature was just recently introduced and might still be a little instable.
The `cache::Symbol=` keyword argument of the solvers might be extended or still change slightly for example._

```@docs
SimpleCacheObjective
objective_cache_factory
```