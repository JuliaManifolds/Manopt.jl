# A manifold objective

```@meta
CurrentModule = Manopt
```

The Objective describes that actual cost function and all its properties.

```@docs
AbstractManifoldObjective
AbstractDecoratedManifoldObjective
```

Which has two main different possibilities for its containing functions concerning the evaluation mode, not necessarily the cost, but for example gradient in an [`AbstractManifoldFirstOrderObjective`](@ref).

```@docs
AbstractEvaluationType
AllocatingEvaluation
AllocatingInplaceEvaluation
InplaceEvaluation
ParentEvaluationType
evaluation_type
```

## Decorators for objectives

An objective can be decorated using the following trait and function to initialize

```@docs
dispatch_objective_decorator
is_objective_decorator
decorate_objective!
```

### [Embedded objectives](@id subsection-embedded-objectives)

```@docs
EmbeddedManifoldObjective
```

### [Scaled objectives](@id subsection-scaled-objectives)

```@docs
ScaledManifoldObjective
```

### [Cache objective](@id subsection-cache-objective)

Since single function calls, for example to the cost or the gradient, might be expensive,
a simple cache objective exists as a decorator, that caches one cost value or gradient.

It can be activated/used with the `cache=` keyword argument available for every solver.

```@docs
Manopt.reset_counters!
Manopt.objective_cache_factory
```

#### A simple cache

A first generic cache is always available, but it only caches one gradient and one cost function evaluation (for the same point).

```@docs
SimpleManifoldCachedObjective
```

#### A generic cache

For the more advanced cache, you need to implement some type of cache yourself, that provides a `get!`
and implement [`init_caches`](@ref).
This is for example provided if you load [`LRUCache.jl`](https://github.com/JuliaCollections/LRUCache.jl). Then you obtain

```@docs
ManifoldCachedObjective
init_caches
```

### [Count objective](@id subsection-count-objective)

```@docs
ManifoldCountObjective
```

#### Internal decorators and functions

```@docs
ReturnManifoldObjective
```

## Specific Objective typed and their access functions

### Cost objective

```@docs
AbstractManifoldCostObjective
ManifoldCostObjective
```

#### Access functions

```@docs
get_cost
```

and internally

```@docs
get_cost_function
```

### First order objectives

```@docs
AbstractManifoldFirstOrderObjective
ManifoldFirstOrderObjective
ManifoldAlternatingGradientObjective
ManifoldStochasticGradientObjective
NonlinearLeastSquaresObjective
```

While the [`ManifoldFirstOrderObjective`](@ref) allows to provide different
first order information, there are also its shortcuts, mainly for historical reasons,
but also since these are the most commonly used ones.

```@docs
ManifoldGradientObjective
ManifoldCostGradientObjective
```

#### Access functions

```@docs
get_gradient
get_gradients
get_differential
get_residuals
get_residuals!
```

and internally

```@docs
get_differential_function
get_gradient_function
```

### Subgradient objective

```@docs
ManifoldSubgradientObjective
```

#### Access functions

```@docs
get_subgradient
```


and internally

```@docs
get_subgradient_function
```

### Proximal map objective

```@docs
ManifoldProximalMapObjective
```

#### Access functions

```@docs
get_proximal_map
```

### Hessian objective

```@docs
AbstractManifoldHessianObjective
ManifoldHessianObjective
```

#### Access functions

```@docs
get_hessian
get_preconditioner
```

and internally

```@docs
get_hessian_function
```

### Primal-dual based objectives

```@docs
AbstractPrimalDualManifoldObjective
PrimalDualManifoldObjective
PrimalDualManifoldSemismoothNewtonObjective
```

#### Access functions

```@docs
adjoint_linearized_operator
forward_operator
get_differential_dual_prox
get_differential_primal_prox
get_dual_prox
get_primal_prox
linearized_forward_operator
```

### Constrained objective

```@docs
ConstrainedManifoldObjective
ManifoldConstrainedSetObjective
```

It might be beneficial to use the adapted problem to specify different ranges for the gradients of the constraints

```@docs
ConstrainedManoptProblem
```

as well as the helper functions

```@docs
AbstractConstrainedFunctor
AbstractConstrainedSlackFunctor
LagrangianCost
LagrangianGradient
LagrangianHessian
```

#### Access functions

```@docs
equality_constraints_length
inequality_constraints_length
get_equality_constraint
get_grad_equality_constraint
get_grad_inequality_constraint
get_hess_equality_constraint
get_hess_inequality_constraint
get_inequality_constraint
get_projected_point
get_projected_point!
get_unconstrained_objective
is_feasible
```

#### Internal functions

```@docs
Manopt.get_feasibility_status
```

### Vectorial objectives

```@docs
Manopt.AbstractVectorFunction
Manopt.AbstractVectorGradientFunction
Manopt.VectorGradientFunction
Manopt.VectorHessianFunction
```


```@docs
Manopt.AbstractVectorialType
Manopt.CoordinateVectorialType
Manopt.ComponentVectorialType
Manopt.FunctionVectorialType
```

#### Access functions

```@docs
Manopt.get_jacobian
Manopt.get_jacobian!
Manopt.get_value
Manopt.get_value_function
Base.length(::VectorGradientFunction)
```

#### Internal functions

```@docs
Manopt._to_iterable_indices
Manopt._change_basis!
Manopt.get_basis
Manopt.get_range
```

### Subproblem objective

This objective can be use when the objective of a sub problem
solver still needs access to the (outer/main) objective.

```@docs
AbstractManifoldSubObjective
```

#### Access functions

```@docs
Manopt.get_objective_cost
Manopt.get_objective_gradient
Manopt.get_objective_hessian
Manopt.get_objective_preconditioner
```

### Proximal Gradient Objective

```@docs
ManifoldProximalGradientObjective
```