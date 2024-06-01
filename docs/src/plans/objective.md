# A manifold objective

```@meta
CurrentModule = Manopt
```

The Objective describes that actual cost function and all its properties.

```@docs
AbstractManifoldObjective
AbstractDecoratedManifoldObjective
```

Which has two main different possibilities for its containing functions concerning the evaluation mode, not necessarily the cost, but for example gradient in an [`AbstractManifoldGradientObjective`](@ref).

```@docs
AbstractEvaluationType
AllocatingEvaluation
InplaceEvaluation
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

### Internal decorators

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

### Gradient objectives

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

#### Access functions

```@docs
get_gradient
get_gradients
```

and internally

```@docs
get_gradient_function
```

#### Internal helpers

```@docs
get_gradient_from_Jacobian!
```

### Subgradient objective

```@docs
ManifoldSubgradientObjective
```

#### Access functions

```@docs
get_subgradient
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
```

It might be beneficial to use the adapted problem to specify different ranges for the gradients of the constraints

```@docs
ConstrainedManoptProblem
```

#### Access functions

```@docs
equality_constraints_length
inequality_constraints_length
get_unconstrained_objective
get_equality_constraint
get_inequality_constraint
get_grad_equality_constraint
get_grad_inequality_constraint
```

### A vectorial cost function

```@docs
Manopt.VectorGradientFunction
```


```@docs
Manopt.AbstractVectorialType
Manopt.CoordinateVectorialType
Manopt.ComponentVectorialType
Manopt.FunctionVectorialType
```

#### Access functions

```@docs
Manopt.get_value
Manopt.get_value_function
Base.length(::VectorGradientFunction)
```

#### Internal functions

```@docs
Manopt._to_iterable_indices
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