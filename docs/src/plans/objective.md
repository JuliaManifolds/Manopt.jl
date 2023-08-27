# [A Manifold Objective](@id ObjectiveSection)

```@meta
CurrentModule = Manopt
```

The Objective describes that actual cost function and all its properties.

```@docs
AbstractManifoldObjective
AbstractDecoratedManifoldObjective
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

## Decorators for Objectives

An objective can be decorated using the following trait and function to initialize

```@docs
dispatch_objective_decorator
is_objective_decorator
decorate_objective!
```

### [Embedded Objectives](@id ManifoldEmbeddedObjective)

```@autodocs
Modules = [Manopt]
Pages = ["plans/embedded_objective.jl"]
Order = [:type]
```

#### Available functions

```@autodocs
Modules = [Manopt]
Pages = ["plans/embedded_objective.jl"]
Order = [:function]
```

### [Cache Objective](@id CacheSection)

Since single function calls, e.g. to the cost or the gradient, might be expensive,
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

#### A Generic Cache

For the more advanced cache, you need to implement some type of cache yourself, that provides a `get!`
and implement [`init_caches`](@ref).
This is for example provided if you load [`LRUCache.jl`](https://github.com/JuliaCollections/LRUCache.jl). Then you obtain

```@docs
ManifoldCachedObjective
init_caches
```

### [Count Objective](@id ManifoldCountObjective)

```@docs
ManifoldCountObjective
```

### Internal Decorators

```@docs
ReturnManifoldObjective
```