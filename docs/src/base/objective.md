```@meta
CurrentModule = Manopt
```

# The Objective

Within the optimization problem

```math
\operatorname*{argmin}_{p \in \mathcal M} f(p)
```

the objective describes the cost ``f(p)`` and its properties relations.
The general abstract type for these is

```@docs
AbstractManifoldObjective
```

There is a hierarchy of objectives in order to provide default implementations for certain parts.

## [Decorated objectives](@id meta-objectives)

Following the [decorator pattern](https://en.wikipedia.org/wiki/Decorator_pattern) approach,
an objective can be “wrapped” to gain certain properties. For example to [cache](@ref SimpleCacheObjective) or [count](@ref CountObjective) function evaluation.

```@docs
AbstractDecoratedManifoldObjective
ReturnManifoldObjective
```

### Functions

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/abstract_objective.jl"]
Order = [:function]
Private = false
Public = true
```

### Internal functions

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/abstract_objective.jl"]
Order = [:function]
Private = true
Public = false
```


## [A zeroth order objective](@id zeroth-order-objectives)

For the first and simples objective, only the cost function itself is available.
This is for example used in solvers like [`NelderMead`](@ref) or [`particle_swarm`](@ref)

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/cost.jl"]
Order = [:type, :function]
Private = false
Public = true
```

### Internal structures and functions

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/cost.jl"]
Order = [:type, :function]
Private = true
Public = false
```

## [A first order objective](@id first-order-objectives)

TODO

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/first_order.jl"]
Order = [:type, :function]
Private = false
Public = true
```

### Internal structures and functions

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/first_order.jl"]
Order = [:type, :function]
Private = true
Public = false
```

## [A first order nonsmooth objective](@id zeroth-order-objectives)

TODO

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/first_order_nonsmooth.jl"]
Order = [:type, :function]
Private = false
Public = true
```

### Internal structures and functions

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/first_order_nonsmooth.jl"]
Order = [:type, :function]
Private = true
Public = false
```

* [Second Order Objectives](@ref second_order_objectives) for objectives that provide second order information such as Hessians
* [Constrained Objectives](@ref constrained_objectives) for objectives that provide constraint information
* [Splitting-based Objectives](@ref splitting_based_objectives) for objectives that provide primal-dual or similar splitting based information
* [Objectives for Linear Systems](@ref objectives_for_linear_models) for objectives that provide linear systems usually in tangent spaces
* [Subproblem Objectives](@ref subproblem_objectives) for objectives that are used in subproblems and need access to the main objective
* [Vectorial Objectives](@ref vectorial_objectives) for objectives that provide vector valued
* [Decorators for Objectives](@ref decorated-objectives) for objectives that decorate other objectives, e.g., to provide caching or scaling
