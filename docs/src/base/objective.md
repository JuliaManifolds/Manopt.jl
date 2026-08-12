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
an objective can be “wrapped” to gain certain properties. For example to [cache](@ref `ManifoldCachedObjective`) or [count](@ref `ManifoldCountObjective`) function evaluation.

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

## [A first order nonsmooth objective](@id first-order-nonsmooth-objectives)

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

## [A second order objective](@id second-order-objectives)

TODO

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/second_order.jl"]
Order = [:type, :function]
Private = false
Public = true
```

### Internal structures and functions

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/second_order.jl"]
Order = [:type, :function]
Private = true
Public = false
```

## [Linear systems in tangent spaces](@id linear-systems-objective)

TODO

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/linear_system.jl"]
Order = [:type, :function]
Private = false
Public = true
```

### Internal structures and functions

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/linear_system.jl"]
Order = [:type, :function]
Private = true
Public = false
```

## [Subsolver objectives](@id second-order-objectives)

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/sub_objective.jl"]
Order = [:type, :function]
Private = false
Public = true
```

### Internal structures and functions

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/sub_objective.jl"]
Order = [:type, :function]
Private = true
Public = false
```