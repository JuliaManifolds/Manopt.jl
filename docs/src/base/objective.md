# The Objective

```@meta
CurrentModule = Manopt
```

Within the optimization problem

```math
\operatorname*{argmin}_{p \in \mathcal M} f(p)
```

the objective describes the cost ``f(p)`` and its properties and relations.
The general abstract type for these is

```@docs
AbstractManifoldObjective
```

There is a hierarchy of objectives in order to provide default implementations for certain parts.

## [Decorated objectives](@id meta-objectives)

Following the [decorator pattern](https://en.wikipedia.org/wiki/Decorator_pattern) approach,
an objective can be “wrapped” to gain certain properties, for example to [cache](@ref ManifoldCachedObjective) or [count](@ref ManifoldCountObjective) function evaluations.

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

For the first and simplest objective, only the cost function itself is available.
This is for example used in solvers like [`NelderMead`](@ref) or [`particle_swarm`](@ref).

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

A smooth first order objective usually contains the gradient.
This interface unifies the access to it.

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

First order nonsmooth objectives come in a variety of flavors, mainly splitting based, where the single summands themselves have certain properties. They are collected in the following.

```@autodocs
Modules = [Manopt]
Pages = ["base/objective/first_order_nonsmooth.jl"]
Order = [:type, :function]
Private = false
Public = true
```

## [A second order objective](@id second-order-objectives)

The following types and functions provide the access to second-order information.

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

## [A linear system in a tangent space](@id sec-linear-systems-objective)

A linear system in a tangent space can be modeled in different ways. Most prominently
either as a matrix as soon as a basis of the tangent space is fixed or as a linear operator
in a basis-free representation.

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

## [Sub solver objectives](@id subsolver-objectives)

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