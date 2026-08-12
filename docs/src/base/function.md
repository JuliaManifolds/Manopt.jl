```@meta
CurrentModule = Manopt
```

# Modelling functions in Manopt.jl

An [objective](objective.md) of an optimisation [problem](problem.md) may contain different
functions related to the objective. In the simplest case a cost function ``f(p)`` and its (Riemannian) gradient
``\operatorname{grad} f(p)`` which returns the tangent vector of the steepest ascent direction of
a differentiable function ``f``. Any function returns points on a manifold, (tangent) vectors or matrices,
is internally assumed to work in-place. For the gradient this for example means the function
is of the form `grad_f!(M, X, p)`, where the gradient is computed in-place of `X`.
The signature follows the scheme in [`ManifoldsBase.jl`](@extref ManifoldsBase :doc:`index`),
that the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) `M` stays first,
the return value is second and then the arguments follow.

## A wrapper to guarantee in-place evaluations

Since a user might instead also implement a function `grad_f(M, p) -> X`. This is then
internally “wrapped” by a [`InplaceManifoldFunction`](@ref) and can be specified for any
[`AbstractManifoldObjective`](@ref) or [high-level interfaces](high-level-interface.md)
with the `evaluation = ` keyword that accepts an [`AbstractEvaluationType`](@ref)
and for the example here one specifies it as [`AllocatingEvaluation`](@ref)`()`.

```@docs
AbstractEvaluationType
AllocatingEvaluation
InplaceEvaluation
maybe_wrap_function
```

## A wrapper to guarantee mutating variables

A few Manifolds like the [`Circle`](@extref `Manifolds.Circle`)`()` or [`PositiveNumbers`](@extref `Manifolds.PositiveNumbers`)
might work on real numbers, which are not mutable. Internally, `Manopt` assumes that its variables,
e.g. points and tangent vectors, are mutable. Therefore, variables in the [high-level interfaces](high-level-interface.md)
are automatically wrapped internally. Similarly functions can be wrapped in a [`MutableManifoldFunction`](@ref).

Both the [`AbstractManifoldObjective`](@ref) and [high-level interfaces](high-level-interface.md)
can determine this when being passed a `p = ` keyword argument providing the point used to define
functions on the manifold.

```@docs
maybe_unwrap_variable
```

## Abstract function types

```@autodocs
Modules = [Manopt]
Pages = ["base/function/abstract_function.jl"]
Order = [:type, :function]
Private = true
Public = true
```

## Functions Modelling constrains

TODO short text and reference to commons

### Types and functions

```@autodocs
Modules = [Manopt]
Pages = ["base/function/constrained.jl"]
Order = [:type, :function]
Private = true
Public = false
```

### Internals

```@autodocs
Modules = [Manopt]
Pages = ["base/function/constrained.jl"]
Order = [:type, :function]
Private = false
Public = true
```

## Robustifiers

TODO short text and reference to commons

### Types

```@autodocs
Modules = [Manopt]
Pages = ["base/function/robustifier.jl"]
Order = [:type, :function]
Private = false
Public = true
```

## Functions that map into vector spaces

TODO short text and reference to commons

```@autodocs
Modules = [Manopt]
Pages = ["base/function/vectorial.jl"]
Order = [:type, :function]
Private = false
Public = true
```

### Internals

```@autodocs
Modules = [Manopt]
Pages = ["base/function/vectorial.jl"]
Order = [:type, :function]
Private = true
Public = false
```