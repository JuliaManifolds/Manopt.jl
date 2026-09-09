# Modeling functions in Manopt.jl

```@meta
CurrentModule = Manopt
```

An [objective](objective.md) of an optimization [problem](problem.md) may contain different
functions related to the objective. In the simplest case these are a cost function ``f(p)`` and its (Riemannian) gradient
``\operatorname{grad} f(p)``, which returns the tangent vector of the steepest ascent direction of
a differentiable function ``f``. Any function that returns points on a manifold, (tangent) vectors or matrices
is internally assumed to work in-place. For the gradient this for example means the function
is of the form `grad_f!(M, X, p)`, where the gradient is computed in-place of `X`.
The signature follows the scheme in [`ManifoldsBase.jl`](@extref ManifoldsBase :doc:`index`),
that the [`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`) `M` stays first,
the return value is second and then the arguments follow.

## A wrapper to guarantee in-place evaluations

A user might instead also implement a function `grad_f(M, p) -> X`. This is then
internally “wrapped” by an [`InplaceManifoldFunction`](@ref) and can be specified for any
[`AbstractManifoldObjective`](@ref) or [high-level interface](high-level-interface.md)
with the `evaluation=` keyword that accepts an [`AbstractEvaluationType`](@ref)
and for the example here, one specifies it as [`AllocatingEvaluation`](@ref)`()`.
Internally, this wrapping is performed by [`maybe_wrap_function`](@ref).

```@docs
AbstractEvaluationType
AllocatingEvaluation
InplaceEvaluation
```

## A wrapper to guarantee mutating variables

A few manifolds like the [`Circle`](@extref `Manifolds.Circle`)`()` or [`PositiveNumbers`](@extref `Manifolds.PositiveNumbers`)`()`
might work on real numbers, which are not mutable. Internally, `Manopt.jl` assumes that its variables,
for example points and tangent vectors, are mutable. Therefore, variables in the [high-level interfaces](high-level-interface.md)
are automatically wrapped internally. Similarly, functions can be wrapped in a [`MutableManifoldFunction`](@ref).

Both the [`AbstractManifoldObjective`](@ref) and [high-level interfaces](high-level-interface.md)
can determine this when being passed a `p=` keyword argument providing the point used to define
functions on the manifold.
Internally, this wrapping is performed by [`maybe_wrap_variable`](@ref) and undone again by [`maybe_unwrap_variable`](@ref).

## Abstract function types

```@autodocs
Modules = [Manopt]
Pages = ["base/function/abstract_function.jl"]
Order = [:type, :function]
Private = true
Public = true
```

## [Functions modeling constraints](@id sec-constrained-function)

Functions modeling constraints can be defined with the following interface.

### Types and functions

```@autodocs
Modules = [Manopt]
Pages = ["base/function/constrained.jl"]
Order = [:type, :function]
Private = true
Public = false
```

## [Robustifier](@id sec-robustifier)

A robustifier is a smoothing technique for nonsmooth objectives.
Here it is applied with the goal of approximating the square root in a smooth way.
For the concrete functions available see the [common robustifiers](../commons/robustifiers.md).

### Types

```@autodocs
Modules = [Manopt]
Pages = ["base/function/robustifier.jl"]
Order = [:type, :function]
Private = false
Public = true
```

## [Functions that map into vector spaces](@id sec-vector-function)

For functions on manifolds that map into a vector space, this section defines
an interface to define both the functions and their derivative information.
Since the derivative information is given in tangent spaces, several different representations
are available.

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
