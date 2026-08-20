# Stepsize

```@meta
CurrentModule = Manopt
```

In many algorithms, once a direction (in the form of a tangent vector) has been determined,
like the steepest descent direction in [`gradient_descent`](@ref), it is common to perform a stepsize computation.
A special case is a line search method; both a stepsize computation and a line search start from an initial guess.

A stepsize is a function, usually implemented as a `struct` that can be called like a function, that based on the parameters `(problem, state, k, η; kwargs...)` computes a new stepsize,
where `k` is the current iteration and `η` the search direction.
A common keyword argument is `initial_guess=`.

Step sizes often have parameters that might depend on the manifold used and therefore often use the [default factory](default_factory.md) pattern.

```@docs
Stepsize
Linesearch
AbstractInitialLinesearchGuess
```

## Functions

```@autodocs
Modules = [Manopt]
Pages = ["base/stepsize.jl"]
Order = [:function]
Private = false
Public = true
```

## Internal functions

```@autodocs
Modules = [Manopt]
Pages = ["base/stepsize.jl"]
Order = [:function]
Private = true
Public = false
```
