```@meta
CurrentModule = Manopt
```

# Stepsize

In many algorithms, once a direction (in form of a tangent vector) has been determined,
like the steepest descent direction in [`gradient_descent`](@ref) it is common to perform a stepsize computation.
A special case is a line search method; both start with an initial guess.

A stepsize is a function, usually implemented as a `struct` that can be called like a function, that based on the parameters `(problem, point, search_direction, gradient; kwargs...)` computes a new stepsize.
A common keyword argument is the `initial_guess = `.

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

## Internal Functions

```@autodocs
Modules = [Manopt]
Pages = ["base/stepsize.jl"]
Order = [:function]
Private = true
Public = false
```
