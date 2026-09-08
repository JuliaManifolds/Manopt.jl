# Commons

```@meta
CurrentModule = Manopt
```

The commons section of the documentation collects common elements used by more than one solver.

## Overview

* [debug actions](debugs.md) that can be used in any solver, including the
  [`DebugActionFactory`](@ref) that turns the input array to a `debug=` keyword of a solver into
  the corresponding concrete debugs, especially when passing symbols like `:Cost`.
* [functions](functions.md) that are shared by several solvers.
* [objectives](objectives.md), including the decorators that cache, count or embed an objective.
* [problems](problems.md) that carry more than a manifold and an objective.
* [record actions](records.md) that can be used in any solver, including the [`RecordActionFactory`](@ref) that turns the input of a `record=` keyword into the corresponding concrete records.
* [robustifiers](robustifiers.md) used in [`LevenbergMarquardt`](@ref) to approximate nonsmooth nonlinear least squares.
* [states](states.md) of the sub solvers.
* [step sizes](stepsizes.md) that can be used with different solvers in the `stepsize=` keyword.
* [stopping criteria](stopping_criteria.md) that can be used with different solvers in the `stopping_criterion=` keyword.
* [vector functions](vectorial_functions.md) to model, for example, constraints.

## Passing parameters

Since the overall design of `Manopt.jl` is modular, one way to set parameters, for example in the objective of a sub problem,
is via [`set_parameter!`](@ref).

```@autodocs
Modules = [Manopt]
Pages = ["commons/parameters.jl"]
Order = [:type, :function]
Private = true
Public = false
```