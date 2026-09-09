# Manopt.jl Developer guide

```@meta
CurrentModule = Manopt
```

This section of the documentation provides an overview of the design concepts behind
`Manopt.jl` and an introduction to its main data types and their relation.

The goal is to provide a detailed description for developers of aspects within `Manopt.jl`
and to convey the design decisions behind the overall structure of `Manopt.jl`.

There are two main ingredients of `Manopt.jl`: The [problem](problem.md) and the [solver state](state.md).
The problem represents the task to be solved, which by default includes the manifold
an objective is defined on and the objective to minimize.
The solver's state represents all variables and parameters a solver requires for setup as
well as during the iterations.


## Pretty printing on REPL

On the [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/) `Manopt.jl` aims to provide detailed information about a solver run
when the user activates such feedback, i.e. when setting `return_state = true` such that
a [high-level interface](high-level-interface.md) returns the whole solver state instead of
(just) the final iterate reached.

```@autodocs
Modules = [Manopt]
Pages = ["base/repl.jl"]
Order = [:type, :function]
Public = true
Private = true
```

## Parameter

Within `Manopt.jl` a parameter is a value within a structure that can be accessed or set from outside. Since the overall design model is modular, [`get_parameter`](@ref) and [`set_parameter!`](@ref) allow specifying a certain “path” into a structure to get or set something.

For example the gradient of an [objective](objective.md) function within a [problem](problem.md), like the [`AugmentedLagrangianGrad`](@ref) used within the sub problem of the [`augmented_Lagrangian_method`](@ref), has certain parameters.
The parameter functions allow generically addressing such objects without having to care about
decorators or in which field exactly the parameter is stored.
This can for example also be used in connection with [`DebugWhenActive`](@ref) to deactivate debug output under certain circumstances.

While the functions can be called with symbols to specify the position of a parameter,
internally, and more efficiently, `Val(:Symbol)`s are used.

Without a structure upfront, starting just with a symbol, properties of `Manopt.jl` itself can be set.

```@autodocs
Modules = [Manopt]
Pages = ["base/parameter.jl"]
Order = [:type, :function]
Public = true
Private = true
```
