# The solver interface functions

```@meta
CurrentModule = Manopt
```

A solver is the combination of a [problem](problem.md), usually providing at least the [manifold](@extref `ManifoldsBase.AbstractManifold`) and the [objective](objective.md), together with a [state](state.md).

Given these two, the function to call is [`solve!`](@ref), which is a framework that you in general should not change or redefine. It uses the following methods, which also need to be implemented for your own
algorithm, if you want to provide one.

```@autodocs
Modules = [Manopt]
Pages = ["base/solver.jl"]
Order = [:type, :function]
Public = true
Private = false
```

## Internal functions

```@autodocs
Modules = [Manopt]
Pages = ["base/solver.jl"]
Order = [:type, :function]
Public = false
Private = true
```
