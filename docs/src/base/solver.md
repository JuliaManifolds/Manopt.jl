
```@meta
CurrentModule = Manopt
```

# The solver interface functions

A solver is the combination of a [problem](problem.md) providing usually at least the [manifold](@extref `ManifoldsBase.AbstractManifold`) and the [objective](objective.md) and a [state](state.md).

Given these two, the function to call the function [`solve!`](@ref), which is a framework that you in general should not change or redefine. It uses the following methods, which also need to be implemented on your own
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
