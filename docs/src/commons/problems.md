# Problems

```@meta
CurrentModule = Manopt
```

A [problem](../base/problem.md) usually only carries a [manifold](@extref `ManifoldsBase.AbstractManifold`) an [objective](objective.md).
For this case one can use the [`DefaultManoptProblem`](@ref), theere are cases where more properties belong to a problem. The following ones are available in `Manopt.jl`


```@autodocs
Modules = [Manopt]
Pages = ["commons/problems.jl"]
Order = [:type, :function]
Public = true
Private = false
```

## Internal functions

```@autodocs
Modules = [Manopt]
Pages = ["commons/problems.jl"]
Order = [:type, :function]
Public = false
Private = true
```
