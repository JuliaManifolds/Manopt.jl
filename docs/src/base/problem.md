# The Manopt Problem

An [`AbstractManoptProblem`](@ref) contains
the manifold (domain) a problem is defined on and the objective
that is to be minimized on that manifold. It can contain further elements, when this is necessary to phrase the problem.

## Abstract problem

```@autodocs
Modules = [Manopt]
Pages = ["base/problem/abstract_problem.jl"]
Order = [:type]
```

### Access functions

```@autodocs
Modules = [Manopt]
Pages = ["base/problem/abstract_problem.jl"]
Order = [:function]
```

From the two ingredients here, you can find more information about
* the [`ManifoldsBase.AbstractManifold`](@extref) in [ManifoldsBase.jl](@extref ManifoldsBase :doc:`index`)
* the [`AbstractManifoldObjective`](@ref) on the [objective](objective.md)