# [Subproblem objectives](@id subproblem_objectives)

```@meta
CurrentModule = Manopt
```

This objective can be use when the objective of a sub problem
solver still needs access to the (outer/main) objective.

```@docs
AbstractManifoldSubObjective
```

#### Access functions

```@docs
Manopt.get_objective_cost
Manopt.get_objective_gradient
Manopt.get_objective_hessian
Manopt.get_objective_preconditioner
```
