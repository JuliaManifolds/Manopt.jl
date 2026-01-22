# [Constrained objectives](@id constrained_objectives)

```@meta
CurrentModule = Manopt
```

```@docs
ConstrainedManifoldObjective
ManifoldConstrainedSetObjective
```

It might be beneficial to use the adapted problem to specify different ranges for the gradients of the constraints

```@docs
ConstrainedManoptProblem
```

as well as the helper functions

```@docs
AbstractConstrainedFunctor
AbstractConstrainedSlackFunctor
LagrangianCost
LagrangianGradient
LagrangianHessian
```

## Access functions

```@docs
equality_constraints_length
inequality_constraints_length
get_equality_constraint
get_grad_equality_constraint
get_grad_inequality_constraint
get_hess_equality_constraint
get_hess_inequality_constraint
get_inequality_constraint
get_projected_point
get_projected_point!
get_unconstrained_objective
is_feasible
```

#### Internal functions

```@docs
Manopt.get_feasibility_status
```