# [The Manopt problems](@id sec-problem)

```@meta
CurrentModule = Manopt
```

A problem is determined by the manifold or domain of the optimisation and the objective with all its properties used within an algorithm, see [The Objective](objective.md). For that one can just use

```@docs
DefaultManoptProblem
get_objective
```

For the constraint optimisation, there are different possibilities to represent the gradients
of the constraints. This can be done with a

```
ConstraintProblem
```

The primal dual-based solvers ([Chambolle-Pock](../solvers/ChambollePock.md) and the [PD Semi-smooth Newton](../solvers/primal_dual_semismooth_Newton.md)),
both need two manifolds as their domains, hence there also exists a

```@docs
TwoManifoldProblem
```