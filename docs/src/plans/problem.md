# [A Manopt problem](@id sec-problem)

```@meta
CurrentModule = Manopt
```

A problem describes all static data of an optimisation task and has as a super type

```@docs
AbstractManoptProblem
get_objective
get_manifold
```

Usually, such a problem is determined by the manifold or domain of the optimisation and the objective with all its properties used within an algorithm, see [The Objective](objective.md). For that one can just use

```@docs
DefaultManoptProblem
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

From the two ingredients here, you can find more information about
* the [`ManifoldsBase.AbstractManifold`](@extref) in [ManifoldsBase.jl](@extref ManifoldsBase :doc:`index`)
* the [`AbstractManifoldObjective`](@ref) on the [page about the objective](objective.md).