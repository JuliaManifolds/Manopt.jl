# [A Manopt Problem](@id ProblemSection)

```@meta
CurrentModule = Manopt
```

A problem describes all static data of an optimisation task and has as a super type

```@docs
AbstractManoptProblem
get_objective
get_manifold
```

Usually, such a problem is determined by the manifold or domain of the optimisation and the objective with all its properties used within an algorithm – see [The Objective](@ref ObjectiveSection). For that we can just use

```@docs
DefaultManoptProblem
```

The exception to these are the primal dual-based solvers ([Chambolle-Pock](@ref ChambollePockSolver) and the [PD Semismooth Newton](@ref PDRSSNSolver)]), which both need two manifolds as their domain(s), hence there also exists a

```@docs
TwoManifoldProblem
```

From the two ingredients here, you can find more information about
* the [`AbstractManifold`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html) in [ManifoldsBase.jl](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/)
* the [`AbstractManifoldObjective`](@ref) on the [page about the objective](@ref ObjectiveSection).