
```@meta
CurrentModule = Manopt
```

# The solver interface functions

A solver is the combination of a [problem](problem.md) providing usually at least the [manifold](@extref `ManifoldsBase.AbstractManifold`) and the [objective](objective.md) and a [state](state.md).

Given these two, the function to call the function

```@docs
solve!(p::AbstractManoptProblem, s::AbstractManoptSolverState)
```

which is a framework that you in general should not change or redefine.
It uses the following methods, which also need to be implemented on your own
algorithm, if you want to provide one.

```@docs
initialize_solver!
step_solver!
get_solver_result
get_solver_return
stop_solver!(p::AbstractManoptProblem, s::AbstractManoptSolverState, Any)
```
