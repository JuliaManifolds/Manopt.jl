# [AbstractManoptSolverState](@id AbstractManoptSolverStateSection)

```@meta
CurrentModule = Manopt
```

Given an [`AbstractManoptProblem`](@ref), that is a certain optimisation task,
the state specifies the solver to use. It contains the parameters of a solver and all
fields necessary during the algorithm, e.g. the current iterate, a [`StoppingCriterion`](@ref)
or a [`Stepsize`](@ref).

```@docs
AbstractManoptSolverState
get_state
```

Since the [`AbstractManoptSolverState`](@ref) directly relate to a solver,
the concrete states are documented together wirth the corresponding [solvers](@ref SolversSection).
This page documents the general functionality available for every state.

A first example is to access, i.e. obtain or set, the current iterate.
This might be useful to continue investigation at the current iterate, or to set up a solver for a next experiment, respectively.

```@docs
get_iterate
set_iterate!
```

## Decorators for AbstractManoptSolverState

A solver state can be decorated using the following trait and function to initialize

```@docs
dispatch_state_decorator
is_state_decorator
decorate_state
```

In general decorators often perform actions so we introduce

```@docs
AbstractStateAction
```

as well as a helper for storing values using keys, i.e.

```@docs
get_storage
has_storage
update_storage!
```

A simple example is the

```@docs
ReturnSolverState
```

as well as [`DebugSolverState`](@ref) and [`RecordSolverState`](@ref).