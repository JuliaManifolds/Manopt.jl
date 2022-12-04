# [AbstractManoptSolverState](@id AbstractManoptSolverStateSection)

```@meta
CurrentModule = Manopt
```

For most algorithms a certain set of options can either be
generated beforehand of the function with keywords can be used.
Generally the type

```@docs
AbstractManoptSolverState
get_state
```

Since the [`AbstractManoptSolverState`](@ref) directly relate to a solver, they are documented with the
corresponding [solvers](@ref SolversSection).
You can always access the options (since they
might be decorated) by calling [`get_state`](@ref).

For easier access, and to abstract where these are actually stored, there exists

```@docs
get_iterate
set_iterate!
```

## Decorators for AbstractManoptSolverState

A solver state can be decorated using the following trait and function to initialize

```@docs
dispatch_state_decorator
is_options_decorator
decorate_state
```

In general decorators often perform actions so we introduce

```@docs
AbstractSolverStateAction
```

as well as a helper for storing values using keys, i.e.

```@docs
StoreSolverStateAction
get_storage
has_storage
update_storage!
```

A simple example is the

```@docs
ReturnSolverState
```

as well as [`DebugSolverState`](@ref) and [`RecordSolverState`](@ref).