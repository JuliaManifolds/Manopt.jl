# Manopt Solver Callbacks

```@meta
CurrentModule = Manopt
```

The callback functionality is meant to provide a user with
direct access to the solver (problem and state) at certain points
of a solver run to inspect or modify it.

The hooks `:BeforeInit`, `:Init`, `:BeforeStep`, `:Step` and `:Stop` receive the possibly decorated state, since they are in the generic [`solve!`](@ref) implementation.
Solver specific ones receive the corresponding solver state.

## Functions

```@autodocs
Modules = [Manopt]
Pages = ["base/state/callback.jl"]
Order = [:function]
Private = false
Public = true
```

## Internal functions

```@autodocs
Modules = [Manopt]
Pages = ["base/state/callback.jl"]
Order = [:function]
Private = true
Public = false
```
