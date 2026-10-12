# [Common Record Actions](@id sec-record)

```@meta
CurrentModule = Manopt
```

Recordings during the iteration can be added to any solver run, since all solvers accept the `record=` keyword.
This is handled by the [`RecordActionFactory`](@ref).

```@autodocs
Modules = [Manopt]
Pages = ["commons/records.jl"]
Order = [:type, :function]
Public = true
Private = false
```

## Internal functions

```@autodocs
Modules = [Manopt]
Pages = ["commons/records.jl"]
Order = [:type, :function]
Public = false
Private = true
```

## Technical details

The decorator to record values during the iterations can be activated by
decorating the state of a solver and implementing
your own [`RecordAction`](@ref)s.
For more details, see [the record solver state decorator](../base/state/record.md).
