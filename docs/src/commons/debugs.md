## [Common Debug Output](@id sec-debug)

```@meta
CurrentModule = Manopt
```

Debug output can be added to any solver run, since all solvers accept the `debug = ` keyword.
This is handles by the [`DebugActionFactory`](@ref).

```@autodocs
Modules = [Manopt]
Pages = ["commons/debugs.jl"]
Order = [:type, :function]
Public = true
Private = false
```

## Internal functions

```@autodocs
Modules = [Manopt]
Pages = ["commons/debugs.jl"]
Order = [:type, :function]
Public = false
Private = true
```

## Technical details

The decorator to print debug during the iterations can be activated by
decorating the state of a solver and implementing
your own [`DebugAction`](@ref)s.
For more details, see [the debug solver state decorator](../base/state/debug.md).
