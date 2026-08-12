# Actions

An action is a callable structure, usually with the signature `(problem, state, iterate)`,
that performs something. Consider them elementary building blocks, for example a single
[debug](debug.md) output action, that can be combined into larger “things acting”.
They share a common supertype in `Manopt.jl`.

## Access functions

```@autodocs
Modules = [Manopt]
Pages = ["base/state/action.jl"]
Order = [:type, :function]
Public=true
Private=false
```

## Internal functions

```@autodocs
Modules = [Manopt]
Pages = ["base/state/action.jl"]
Order = [:type, :function]
Public=false
Private=true
```