# Keyword handling

```@meta
CurrentModule = Manopt
```

For the [high-level interfaces](high-level-interface.md) `Manopt.jl` has an internal system to keep
track of accepted keywords.
The `kwargs...` are passed to at least the [solver state](state.md) and the [state decorators](state/decorator.md).

The methods on this page keep track of where keywords are passed to and issue a warning if a keyword stays “unused”.

```@autodocs
Modules = [Manopt]
Pages = ["base/keyword.jl"]
Order = [:type, :function]
Private = false
Public = true
```

## Internal structures and functions

```@autodocs
Modules = [Manopt]
Pages = ["base/keyword.jl"]
Order = [:type, :function]
Private = true
Public = false
```