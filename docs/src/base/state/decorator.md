# The State Decorator Pattern

```@meta
CurrentModule = Manopt
```

Features can be added to a solver by decorating it.
Decorators wrap a solver state to perform additional operations
in the initialization step, before or after the iteration step or when the solver stops.
The advantage of the decorator pattern here is that those additional operations can be implemented generically and hence used with any existing or new solver.

```@autodocs
Modules = [Manopt]
Pages = ["base/state/decorator.jl"]
Order = [:type, :function]
```