# [Debug Output](@id DebugSection)

```@meta
CurrentModule = Manopt
```

Debug output can easily be added to any solver run.
On the high level interfaces, like [`gradient_descent`](@ref), you can just use the `debug=` keyword.

```@autodocs
Modules = [Manopt]
Pages = ["plans/debug.jl"]
Order = [:type, :function]
Private = true
```

## Technical Details: The Debug Solver

The decorator to print debug during the iterations can be activated by
decorating the [`AbstractManoptSolverState`](@ref) with [`DebugSolverState`](@ref) and implementing
your own [`DebugAction`](@ref)s.
For example printing a gradient from the [`GradientDescentState`](@ref) is
automatically available, as explained in the [`gradient_descent`](@ref) solver.

```@autodocs
Modules = [Manopt]
Pages   = ["debug_solver.jl"]
Private = true
```
