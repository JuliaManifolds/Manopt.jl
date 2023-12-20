# [Debug output](@id DebugSection)

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

## Technical details

The decorator to print debug during the iterations can be activated by
decorating the state of a solver and implementing
your own [`DebugAction`](@ref)s.
For example printing a gradient from the [`GradientDescentState`](@ref) is
automatically available, as explained in the [`gradient_descent`](@ref) solver.

```@docs
initialize_solver!(amp::AbstractManoptProblem, dss::DebugSolverState)
step_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i)
stop_solver!(amp::AbstractManoptProblem, dss::DebugSolverState, i::Int)
```
