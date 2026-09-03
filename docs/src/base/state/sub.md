# [Modeling a sub solver](@id sec-subsolver)

```@meta
CurrentModule = Manopt
```

An algorithm can have a certain sub task that needs to be solved.
This is usually stored in the form of a `state.sub_problem` and a `state.sub_state` so that a specific sub solver can be specified by that pair.
The special case that the task can be solved in closed form is modeled by providing a closed form function,
usually working in-place, as the `sub_problem` together with a [`ClosedFormSubSolverState`](@ref) as the `sub_state`.
If the closed form solution is only available in an allocating form, wrap it in an [`InplaceManifoldFunction`](@ref)
and the state is set to a [`ClosedFormSubSolverState`](@ref).

Concrete functions that are based on objectives and model sub objective functions can be found
[in the commons area](@ref sec-sub-functions).

## Types and functions

```@autodocs
Modules = [Manopt]
Pages = ["base/state/sub_state.jl"]
Order = [:type, :function]
Private = false
Public = true
```

## Internals

```@autodocs
Modules = [Manopt]
Pages = ["base/state/sub_state.jl"]
Order = [:type, :function]
Private = true
Public = false
```