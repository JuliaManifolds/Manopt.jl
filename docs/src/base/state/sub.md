```@meta
CurrentModule = Manopt
```

# Subsolvers

An algorithm can have a certain sub task that needs to be solved.
This is usually stored in the form of a `state.sub_problem` and a `state.sub_state` so that a specific sub solver can be specified by that pair.
The special case that the task can be solved in closed form is modelled by providing a closed form function,
usually working in-place, as the `sub_problem` and the following state.
If the closed form solution is only available in an allocating form, wrap it in an [`InplaceManifoldFunction`](@ref)
and the state is set to an [`ClosedFormSubSolverState`](@ref).
