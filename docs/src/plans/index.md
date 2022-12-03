# [Plans for solvers](@id planSection)

```@meta
CurrentModule = Manopt
```

In order to start a solver, both a [AbstractManoptProblem](@ref AbstractManoptProblemSection) and [AbstractManoptSolverState](@ref AbstractManoptSolverStateSection) are required.
Together they form a __plan__.
Everything related to problems, options, and their tools in general, is explained in this
section and its subpages. The specific `AbstractManoptSolverState` related to a certain (concrete) solver can be
found on the specific solver page, see [The solvers overview](@ref SolversSection).