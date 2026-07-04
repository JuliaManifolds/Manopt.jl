# Manopt.jl Developer guide

This section of the documentation provides an overview on the main abstract types,
their ideas and design concepts.

!!! note
    Formerly, these were part of the [Plan](../plans/index.md), which will now sever to host
    concrete types and properties several solvers have in common, while abstract types, concepts and design ideas will migrate here.

There are two main ingredients of `Manopt.jl`: The [problem](problem.md) and the [solver state](state/index.md).
The problem represents the task to be solved, which by default includes the manifold
an objective is defined on and the objective to solve.
The solver's state represents all variables and parameters a solver requires for setup as
well as during the iterations.
