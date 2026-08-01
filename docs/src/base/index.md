# Manopt.jl Developer guide

This section of the documentation provides an overview of the design concepts behind
`Manot.jl` and an introduction to its main data types and their relation.

The goal is to provide a detailed description for developers of new aspects within `Manopt.jl`
and to convey the design decisions behind the overall structure of `Manopt.jl`.

There are two main ingredients of `Manopt.jl`: The [problem](problem.md) and the [solver state](state.md).
The problem represents the task to be solved, which by default includes the manifold
an objective is defined on and the objective to solve.
The solver's state represents all variables and parameters a solver requires for setup as
well as during the iterations.
