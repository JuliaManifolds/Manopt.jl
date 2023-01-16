# [Plans for solvers](@id planSection)

```@meta
CurrentModule = Manopt
```

For any optimisation performed in `Manopt.jl`
we need information about both the optimisation task or “problem” at hand as well as the solver and all its parameters.
This together is called a __plan__ in `Manopt.jl` and it consists of two data structures:

* The [Manopt Problem](@ref ProblemSection) describes all _static_ data of our task, most prominently the manifold and the objective.
* The [Solver State](@ref SolverStateSection) describes all _varying_ data and parameters for the solver we aim to use. This also means that each solver has its own data structure for the state.

By splitting these two parts, we can use one problem and solve it using different solvers.

Still there might be the need to set certain parameters within any of these structures. For that there is

```@docs
set_manopt_parameter!
```