# [Plans for solvers](@id planSection)

```@meta
CurrentModule = Manopt
```

For any optimisation performed in `Manopt.jl`
information is required about both the optimisation task or “problem” at hand as well as the solver and all its parameters.
This together is called a __plan__ in `Manopt.jl` and it consists of two data structures:

* The [Manopt Problem](@ref ProblemSection) describes all _static_ data of a task, most prominently the manifold and the objective.
* The [Solver State](@ref SolverStateSection) describes all _varying_ data and parameters for the solver that is used. This also means that each solver has its own data structure for the state.

By splitting these two parts, one problem can be define an then be solved  using different solvers.

Still there might be the need to set certain parameters within any of these structures. For that there is

```@docs
set_manopt_parameter!
get_manopt_parameter
Manopt.status_summary
```

Where the following Symbols are used

The following symbols are used.
The column “generic” refers to a short hand that might be used for readability if clear from context.

| Symbol       | Used in | Description                                                | generic |
| :----------- | :------: | ;-------------------------------------------------------- | :------ |
| `:active` | [`DebugWhenActive`](@ref) | activity of the debug action stored within | |
| `:Basepoint` | [`TangentSpace`]() | the point the tangent space is at           | `:p` |
| `:Cost` | generic |the cost function (within an objective, as pass down) | |
| `:Debug` | [`DebugSolverState`](@ref) | the stored `debugDictionary` | |
| `:Gradient` | generic |the gradient function (within an objective, as pass down) | |
| `:Iterate` | generic | the (current) iterate, similar to [`set_iterate!`](@ref), within a state | |
| `:Manifold` | generic |the manifold (within a problem, as pass down) | |
| `:Objective` | generic | the objective (within a problem, as pass down) | |
| `:SubProblem` | generic | the sub problem (within a state, as pass down) | |
| `:SubState` | generic | the sub state (within a state, as pass down) | |
| `:λ` | [`ProximalDCCost`](@ref), [`ProximalDCGrad`](@ref) | set the proximal parameter within the proximal sub objective elements | |
| `:p`         | generic | a certain point         | |
| `:X`         | generic | a certain tangent vector | |
| `:TrustRegionRadius` | [`TrustRegionsState`](@ref) | the trust region radius | `:σ` |
| `:ρ`, `:u` | [`ExactPenaltyCost`](@ref), [`ExactPenaltyGrad`](@ref) | Parameters within the exact penalty objective | |
| `:ρ`, `:μ`, `:λ` | [`AugmentedLagrangianCost`](@ref) and [`AugmentedLagrangianGrad`](@ref) | Parameters of the Lagrangian function | |
