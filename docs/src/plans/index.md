# [Plans for solvers](@id sec-plan)

```@meta
CurrentModule = Manopt
```

For any optimisation performed in `Manopt.jl`
information is required about both the optimisation task or “problem” at hand as well as the solver and all its parameters.
This together is called a __plan__ in `Manopt.jl` and it consists of two data structures:

* The [Manopt Problem](@ref sec-problem) describes all _static_ data of a task, most prominently the manifold and the objective.
* The [Solver State](@ref sec-solver-state) describes all _varying_ data and parameters for the solver that is used. This also means that each solver has its own data structure for the state.

By splitting these two parts, one problem can be define an then be solved  using different solvers.

Still there might be the need to set certain parameters within any of these structures. For that there is

```@docs
set_parameter!
get_parameter
Manopt.status_summary
```

The following symbols are used.

| Symbol       | Used in  | Description                                                |
| :----------- | :------ | :--------------------------------------------------------- |
| `:Activity` | [`DebugWhenActive`](@ref) | activity of the debug action stored within |
| `:Basepoint` | [`TangentSpace`](@extref ManifoldsBase `ManifoldsBase.TangentSpace`) | the point the tangent space is at |
| `:Cost` | generic |the cost function (within an objective, as pass down) |
| `:Debug` | [`DebugSolverState`](@ref) | the stored `debugDictionary` |
| `:Gradient` | generic | the gradient function (within an objective, as pass down) |
| `:Iterate` | generic | the (current) iterate, similar to [`set_iterate!`](@ref), within a state |
| `:Manifold` | generic |the manifold (within a problem, as pass down) |
| `:Objective` | generic | the objective (within a problem, as pass down) |
| `:SubProblem` | generic | the sub problem (within a state, as pass down) |
| `:SubState` | generic | the sub state (within a state, as pass down) |
| `:λ` | [`ProximalDCCost`](@ref), [`ProximalDCGrad`](@ref) | set the proximal parameter within the proximal sub objective elements |
| `:Population` | [`ParticleSwarmState`](@ref) | a certain population of points, for example [`particle_swarm`](@ref)s swarm |
| `:Record` | [`RecordSolverState`](@ref) |
| `:TrustRegionRadius` | [`TrustRegionsState`](@ref) | the trust region radius, equivalent to `:σ` |
| `:ρ`, `:u` | [`ExactPenaltyCost`](@ref), [`ExactPenaltyGrad`](@ref) | Parameters within the exact penalty objective |
| `:ρ`, `:μ`, `:λ` | [`AugmentedLagrangianCost`](@ref), [`AugmentedLagrangianGrad`](@ref) | Parameters of the Lagrangian function |
| `:p`, `:X` | [`LinearizedDCCost`](@ref), [`LinearizedDCGrad`](@ref) | Parameters withing the linearized functional used for the sub problem of the [difference of convex algorithm](@ref solver-difference-of-convex) |

Any other lower case name or letter as well as single upper case letters access fields of the corresponding first argument.
for example `:p` could be used to access the field `s.p` of a state.
This is often, where the iterate is stored, so the recommended way is to use `:Iterate` from before.

Since the iterate is often stored in the states fields `s.p` one _could_ access the iterate
often also with `:p` and similarly the gradient with `:X`.
This is discouraged for both readability as well as to stay more generic, and it is recommended
to use `:Iterate` and `:Gradient` instead in generic settings.

You can further activate a “Tutorial” mode by `set_parameter!(:Mode, "Tutorial")`. Internally, the following convenience function is available.

```@docs
Manopt.is_tutorial_mode
```

## A factory for providing manifold defaults

In several cases a manifold might not yet be known at the time a (keyword) argument should be provided. Therefore, any type with a manifold default can be wrapped into a factory.

```@docs
Manopt.ManifoldDefaultsFactory
Manopt._produce_type
```

## Keyword arguments and their verification

Internally `Manopt.jl` passes keywords for the (high-level) solver
functions to several inner functions, e.g. to add debug or caching.
Besides the documentation, one can check with the internal function [`Manopt.accepted_keywords`](@ref) which keywords a solver accepts.

A solver also warns, if a keyword is passed, that is not handled by the
solver or any of the inner functions it calls.

```@docs
Manopt.Keywords
Manopt.accepted_keywords
Manopt.add!
Manopt.calls_with_kwargs
Manopt.direct_keywords
Manopt.keywords_accepted
```