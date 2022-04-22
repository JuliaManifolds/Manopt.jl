
# [Solvers](@id SolversSection)

```@meta
CurrentModule = Manopt
```

Solvers can be applied to [`Problem`](@ref)s with solver
specific [`Options`](@ref).

# List of Algorithms

The following algorithms are currently available

| Solver  | File   | Problem & Option  |
----------|--------|-------------------|
[Alternating Gradient Descent](@ref AlternatingGradientDescentSolver) | `alterating_gradient_descent.jl` | [`AlternatingGradientProblem`](@ref), [`AlternatingGradientDescentOptions`](@ref)
[Chambolle-Pock](@ref ChambollePockSolver) | `Chambolle-Pock.jl` | [`PrimalDualProblem`](@ref), [`ChambollePockOptions`](@ref)
[Cyclic Proximal Point](@ref CPPSolver) | `cyclic_proximal_point.jl` | [`ProximalProblem`](@ref), [`CyclicProximalPointOptions`](@ref)
[Douglas–Rachford](@ref DRSolver) | `DouglasRachford.jl` | [`ProximalProblem`](@ref), [`DouglasRachfordOptions`](@ref)
[Gradient Descent](@ref GradientDescentSolver) | `gradient_descent.jl` |  [`GradientProblem`](@ref), [`GradientDescentOptions`](@ref)
[Nelder-Mead](@ref NelderMeadSolver) | `NelderMead.jl` | [`CostProblem`](@ref), [`NelderMeadOptions`](@ref)
[Particle Swarm](@ref ParticleSwarmSolver) | `particle_swarm.jl` | [`CostProblem`](@ref), [`ParticleSwarmOptions`](@ref)
[Quasi-Newton Method](@ref quasiNewton) | `quasi_newton.jl`| [`GradientProblem`](@ref), [`QuasiNewtonOptions`](@ref)
[Subgradient Method](@ref SubgradientSolver) | `subgradient_method.jl` | [`SubGradientProblem`](@ref), [`SubGradientMethodOptions`](@ref)
[Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) | `truncated_conjugate_gradient_descent.jl` | [`HessianProblem`](@ref), [`TruncatedConjugateGradientOptions`](@ref)
[The Riemannian Trust-Regions Solver](@ref trust_regions) | `trust_regions.jl` | [`HessianProblem`](@ref), [`TrustRegionsOptions`](@ref)

Note that the solvers (or their [`Options`](@ref) to be precise) can also be decorated to enhance your algorithm by general additional properties, see [Decorated Solvers](@ref DecoratedSolvers).

## [StoppingCriteria](@id StoppingCriteria)

Stopping criteria are implemented as a `functor`, i.e. inherit from the base type

```@docs
StoppingCriterion
StoppingCriterionSet
```

```@autodocs
Modules = [Manopt]
Pages = ["plans/stopping_criterion.jl"]
Order = [:type]
```

as well as the functions

```@docs
Base.:&(::StoppingCriterion, ::StoppingCriterion)
Base.:|(::StoppingCriterion, ::StoppingCriterion)
get_reason
get_stopping_criteria
get_active_stopping_criteria
are_these_stopping_critera_active
```

further stopping criteria might be available for individual Solvers.

## [Decorated Solvers](@id DecoratedSolvers)

The following decorators are available.

### [Debug Solver](@id DebugSolver)

The decorator to print debug during the iterations can be activated by
decorating the [`Options`](@ref) with [`DebugOptions`](@ref) and implementing
your own [`DebugAction`](@ref)s.
For example printing a gradient from the [`GradientDescentOptions`](@ref) is
automatically available, as explained in the [`gradient_descent`](@ref) solver.

```@autodocs
Modules = [Manopt]
Pages   = ["debug_solver.jl"]
```

### [Record Solver](@id RecordSolver)

The decorator to record certain values during the iterations can be activated by
decorating the [`Options`](@ref) with [`RecordOptions`](@ref) and implementing
your own [`RecordAction`](@ref)s.
For example recording the gradient from the [`GradientDescentOptions`](@ref) is
automatically available, as explained in the [`gradient_descent`](@ref) solver.

```@autodocs
Modules = [Manopt]
Pages   = ["record_solver.jl"]
```

## Technical Details

 The main function a solver calls is

```@docs
solve(p::Problem, o::Options)
```

which is a framework, that you in general should not change or redefine.
It uses the following methods, which also need to be implemented on your own
algorithm, if you want to provide one.

```@docs
initialize_solver!
step_solver!
get_solver_result
stop_solver!(p::Problem, o::Options, i::Int)
```
