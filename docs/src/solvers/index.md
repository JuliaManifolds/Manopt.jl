
# Solvers

```@meta
CurrentModule = Manopt
```

Solvers can be applied to [`Problem`](@ref)s with solver
specific [`Options`](@ref).

# List of Algorithms

The following algorithms are currently available

| Solver  | File   | Problem & Option  |
----------|--------|-------------------|
[steepest Descent](@ref GradientDescentSolver) | `steepest_descent.jl` |  [`GradientProblem`](@ref), [`GradientDescentOptions`](@ref)
[Cyclic Proximal Point](@ref CPPSolver) | `cyclic_proximal_point.jl` | [`ProximalProblem`](@ref), [`CyclicProximalPointOptions`](@ref)
[Douglas–Rachford](@ref DRSolver) | `DouglasRachford.jl` | [`ProximalProblem`](@ref), [`DouglasRachfordOptions`](@ref)
[Nelder-Mead](@ref NelderMeadSolver) | `NelderMead.jl` | [`CostProblem`](@ref), [`NelderMeadOptions`](@ref)
[Subgradient Method](@ref SubgradientSolver) | `subgradient_method.jl` | [`SubGradientProblem`](@ref), [`SubGradientMethodOptions`](@ref)
[Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) | `truncatedConjugateGradient.jl` | [`HessianProblem`](@ref),
[`TruncatedConjugateGradientOptions`](@ref)
[The Riemannian Trust-Regions Solver](@ref trust_regions) | `trust_regions.jl` |
[`HessianProblem`](@ref), [`TrustRegionsOptions`](@ref)

Note that the [`Options`](@ref) can also be decorated to enhance your algorithm
by general additional properties.

## [StoppingCriteria](@id StoppingCriteria)

Stopping criteria are implemented as a `functor`, i.e. inherit from the base type

```@docs
StoppingCriterion
StoppingCriterionSet
```

```@autodocs
Modules = [Manopt]
Pages = ["plans/stoppingCriterion.jl"]
Order = [:type]
```

as well as the functions

```@docs
get_reason
get_stopping_criteria
get_active_stopping_criteria
```

further stopping criteria might be available for individual Solvers.

## [Decorated Solvers](@id DecoratedSolvers)

The following decorators are available.

### [Debug Solver](@id DebugSolver)

The decorator to print debug during the iterations can be activated by
decorating the [`Options`](@ref) with [`DebugOptions`](@ref) and implementing
your own [`DebugAction`](@ref)s.
For example printing a gradient from the [`GradientDescentOptions`](@ref) is
automatically available, as explained in the [`steepest_descent`](@ref) solver.

```@autodocs
Modules = [Manopt]
Pages   = ["debug_solver.jl"]
```

### [Record Solver](@id RecordSolver)

The decorator to record certain values during the iterations can be activated by
decorating the [`Options`](@ref) with [`RecordOptions`](@ref) and implementing
your own [`RecordAction`](@ref)s.
For example recording the gradient from the [`GradientDescentOptions`](@ref) is
automatically available, as explained in the [`steepest_descent`](@ref) solver.

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
