
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
[steepest Descent](@ref GradientDescentSolver) | `steepestDescent.jl` |  [`GradientProblem`](@ref), [`GradientDescentOptions`](@ref)
[Cyclic Proximal Point](@ref CPPSolver) | `cyclicProximalPoint.jl` | [`ProximalProblem`](@ref), [`CyclicProximalPointOptions`](@ref)
[Douglas–Rachford](@ref DRSolver) | `DouglasRachford.jl` | [`ProximalProblem`](@ref), [`DouglasRachfordOptions`](@ref)
[Subgradient Method](@ref SubgradientSolver) | `subGradientMethod.jl` | [`SubGradientProblem`](@ref), [`SubGradientMethodOptions`](@ref)

Note that the [`Options`](@ref) can also be decorated to enhance your algorithm
by general additional properties.

## [StoppingCriteria](@id StoppingCriteria)

Stopping criteria are implemented as a `functor`, i.e. inherit from the base type

```@docs
StoppingCriterion
```

```@autodocs
Modules = [Manopt]
Pages = ["plans/stoppingCriterion.jl"]
Order = [:type]
```

as well as the function

```@docs
getReason
```

further stopping criteria might be available for individual Solvers.

## [Decorated Solvers](@id DecoratedSolvers)

The following decorators are available.

### [Debug Solver](@id DebugSolver)

The decorator to print debug during the iterations can be activated by
decorating the [`Options`](@ref) with [`DebugOptions`](@ref) and implementing
your own [`DebugAction`](@ref)s.
For example printing a gradient from the [`GradientDescentOptions`](@ref) is automatically available, as explained in the [`steepestDescent`](@ref) solver.

```@docs
initializeSolver!(p::P,o::O) where {P <: Problem, O <: DebugOptions}
doSolverStep!(p::P,o::O, iter) where {P <: Problem, O <: DebugOptions}
getSolverResult(p::P,o::O) where {P <: Problem, O <: DebugOptions}
stopSolver!(p::P,o::O, i::Int) where {P <: Problem, O <: DebugOptions}
```

### [Record Solver](@id RecordSolver)

The decorator to record certain values during the iterations can be activated by
decorating the [`Options`](@ref) with [`RecordOptions`](@ref) and implementing
your own [`RecordAction`](@ref)s.
For example recording the gradient from the [`GradientDescentOptions`](@ref) is
automatically available, as explained in the [`steepestDescent`](@ref) solver.

```@docs
initializeSolver!(p::P,o::O) where {P <: Problem, O <: RecordOptions}
doSolverStep!(p::P,o::O, iter) where {P <: Problem, O <: RecordOptions}
getSolverResult(p::P,o::O) where {P <: Problem, O <: RecordOptions}
stopSolver!(p::P,o::O, i::Int) where {P <: Problem, O <: RecordOptions}
```

## Technical Details

 The main function a solver calls is

```@docs
solve(p::P, o::O) where {P <: Problem, O <: Options}
```

which is a framework, that you in general should not change or redefine.
It uses the following methods, which also need to be implemented on your own
algorithm, if you want to provide one.

```@docs
initializeSolver!(p::P,o::O) where {P <: Problem, O <: Options}
doSolverStep!(p::P,o::O, iter) where {P <: Problem, O <: Options}
getSolverResult(p::P,o::O) where {P <: Problem, O <: Options}
stopSolver!(p::P,o::O, i::Int) where {P <: Problem, O <: Options}
```
