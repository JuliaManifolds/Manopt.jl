
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

## [Stopping Criteria](@id StoppingCriterion)
Stopping criteria are in general modeled as a function

```
(p::P where {P <: Problem}, o::O where {O <: Options}, i::Int) -> s,r
```
where `s` is a `Bool` indicating whether to stop or not and `r` is a `String`,
which is empty if `s` is false and contains a reason, why the algorithm stopped
otherwise. Providing such a function as the usual `stoppingCriterion` option or
field might be cumbersome, so there are a few default ones

```@docs
stopAtIteration
stopGradientNormLess
stopChangeLess
```

as well as two concatenator functions, i.e.
```@docs
stopWhenAny
stopWhenAll
```

## [Decorated solvers](@id DecoratedSolvers)

The following decorators are available.

### [Debug Solver](@id DebugSolver)

The decorator to print debug during the iterations can be activated by
decorating the [`Options`](@ref) with [`DebugOptions`](@ref) and implementing
your own `Symbols` for the [`debug`](@ref) function. For example printing a
gradient from the [`GradientDescentOptions`](@ref) is automatically available,
see  [`DebugOptions`](@ref) for general keys and the specific solvers for more
details on specific ones.

```@docs
initializeSolver!(p::P,o::O) where {P <: Problem, O <: DebugOptions}
doSolverStep!(p::P,o::O, iter) where {P <: Problem, O <: DebugOptions}
getSolverResult(p::P,o::O) where {P <: Problem, O <: DebugOptions}
```

### [Record Solver](@id RecordSolver)

The record solver acts on the [`RecordOptions`](@ref), which also provide the
functions to handle access to the recorded values afterwards. Internally each of
the following functions calls the solver the [`RecordOptions`](@ref) decorate

```@docs
initializeSolver!(p::P,o::O) where {P <: Problem, O <: RecordOptions}
doSolverStep!(p::P,o::O, iter) where {P <: Problem, O <: RecordOptions}
getSolverResult(p::P,o::O) where {P <: Problem, O <: RecordOptions}
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
```
