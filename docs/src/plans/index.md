```@meta
CurrentModule = Manopt
```
# Plans for solvers
In order to start a solver, both a `:Problem` and `:Options` are required.
Together they form a `plan` and these are stored in this folder. For
sub-problems there are maybe also only `options`, since they than refer to the
same problem.
Since the `options` directly relate to a solver, they are documented with the
corresponding [Solvers](@ref) for now.

## Problems
A problem usually contains its cost function and provides and
implementation to access the cost
```@docs
Problem
getCost
```

### Gradient based problems
```@docs
GradientProblem
getGradient
```

### Proximal Map(s) based problems
```@docs
ProximalProblem
getProximalMap
getProximalMaps
```

## Options
```@docs
ArmijoLineSearchOptions
DouglasRachfordOptions
GradientDescentOptions
LineSearchOptions
SimpleLineSearchOptions
ConjugateGradientOptions
SimpleDirectionUpdateOptions
```

# Until I find a better place to reduce errors...
```@docs
ArmijoLineSearch
```
