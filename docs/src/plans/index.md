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

### Hessian based problems
*note that this section is preliminary, there is no Hessian based algorithm yet*
```@docs
HessianProblem
```

### Proximal Map(s) based problems
```@docs
ProximalProblem
getProximalMap
getProximalMaps
```

## Options
For most algorithms a certain set of options can either be
generated beforehand of the function with keywords can be used.
Generally the type
```@docs
Options
```
serves as a common base type for the following
```@docs
ArmijoLineSearchOptions
DouglasRachfordOptions
GradientDescentOptions
LineSearchOptions
SimpleLineSearchOptions
ConjugateGradientOptions
SimpleDirectionUpdateOptions
TrustRegionOptions
TrustRegionSubOptions
```

# Until I find a better place to reduce errors...
```@docs
ArmijoLineSearch
```
