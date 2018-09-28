```@meta
CurrentModule = Manopt
```
# [Plans for solvers](@id planSection)
In order to start a solver, both a `:Problem` and `:Options` are required.
Together they form a `plan` and these are stored in this folder. For
sub-problems there are maybe also only `options`, since they than refer to the
same problem.

## Options
For most algorithms a certain set of options can either be
generated beforehand of the function with keywords can be used.
Generally the type
```@docs
Options
```
Since the `options` directly relate to a solver, they are documented with the
corresponding [Solvers](@ref) for now.
You can always access the options (since they
might be decorated for example with [Debug](@ref SectionDebug)).

A problem usually contains its cost function and provides and
implementation to access the cost
```@docs
Problem
getCost
evaluateStoppingCriterion
```

### Gradient based problems and options
```@docs
GradientProblem
getGradient
```
## Subgradient based problem and options
```@docs
SubGradientProblem
getSubGradient
```
Furthermore there are the following StepSize rules
```@docs
linearDecreasingStepSize
```

### Hessian based problems
*note that this section is preliminary, there is no Hessian based algorithm yet*
```@docs
HessianProblem
getHessian
```

### Proximal Map(s) based problems
```@docs
ProximalProblem
getProximalMap
getProximalMaps
```

serves as a common base type for the following
```@docs
SimpleDirectionUpdateOptions
```
