# [Plans for solvers](@id planSection)

```@meta
CurrentModule = Manopt
```

In order to start a solver, both a [`Problem`](@ref) and [`Options`](@ref) are required.
Together they form a __plan__ and these are stored in this folder. For
sub-problems there are maybe also only [`Options`](@ref), since they than refer to the
same problem.

## Options

For most algorithms a certain set of options can either be
generated beforehand of the function with keywords can be used.
Generally the type

```@docs
Options
getOptions
```

Since the `Options` directly relate to a solver, they are documented with the
corresponding [Solvers](@ref).
You can always access the options (since they
might be decorated) by calling [`getOptions`](@ref).

### Decorators for Options

Options can be decorated using the following trait and function to initialize 

```@docs
IsOptionsDecorator
decorateOptions
```

In general decorators often perform actions so we introduce

```@docs
Action
```

as well as a helper for storing values using keys, i.e.

```@docs
StoreOptionsAction
getStorage
hasStorage
updateStorage!
```

#### [Debug Options](@id DebugOptions)

```@autodocs
Modules = [Manopt]
Pages = ["plans/debugOptions.jl"]
Order = [:type, :function]
```

see [DebugSolver](@ref DebugSolver) for details on the decorated solver.

Further specific [`DebugAction`](@ref)s can be found at the specific Options.

#### [Record Options](@id RecordOptions)

```@autodocs
Modules = [Manopt]
Pages = ["plans/recordOptions.jl"]
Order = [:type, :function]
Private = false
```

see [RecordSolver](@ref RecordSolver) for details on the decorated solver.

Further specific [`RecordAction`](@ref)s can be found at the specific Options.

there's one internal helper that might be useful for you own actions, namely

```@docs
recordOrReset!
```

### [Stepsize and Linesearch](@id Stepsize)
The step size determination is implemented as a `Functor` based on
```@docs
Stepsize
```
in general there are

```@autodocs
Modules = [Manopt]
Pages = ["plans/stepsize.jl"]
Order = [:type]
```

## Problems
A problem usually contains its cost function and provides and
implementation to access the cost
```@docs
Problem
getCost
```

For any algorithm that involves a cyclic evalutaion, e.g.
[`cyclicProximalPoint`](@ref), one can specify the [`EvalOrder`](@ref) as
```@docs
EvalOrder
LinearEvalOrder
RandomEvalOrder
FixedRandomEvalOrder
```

### Gradient based problem
```@docs
GradientProblem
getGradient
```

### Subgradient based problem
```@docs
SubGradientProblem
getSubGradient
```

### [Proximal Map(s) based problem](@id ProximalProblem)
```@docs
ProximalProblem
getProximalMap
```

### Further planned problems
```@docs
HessianProblem
```
