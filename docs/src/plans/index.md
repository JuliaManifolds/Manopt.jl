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
get_options
```

Since the `Options` directly relate to a solver, they are documented with the
corresponding [Solvers](@ref).
You can always access the options (since they
might be decorated) by calling [`get_options`](@ref).

### Decorators for Options

Options can be decorated using the following trait and function to initialize

```@docs
dispatch_options_decorator
is_options_decorator
decorate_options
```

In general decorators often perform actions so we introduce

```@docs
AbstractOptionsAction
```

as well as a helper for storing values using keys, i.e.

```@docs
StoreOptionsAction
get_storage
has_storage
update_storage!
```

#### [Debug Options](@id DebugOptions)

```@autodocs
Modules = [Manopt]
Pages = ["plans/debug_options.jl"]
Order = [:type, :function]
```

see [DebugSolver](@ref DebugSolver) for details on the decorated solver.

Further specific [`DebugAction`](@ref)s can be found at the specific Options.

#### [Record Options](@id RecordOptions)

```@autodocs
Modules = [Manopt]
Pages = ["plans/record_options.jl"]
Order = [:type, :function]
Private = false
```

see [RecordSolver](@ref RecordSolver) for details on the decorated solver.

Further specific [`RecordAction`](@ref)s can be found at the specific Options.

there's one internal helper that might be useful for you own actions, namely

```@docs
record_or_eset!
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
Order = [:type,:function]
```

## Problems

A problem usually contains its cost function and provides and
implementation to access the cost

```@docs
Problem
get_cost
```

For any algorithm that involves a cyclic evalutaion, e.g.
[`cyclic_proximal_point`](@ref), one can specify the [`EvalOrder`](@ref) as

```@docs
EvalOrder
LinearEvalOrder
RandomEvalOrder
FixedRandomEvalOrder
```

### Cost based problem

```@docs
CostProblem
```

### Gradient based problem

```@docs
GradientProblem
get_gradient
```

### Subgradient based problem

```@docs
SubGradientProblem
get_subgradient
```

### [Proximal Map(s) based problem](@id ProximalProblem)

```@docs
ProximalProblem
getProximalMap
```

### Further planned problems

```@docs
HessianProblem
getHessian
get_preconditioner
```
