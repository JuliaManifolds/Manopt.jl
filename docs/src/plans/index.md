# [Plans for solvers](@id planSection)

```@meta
CurrentModule = Manopt
```

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

#### [Debug Options](@id DebugOptions)

```@autodocs
Modules = [Manopt]
Pages = ["plans/debugOptions.jl"]
Order = [:type, :function]
```

see [DebugSolver](@ref DebugSolver) for details on the decorated solver.

For arbitrary [`Problem`](@ref)s and [`Options`](@ref) the following Symbols, i.e. types of debug, are automatically available.

* `:Change` - print the last change in the variable
* `:Cost` - print the cost function evaluated at the current iterate
* `:InitialCost` - print the initial cost function
* `:FinalCost` - print the final cost function. This is explicitly called after the last iteration if activated.
* `:Divider` – print a divider `" |  "` between the other debugs.
* `:Iteration` – print the current iteration number
* `:Iterate` – print the current iterate `o.x`
* `:Newline` – print a newline character.
* `:Solver` - print status of the solver
* `:StoppingCriterion` - print status of the stopping criterion

These debug symbols assume, that `p.M` is the manifold the optimization problem is defined on, `o.x` is the current iterate, `o.xOld` is the last iterates value, `getCost(p,x)` evaluates the cost function associated to the [`Problem`](@ref) `p`. Both `:Solver` and `:StoppingCriterion` by default print a `String` they get passed to, i.e. to print status of the solver or the string the `StoppingCriterion` provide.

For further `:Symbols` that add additional debug print capabilities, see the specific solvers and their plans.

#### [Record Options](@id RecordOptons)
```@autodocs
Modules = [Manopt]
Pages = ["plans/recordOptions.jl"]
Order = [:type, :function]
```

Your own `:Symbol` has to provide both the [`record`](@ref) and the [`recordType`](@ref) function. The following records are available by default assuming that `p.M` denotes the manifold we optimize on,
`o.x`, `o.xOld` are the current and last iterate. For each symbol the
type is given in brackets.
* `:Iteration` (`Int`) – the current iteration number
* `:Iterate` (`typeof(o.x)` ) – the current iterate, i.e. the type is a `<: MPoint`
* `:Change` (`Float64`) the last change
* `:Cost` (`Float64`) the cost function of the current iterate.

These records assume that `o.x`,`o.xOld` are the current and last iterate within the current [`Options`](@ref) decorated, respectively, and that `p.M` refers to the [`Manifold`](@ref) the [`Problem`](@ref) is formulated on. 

For further `:Symbols` providing special recording capabilities of special solvers, see
the details in the specific solvers.

### [Stepsize and Linesearch Options](@id StepsizeOptions)

using the following `<:Options`.
```@docs
StepsizeOptions
SimpleStepsizeOptions
LinesearchOptions
ArmijoLinesearchOptions
```

## Problems
A problem usually contains its cost function and provides and
implementation to access the cost
```@docs
Problem
getCost
evaluateStoppingCriterion
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
Whenever we have a problem involving a gradient, we can employ a line search,
which is itself not a complete solver but organized in the Algorithms section.
```@docs
ArmijoLineSearch
getInitialStepsize
getStepsize
```
In order to get a step size rule running, the algorithms require both a function
performing the line search and options for the line search, see [`LinesearchOptions`](@ref).

There are short hands available, namely
```@docs
ConstantStepsize
DecreasingStepsize
Armijo
```


## Subgradient based problem
```@docs
SubGradientProblem
getSubGradient
```

### Hessian based problem
*note that this section is preliminary, there is no Hessian based algorithm yet*
```@docs
HessianProblem
getHessian
```

### Proximal Map(s) based problem
```@docs
ProximalProblem
getProximalMap
getProximalMaps
```

serves as a common base type for the following
```@docs
SimpleDirectionUpdateOptions
HessianDirectionUpdateOptions
```
