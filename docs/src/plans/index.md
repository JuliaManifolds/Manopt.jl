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

Further specific [`DebugAction`](@ref)s can be found at the specific Options.

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