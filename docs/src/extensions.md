# Extensions

## LineSearches.jl

Manopt can be used with line search algorithms implemented in [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl).
This can be illustrated by the following example of optimizing Rosenbrock function constrained to the unit sphere.

```@example
using Manopt, Manifolds, LineSearches

# define objective function and its gradient
p = [1.0, 100.0]
function rosenbrock(::AbstractManifold, x)
    val = zero(eltype(x))
    for i in 1:(length(x) - 1)
        val += (p[1] - x[i])^2 + p[2] * (x[i + 1] - x[i]^2)^2
    end
    return val
end
function rosenbrock_grad!(M::AbstractManifold, storage, x)
    storage .= 0.0
    for i in 1:(length(x) - 1)
        storage[i] += -2.0 * (p[1] - x[i]) - 4.0 * p[2] * (x[i + 1] - x[i]^2) * x[i]
        storage[i + 1] += 2.0 * p[2] * (x[i + 1] - x[i]^2)
    end
    project!(M, storage, x, storage)
    return storage
end
# define constraint
n_dims = 5
M = Manifolds.Sphere(n_dims)
# set initial point
x0 = vcat(zeros(n_dims - 1), 1.0)
# use LineSearches.jl HagerZhang method with Manopt.jl quasiNewton solver
ls_hz = Manopt.LineSearchesStepsize(M, LineSearches.HagerZhang())
x_opt = quasi_Newton(
    M,
    rosenbrock,
    rosenbrock_grad!,
    x0;
    stepsize=ls_hz,
    evaluation=InplaceEvaluation(),
    stopping_criterion=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
    return_state=true,
)
```

In general this defines the following new [stepsize](@ref Stepsize)

```@docs
Manopt.LineSearchesStepsize
```

## Manifolds.jl

Loading `Manifolds.jl` introduces the following additional functions

```@docs
Manopt.max_stepsize(::FixedRankMatrices, ::Any)
Manopt.max_stepsize(::Hyperrectangle, ::Any)
Manopt.max_stepsize(::TangentBundle, ::Any)
mid_point
```

Internally, `Manopt.jl` provides the two additional functions to choose some
Euclidean space when needed as

```@docs
Manopt.Rn
Manopt.Rn_default
```

## JuMP.jl

Manopt can be used using the [JuMP.jl](https://jump.dev) interface, see the tutorial TODO:

### Setting s solver and its options

A main thing to choose is the solver to use. By default this is set to the [`GradientDescentState`](@ref). To change the solver you can set it with `set_attribute`. For exmple to use the [`quasi_Newton`](@ref) instead, use

```{julia}
model =  Model(Manopt.JuMP_Optimizer)
set_attribute(model, "descent_state_type", Manopt.QuasiNewtonState)
```

Any of the keywords of the solver you can set with `set_attribute)model, keyword, value)` for example to change the retraction to use, call

```{julia}
set_attribute(model, "retraction_method", ManifoldsBase.ProjectionRetraction())
```

### Interface functions

Several functions from the [Mathematical Optimization Interface](https://github.com/jump-dev/MOI.jl) (MOI) are
extended when both `Manopt.jl and [JuMP.jl](https://jump.dev) are loaded:

```@docs
Manopt.JuMP_Optimizer
Manopt.JuMP_ManifoldSet
JuMP.build_variable
JuMP.jump_function
JuMP.jump_function_type
JuMP.set_objective_function
MOI.add_constrained_variables
MOI.copy_to
MOI.empty!
MOI.dimension
MOI.Utilities.map_indices
MOI.supports_add_constrained_variables
MOI.get
MOI.is_valid
MOI.supports
MOI.supports_incremental_interface
MOI.set
```

as well as the internal functions

```@autodocs
Modules = [Base.get_extension(Manopt, :ManoptJuMPExt)]
Order   = [:type, :function]
```
