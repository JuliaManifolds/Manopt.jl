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

Manopt can be used using the [JuMP.jl](https://github.com/jump-dev/JuMP.jl) interface.
The manifold is provided in the `@variable` macro. Note that until now,
only variables (points on manifolds) are supported, that are arrays, especially structs do not yet work.
The algebraic expression of the objective function is specified in the `@objective` macro.
The `descent_state_type` attribute specifies the solver.

```julia
using JuMP, Manopt, Manifolds
model = Model(Manopt.JuMP_Optimizer)
# Change the solver with this option, `GradientDescentState` is the default
set_attribute("descent_state_type", GradientDescentState)
@variable(model, U[1:2, 1:2] in Stiefel(2, 2), start = 1.0)
@objective(model, Min, sum((A - U) .^ 2))
optimize!(model)
solution_summary(model)
```

### Interface functions

```@docs
Manopt.JuMP_Optimizer
MOI.dimension
MOI.empty!
MOI.get
MOI.is_valid
MathOptInterface.set :: Tuple{ManoptJuMPExt.ManoptOptimizer, MathOptInterface.VariablePrimalStart, MathOptInterface.VariableIndex, Union{Nothing, Real}}
MOI.supports
MOI.supports_incremental_interface
MOI.supports_add_constrained_variables
```