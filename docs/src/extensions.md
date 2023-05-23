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

```@docs
Manopt.LineSearchesStepsize
```
