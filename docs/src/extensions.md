# Extensions

## LineSearches.jl

`Manopt.jl` can be used with line search algorithms implemented in [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl).
This can be illustrated by the following example of optimizing the Rosenbrock function constrained to the unit sphere.

```@example
using Manopt, Manifolds, LineSearches

# define objective function and its gradient
a, b = 1.0, 100.0
function rosenbrock(::AbstractManifold, p)
    val = zero(eltype(p))
    for i in 1:(length(p) - 1)
        val += (a - p[i])^2 + b * (p[i + 1] - p[i]^2)^2
    end
    return val
end
function rosenbrock_grad!(M::AbstractManifold, X, p)
    X .= 0.0
    for i in 1:(length(p) - 1)
        X[i] += -2.0 * (a - p[i]) - 4.0 * b * (p[i + 1] - p[i]^2) * p[i]
        X[i + 1] += 2.0 * b * (p[i + 1] - p[i]^2)
    end
    project!(M, X, p, X)
    return X
end
# define constraint
n_dims = 5
M = Manifolds.Sphere(n_dims)
# set initial point
p0 = vcat(zeros(n_dims), 1.0)
# use LineSearches.jl HagerZhang method with Manopt.jl quasi_Newton solver
ls_hz = Manopt.LineSearchesStepsize(M, LineSearches.HagerZhang())
p_opt = quasi_Newton(
    M,
    rosenbrock,
    rosenbrock_grad!,
    p0;
    stepsize=ls_hz,
    evaluation=InplaceEvaluation(),
    stopping_criterion=StopAfterIteration(1000) | StopWhenGradientNormLess(1e-6),
    return_state=true,
)
```

In general this defines the following new [step size](@ref Stepsize) with helper functions for setting and getting the maximum step size:

```@docs
Manopt.LineSearchesStepsize
Manopt.linesearches_get_max_alpha
Manopt.linesearches_set_max_alpha
```

## Manifolds.jl

Loading `Manifolds.jl` introduces the following additional functions:

```@docs
Manopt.max_stepsize(::FixedRankMatrices, ::Any)
Manopt.max_stepsize(::Hyperrectangle, ::Any)
Manopt.max_stepsize(::TangentBundle, ::Any)
mid_point
```

Internally, `Manopt.jl` provides the following two additional functions to choose some
Euclidean space when needed:

```@docs
Manopt.Rn
Manopt.Rn_default
```

## RecursiveArrayTools.jl

Loading `RecursiveArrayTools.jl` provides the [alternating gradient descent](@ref solver-alternating-gradient-descent) solver
on a [`ProductManifold`](@extref ManifoldsBase ProductManifold) as well as the following two ways to
evaluate the gradient of a [`ManifoldAlternatingGradientObjective`](@ref).

```@docs

Manopt.get_gradient(::ProductManifold, ::Manopt.ManifoldAlternatingGradientObjective, ::Any)
Manopt.get_gradient!(::ProductManifold, ::Any, ::Manopt.ManifoldAlternatingGradientObjective, ::Any)
Manopt.get_gradient(::AbstractManifold, ::Manopt.ManifoldAlternatingGradientObjective, ::Any, ::Any)
Manopt.get_gradient!(::AbstractManifold, ::Any, ::Manopt.ManifoldAlternatingGradientObjective, ::Any, ::Any)
```
