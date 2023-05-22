# How to Count and Cache Function Calls
Ronny Bergmann

In this tutorial, we want to investigate the caching and counting (i.e. statistics) features
of [Manopt.jl](https://manoptjl.org). We will reuse the optimization tasks from the
introductory tutorial [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/Optimize!.html).

## Introduction

There are surely many ways to keep track for example of how often the cost function is called,
for example with a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects), as we used in an example in [How to Record Data](https://manoptjl.org/stable/tutorials/HowtoRecord.html)

``` julia
mutable struct MyCost{I<:Integer}
    count::I
end
MyCost() = MyCost{Int64}(0)
function (c::MyCost)(M, x)
    c.count += 1
    # [ .. Actual implementation of the cost here ]
end
```

This still leaves a bit of work to the user, especially for tracking more than just the number of cost function evaluations.

When a function like the objective or gradient is expensive to compute, it may make sense to cache its results.
Manopt.jl tries to minimize the number of repeated calls but sometimes they are necessary and harmless when the function is cheap to compute.
Caching of expensive function calls can for example be added using [Memoize.jl](https://github.com/JuliaCollections/Memoize.jl) by the user.
The approach in the solvers of [Manopt.jl](https://manoptjl.org) aims to simplify adding
both these capabilities on the level of calling a solver.

## Technical Background

The two ingredients for a solver in [Manopt.jl](https://manoptjl.org)
are the [`AbstractManoptProblem`](@ref) and the [`AbstractManoptSolverState`](@ref), where the
former consists of the domain, that is the [manifold](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/types.html#The-AbstractManifold) and [`AbstractManifoldObjective`](@ref).

Both recording and debug capabilities are implemented in a decorator pattern to the solver state.
They can be easily added using the `record=` and `debug=` in any solver call.
This pattern was recently extended, such that also the objective can be decorated.
This is how both caching and counting are implemented, as decorators of the [`AbstractManifoldObjective`](@ref)
and hence for example changing/extending the behaviour of a call to [`get_cost`](@ref).

Let’s finish off the technical background by loading the necessary packages.
Besides [Manopt.jl](https://manoptjl.org) and [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/latest/) we also need
[LRUCaches.jl](https://github.com/JuliaCollections/LRUCache.jl) which are (since Julia 1.9) a weak dependency and provide
the *least recently used* strategy for our caches.

``` julia
using Manopt, Manifolds, Random, LRUCache
```

## Counting

We first define our task, the Riemannian Center of Mass from the [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/Optimize!.html) tutorial.

``` julia
n = 100
σ = π / 8
M = Sphere(2)
p = 1 / sqrt(2) * [1.0, 0.0, 1.0]
data = [exp(M, p,  σ * rand(M; vector_at=p)) for i in 1:n];
f(M, p) = sum(1 / (2 * n) * distance.(Ref(M), Ref(p), data) .^ 2)
grad_f(M, p) = sum(1 / n * grad_distance.(Ref(M), data, Ref(p)));
```

to now count how often the cost and the gradient are called, we use the `count=` keyword
argument that works in any solver to specify the elements of the objective whose calls we
want to count calls to. A full list is available in the documentation of the
[`AbstractManifoldObjective`](@ref).
To also see the result, we have to set `return_objective=true`.
This returns `(objective, p)` instead of just the solver result `p`.
We can further also set `return_state=true` to get even more information about the solver run.

``` julia
gradient_descent(M, f, grad_f, data[1]; count=[:Cost, :Gradient], return_objective=true, return_state=true)
```

    # Solver state for `Manopt.jl`s Gradient Descent
    After 72 iterations

    ## Parameters
    * retraction method: ExponentialRetraction()

    ## Stepsize
    ArmijoLineseach() with keyword parameters
      * initial_stepsize    = 1.0
      * retraction_method   = ExponentialRetraction()
      * contraction_factor  = 0.95
      * sufficient_decrease = 0.1

    ## Stopping Criterion
    Stop When _one_ of the following are fulfilled:
        Max Iteration 200:  not reached
        |grad f| < 1.0e-9: reached
    Overall: reached
    This indicates convergence: Yes

    ## Statistics on function calls
      * :Gradient :  217
      * :Cost     :  298
    on a ManifoldGradientObjective{AllocatingEvaluation}

And we see that statistics are shown in the end. To now also cache these calls,
we can use the `cache=` keyword argument.
Since now both the cache and the count “extend” the functionality of the objective,
the order is important: On the high-level interface, the `count` is treated first, which
means that only actual function calls and not cache look-ups are counted.
With the proper initialisation, you can use any caches here that support the
`get!(function, cache, key)!` update. All parts of the objective that can currently be cached are listed at [`ManifoldCachedObjective`](@ref). The solver call has a keyword `cache` that takes a tuple`(c, vs, n)` of three arguments, where `c` is a symbol for the type of cache, `vs` is a vector of symbols, which calls to cache and `n` is the size of the cache. If the last element is not provided, a suitable default (currently`n=10`) is used.

Here we want to use `c=:LRU` caches for `vs=[Cost, :Gradient]` with a size of `n=25`.

``` julia
r = gradient_descent(M, f, grad_f, data[1];
    count=[:Cost, :Gradient],
    cache=(:LRU, [:Cost, :Gradient], 25),
    return_objective=true, return_state=true)
```

    # Solver state for `Manopt.jl`s Gradient Descent
    After 72 iterations

    ## Parameters
    * retraction method: ExponentialRetraction()

    ## Stepsize
    ArmijoLineseach() with keyword parameters
      * initial_stepsize    = 1.0
      * retraction_method   = ExponentialRetraction()
      * contraction_factor  = 0.95
      * sufficient_decrease = 0.1

    ## Stopping Criterion
    Stop When _one_ of the following are fulfilled:
        Max Iteration 200:  not reached
        |grad f| < 1.0e-9: reached
    Overall: reached
    This indicates convergence: Yes

    ## Statistics on function calls
      * :Gradient :  72
      * :Cost     :  164
    on a ManifoldGradientObjective{AllocatingEvaluation}

Since the default setup with [`ArmijoLinesearch`](@ref) needs the gradient and the
cost, and similarly the stopping criterion might (independently) evaluate the gradient,
the caching is quite helpful here.

And of course also for this advanced return value of the solver, we can still access the
result as usual:

``` julia
get_solver_result(r)
```

    3-element Vector{Float64}:
     0.7298774364923435
     0.047665824852873
     0.6819141418393224
