# Speedup using in-place evaluation
Ronny Bergmann

When it comes to time critical operations, a main ingredient in Julia is given by
mutating functions, that is those that compute in place without additional memory
allocations. In the following, we illustrate how to do this with `Manopt.jl`.

Let’s start with the same function as in [Get started: optimize!](https://manoptjl.org/stable/tutorials/Optimize!.html)
and compute the mean of some points, only that here we use the sphere $\mathbb S^{30}$
and $n=800$ points.

From the aforementioned example.

We first load all necessary packages.

``` julia
using Manopt, Manifolds, Random, BenchmarkTools
using ManifoldDiff: grad_distance, grad_distance!
Random.seed!(42);
```

And setup our data

``` julia
Random.seed!(42)
m = 30
M = Sphere(m)
n = 800
σ = π / 8
p = zeros(Float64, m + 1)
p[2] = 1.0
data = [exp(M, p, σ * rand(M; vector_at=p)) for i in 1:n];
```

## Classical Definition

The variant from the previous tutorial defines a cost $f(x)$ and its gradient $\operatorname{grad}f(p)$
““”

``` julia
f(M, p) = sum(1 / (2 * n) * distance.(Ref(M), Ref(p), data) .^ 2)
grad_f(M, p) = sum(1 / n * grad_distance.(Ref(M), data, Ref(p)))
```

    grad_f (generic function with 1 method)

We further set the stopping criterion to be a little more strict. Then we obtain

``` julia
sc = StopWhenGradientNormLess(3e-10)
p0 = zeros(Float64, m + 1); p0[1] = 1/sqrt(2); p0[2] = 1/sqrt(2)
m1 = gradient_descent(M, f, grad_f, p0; stopping_criterion=sc);
```

We can also benchmark this as

``` julia
@benchmark gradient_descent($M, $f, $grad_f, $p0; stopping_criterion=$sc)
```

    BenchmarkTools.Trial: 106 samples with 1 evaluation.
     Range (min … max):  46.774 ms …  50.326 ms  ┊ GC (min … max): 2.31% … 2.47%
     Time  (median):     47.207 ms               ┊ GC (median):    2.45%
     Time  (mean ± σ):   47.364 ms ± 608.514 μs  ┊ GC (mean ± σ):  2.53% ± 0.25%

         ▄▇▅▇█▄▇                                                    
      ▅▇▆████████▇▇▅▅▃▁▆▁▁▁▅▁▁▅▁▃▃▁▁▁▁▁▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅ ▃
      46.8 ms         Histogram: frequency by time         50.2 ms <

     Memory estimate: 182.50 MiB, allocs estimate: 615822.

## In-place Computation of the Gradient

We can reduce the memory allocations by implementing the gradient to be evaluated in-place.
We do this by using a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects).
The motivation is twofold: on one hand, we want to avoid variables from the global scope,
for example the manifold `M` or the `data`, being used within the function.
Considering to do the same for more complicated cost functions might also be worth pursuing.

Here, we store the data (as reference) and one introduce temporary memory in order to avoid
reallocation of memory per `grad_distance` computation. We get

``` julia
struct GradF!{TD,TTMP}
    data::TD
    tmp::TTMP
end
function (grad_f!::GradF!)(M, X, p)
    fill!(X, 0)
    for di in grad_f!.data
        grad_distance!(M, grad_f!.tmp, di, p)
        X .+= grad_f!.tmp
    end
    X ./= length(grad_f!.data)
    return X
end
```

For the actual call to the solver, we first have to generate an instance of `GradF!`
and tell the solver, that the gradient is provided in an [`InplaceEvaluation`](https://manoptjl.org/stable/plans/objective/#Manopt.InplaceEvaluation).
We can further also use [`gradient_descent!`](https://manoptjl.org/stable/solvers/gradient_descent/#Manopt.gradient_descent!) to even work in-place of the initial point we pass.

``` julia
grad_f2! = GradF!(data, similar(data[1]))
m2 = deepcopy(p0)
gradient_descent!(
    M, f, grad_f2!, m2; evaluation=InplaceEvaluation(), stopping_criterion=sc
);
```

We can again benchmark this

``` julia
@benchmark gradient_descent!(
    $M, $f, $grad_f2!, m2; evaluation=$(InplaceEvaluation()), stopping_criterion=$sc
) setup = (m2 = deepcopy($p0))
```

    BenchmarkTools.Trial: 176 samples with 1 evaluation.
     Range (min … max):  27.358 ms … 84.206 ms  ┊ GC (min … max): 0.00% … 0.00%
     Time  (median):     27.768 ms              ┊ GC (median):    0.00%
     Time  (mean ± σ):   28.504 ms ±  4.338 ms  ┊ GC (mean ± σ):  0.60% ± 1.96%

        ▂█▇▂ ▂                                                     
      ▆▇████▆█▆▆▄▄▃▄▄▃▃▃▁▃▃▃▃▃▃▃▃▃▄▃▃▃▃▃▃▁▃▁▁▃▁▁▁▁▁▁▃▃▁▁▃▃▁▁▁▁▃▃▃ ▃
      27.4 ms         Histogram: frequency by time        31.4 ms <

     Memory estimate: 3.83 MiB, allocs estimate: 5797.

which is faster by about a factor of 2 compared to the first solver-call.
Note that the results `m1` and `m2` are of course the same.

``` julia
distance(M, m1, m2)
```

    2.4669338186126805e-17

## Technical details

This tutorial is cached. It was last run on the following package versions.

``` julia
using Pkg
Pkg.status()
```

    Status `~/Repositories/Julia/Manopt.jl/tutorials/Project.toml`
      [6e4b80f9] BenchmarkTools v1.5.0
      [5ae59095] Colors v0.12.11
      [31c24e10] Distributions v0.25.108
      [26cc04aa] FiniteDifferences v0.12.31
      [7073ff75] IJulia v1.24.2
      [8ac3fa9e] LRUCache v1.6.1
      [af67fdf4] ManifoldDiff v0.3.10
      [1cead3c2] Manifolds v0.9.18
      [3362f125] ManifoldsBase v0.15.10
      [0fc0a36d] Manopt v0.4.63 `..`
      [91a5bcdd] Plots v1.40.4

``` julia
using Dates
now()
```

    2024-05-26T13:52:05.613
