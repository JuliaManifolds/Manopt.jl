# Speedup using in-place evaluation
Ronny Bergmann

When it comes to time critical operations, a main ingredient in Julia is given by
mutating functions, that is those that compute in place without additional memory
allocations. In the following, we illustrate how to do this with `Manopt.jl`.

Let’s start with the same function as in [🏔️ Get started with Manopt.jl](getstarted.md)
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

## Classical definition

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

    BenchmarkTools.Trial: 90 samples with 1 evaluation per sample.
     Range (min … max):  51.678 ms … 134.204 ms  ┊ GC (min … max):  9.64% … 38.77%
     Time  (median):     53.536 ms               ┊ GC (median):    11.71%
     Time  (mean ± σ):   55.776 ms ±   9.262 ms  ┊ GC (mean ± σ):  12.53% ±  3.19%

      █▇▁▇▁▅▂     ▁                                                 
      ███████▆▅▆▆▆█▁▃▆▅▃▃▅▃▁▁▁▁▁▁▁▃▁▁▁▁▁▁▁▃▁▁▁▁▁▁▅▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▅ ▁
      51.7 ms         Histogram: frequency by time         71.5 ms <

     Memory estimate: 173.76 MiB, allocs estimate: 1167364.

## In-place computation of the gradient

We can reduce the memory allocations by implementing the gradient to be evaluated in-place.
We do this by using a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects).
The motivation is twofold: on one hand, we want to avoid variables from the global scope,
for example the manifold `M` or the `data`, being used within the function.
Considering to do the same for more complicated cost functions might also be worth pursuing.

Here, we store the data (as reference) and one introduce temporary memory to avoid
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

    BenchmarkTools.Trial: 137 samples with 1 evaluation per sample.
     Range (min … max):  35.297 ms … 49.118 ms  ┊ GC (min … max): 0.00% … 25.92%
     Time  (median):     35.863 ms              ┊ GC (median):    0.00%
     Time  (mean ± σ):   36.604 ms ±  1.640 ms  ┊ GC (mean ± σ):  0.67% ±  2.89%

       ▇▇█                                                         
      ▇███▃▅▄▅▄▃▃▅▅▅▃█▇▄▃▃▃▃▃▁▃▃▃▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃ ▃
      35.3 ms         Histogram: frequency by time        44.1 ms <

     Memory estimate: 3.72 MiB, allocs estimate: 6879.

which is faster by about a factor of 2 compared to the first solver-call.
Note that the results `m1` and `m2` are of course the same.

``` julia
distance(M, m1, m2)
```

    4.8317610992693745e-11

## Technical details

This tutorial is cached. It was last run on the following package versions.

    Status `~/Repositories/Julia/Manopt.jl/tutorials/Project.toml`
      [47edcb42] ADTypes v1.14.0
      [6e4b80f9] BenchmarkTools v1.6.0
      [5ae59095] Colors v0.13.0
      [31c24e10] Distributions v0.25.119
      [26cc04aa] FiniteDifferences v0.12.32
      [7073ff75] IJulia v1.27.0
      [8ac3fa9e] LRUCache v1.6.2
      [af67fdf4] ManifoldDiff v0.4.2
      [1cead3c2] Manifolds v0.10.17
      [3362f125] ManifoldsBase v1.1.0
      [0fc0a36d] Manopt v0.5.14 `..`
      [91a5bcdd] Plots v1.40.13
      [731186ca] RecursiveArrayTools v3.33.0

This tutorial was last rendered May 2, 2025, 15:48:41.
