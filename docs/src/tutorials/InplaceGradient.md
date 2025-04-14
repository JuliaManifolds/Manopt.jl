# Speedup using in-place evaluation
Ronny Bergmann

When it comes to time critical operations, a main ingredient in Julia is given by
mutating functions, that is those that compute in place without additional memory
allocations. In the following, we illustrate how to do this with `Manopt.jl`.

Let’s start with the same function as in [🏔️ Get started with Manopt.jl](https://manoptjl.org/stable/tutorials/getstarted.html)
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

    BenchmarkTools.Trial: 89 samples with 1 evaluation per sample.
     Range (min … max):  52.976 ms … 104.222 ms  ┊ GC (min … max): 8.05% … 5.55%
     Time  (median):     55.145 ms               ┊ GC (median):    9.99%
     Time  (mean ± σ):   56.391 ms ±   6.102 ms  ┊ GC (mean ± σ):  9.92% ± 1.43%

        ▅██▅▃▁
      ▅███████▁▅▇▅▁▅▁▁▅▅▁▁▁▅▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅ ▁
      53 ms         Histogram: log(frequency) by time      81.7 ms <

     Memory estimate: 173.54 MiB, allocs estimate: 1167348.

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

    BenchmarkTools.Trial: 130 samples with 1 evaluation per sample.
     Range (min … max):  36.646 ms … 64.781 ms  ┊ GC (min … max): 0.00% … 0.00%
     Time  (median):     37.559 ms              ┊ GC (median):    0.00%
     Time  (mean ± σ):   38.658 ms ±  3.904 ms  ┊ GC (mean ± σ):  0.73% ± 2.68%

      ██▅▅▄▂▁ ▂
      ███████▁██▁▅▁▁▁▅▁▁▁▁▅▅▅▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅▁▁▁▁▁▁▁▁▁▅ ▅
      36.6 ms      Histogram: log(frequency) by time        61 ms <

     Memory estimate: 3.59 MiB, allocs estimate: 6863.

which is faster by about a factor of 2 compared to the first solver-call.
Note that the results `m1` and `m2` are of course the same.

``` julia
distance(M, m1, m2)
```

    4.8317610992693745e-11

## Technical details

This tutorial is cached. It was last run on the following package versions.

``` julia
using Pkg
Pkg.status()
```

    Status `~/Repositories/Julia/Manopt.jl/tutorials/Project.toml`
      [47edcb42] ADTypes v1.13.0
      [6e4b80f9] BenchmarkTools v1.6.0
    ⌃ [5ae59095] Colors v0.12.11
      [31c24e10] Distributions v0.25.117
      [26cc04aa] FiniteDifferences v0.12.32
      [7073ff75] IJulia v1.26.0
      [8ac3fa9e] LRUCache v1.6.1
      [af67fdf4] ManifoldDiff v0.4.2
      [1cead3c2] Manifolds v0.10.13
      [3362f125] ManifoldsBase v1.0.1
      [0fc0a36d] Manopt v0.5.5 `..`
      [91a5bcdd] Plots v1.40.9
      [731186ca] RecursiveArrayTools v3.29.0
    Info Packages marked with ⌃ have new versions available and may be upgradable.

``` julia
using Dates
now()
```

    2025-02-10T13:22:51.002
