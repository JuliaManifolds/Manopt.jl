# Speedup using in-place evaluation
Ronny Bergmann

When it comes to time critical operations, a main ingredient in Julia is given by
mutating functions, that is those that compute in place without additional memory
allocations. In the following, we illustrate how to do this with `Manopt.jl`.

Let’s start with the same function as in [🏔️ Get started with Manopt.jl](getstarted.md)
and compute the mean of some points, only that here we use the sphere $\mathbb S^{30}$
and $n=800$ points.

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

The variant from the previous tutorial defines a cost $f(p)$ and its gradient $\operatorname{grad}f(p)$

``` julia
f(M, p) = sum(1 / (2 * n) * distance.(Ref(M), Ref(p), data) .^ 2)
grad_f(M, p) = sum(1 / n * grad_distance.(Ref(M), data, Ref(p)))
```

    grad_f (generic function with 1 method)

We further set the stopping criterion to be a little more strict. Then we obtain

``` julia
sc = StopWhenGradientNormLess(5e-9)
p0 = zeros(Float64, m + 1); p0[1] = 1/sqrt(2); p0[2] = 1/sqrt(2)
m1 = gradient_descent(M, f, grad_f, p0; stopping_criterion=sc);
```

We can also benchmark this as

``` julia
@benchmark gradient_descent($M, $f, $grad_f, $p0; stopping_criterion=$sc)
```

    BenchmarkTools.Trial: 109 samples with 1 evaluation per sample.
     Range (min … max):  40.811 ms … 203.364 ms  ┊ GC (min … max):  0.00% … 81.63%
     Time  (median):     42.866 ms               ┊ GC (median):    13.02%
     Time  (mean ± σ):   46.251 ms ±  21.471 ms  ┊ GC (mean ± σ):  18.45% ±  9.91%

      █▁                                                            
      ██▆▅▄▁▁▁▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄ ▄
      40.8 ms       Histogram: log(frequency) by time       200 ms <

     Memory estimate: 129.63 MiB, allocs estimate: 864013.

## In-place computation of the gradient

We can reduce the memory allocations by implementing the gradient to be evaluated in-place.
We do this by using a [functor](https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects).
The motivation is twofold: on one hand, we want to avoid variables from the global scope,
for example the manifold `M` or the `data`, being used within the function.
Considering to do the same for more complicated cost functions might also be worth pursuing.

Here, we store the data (as reference) and introduce temporary memory to avoid
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
and tell the solver, that the gradient is provided in an [`InplaceEvaluation`](@ref).
We can further also use [`gradient_descent!`](@ref) to even work in-place of the initial point we pass.

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

    BenchmarkTools.Trial: 155 samples with 1 evaluation per sample.
     Range (min … max):  29.994 ms … 71.361 ms  ┊ GC (min … max): 0.00% … 0.00%
     Time  (median):     30.624 ms              ┊ GC (median):    0.00%
     Time  (mean ± σ):   32.412 ms ±  6.217 ms  ┊ GC (mean ± σ):  1.83% ± 4.96%

      █▆▂▁                                                         
      ████▆▆▄▆▆▄▁▄▁▆▁▄▄▄▁▁▁▁▁▁▁▄▁▁▁▁▆▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▄▁▁▄ ▄
      30 ms        Histogram: log(frequency) by time      63.9 ms <

     Memory estimate: 4.45 MiB, allocs estimate: 10071.

which is faster than the first solver-call and, more importantly, allocates far less memory:
compare both the memory estimate and the number of allocations of the two runs.
Note that the results `m1` and `m2` are of course the same.

``` julia
distance(M, m1, m2)
```

    9.962086562301663e-9

```@raw html
<details>
  <summary>Technical Details</summary>
```

This tutorial is cached. It was last run on the following package versions.

    Status `~/Repositories/Julia/Manopt.jl/tutorials/Project.toml`
    ⌃ [47edcb42] ADTypes v1.22.1
      [6e4b80f9] BenchmarkTools v1.8.0
      [5ae59095] Colors v0.13.1
    ⌃ [a0c0ee7d] DifferentiationInterface v0.7.19
    ⌃ [31c24e10] Distributions v0.25.129
      [26cc04aa] FiniteDifferences v0.12.34
    ⌃ [f6369f11] ForwardDiff v1.4.1
      [8ac3fa9e] LRUCache v1.6.2
      [af67fdf4] ManifoldDiff v0.4.5
      [1cead3c2] Manifolds v0.11.29
    ⌃ [3362f125] ManifoldsBase v2.5.0
      [0fc0a36d] Manopt v0.6.7 `.`
      [91a5bcdd] Plots v1.41.7
      [731186ca] RecursiveArrayTools v4.5.1
      [37e2e46d] LinearAlgebra v1.12.0
      [9a3f8284] Random v1.11.0
    Info Packages marked with ⌃ have new versions available and may be upgradable.

This tutorial was last rendered September 3, 2026, 16:10:43.

```@raw html
</details>
```
