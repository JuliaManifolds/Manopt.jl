# Speedup using Inplace Evaluation
Ronny Bergmann

When it comes to time critital operations, a main ingredient in Julia is given by
mutating functions, i.e. those that compute in place without additional memory
allocations. In the following, we illustrate how to do this with `Manopt.jl`.

Let’s start with the same function as in [Get Started: Optimize!](https://manoptjl.org/stable/tutorials/Optimize!.html)
and compute the mean of some points, only that here we use the sphere $\mathbb S^{30}$
and $n=800$ points.

From the aforementioned example.

We first load all necessary packages.

``` julia
using Manopt, Manifolds, Random, BenchmarkTools
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

    BenchmarkTools.Trial: 102 samples with 1 evaluation.
     Range (min … max):  47.810 ms …  53.557 ms  ┊ GC (min … max): 5.09% … 6.53%
     Time  (median):     48.820 ms               ┊ GC (median):    5.34%
     Time  (mean ± σ):   49.060 ms ± 818.642 μs  ┊ GC (mean ± σ):  5.77% ± 0.64%

                ▅▅█      ▃▃                                         
      ▄▃▁▅▄▁▅▃▃▄███▅▅▇▃▁▆███▁▃▅▁▃▁▁▁▁▁▁▁▁▁▁▁▃▃▃▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▃ ▃
      47.8 ms         Histogram: frequency by time         52.4 ms <

     Memory estimate: 194.10 MiB, allocs estimate: 655347.

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
We can further also use [`gradient_descent!`](https://manoptjl.org/stable/solvers/gradient_descent/#Manopt.gradient_descent!) to even work inplace of the initial point we pass.

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

    BenchmarkTools.Trial: 179 samples with 1 evaluation.
     Range (min … max):  27.027 ms …  31.367 ms  ┊ GC (min … max): 0.00% … 11.00%
     Time  (median):     27.712 ms               ┊ GC (median):    0.00%
     Time  (mean ± σ):   27.939 ms ± 779.920 μs  ┊ GC (mean ± σ):  0.84% ±  2.56%

             ▄▃▆█▇▄▇                                                
      ▅▁▁▅▅▅▇████████▅▇▆▁▅▁▁▅▁▁▁▅▁▁▁▁▁▁▁▁▁▁▁▁▅▁▁▁▅▁▁▁▁▁▅▆▁▅▅▁▁▁▇▁▇ ▅
      27 ms         Histogram: log(frequency) by time      30.7 ms <

     Memory estimate: 3.76 MiB, allocs estimate: 5949.

which is faster by about a factor of 2 compared to the first solver-call.
Note that the results `m1` and `m2` are of course the same.

``` julia
distance(M, m1, m2)
```

    2.0004809792350595e-10
