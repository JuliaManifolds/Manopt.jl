Get Started: Optimize!
================

In this tutorial, we will both introduce the basics of optimisation on manifolds as well as
how to use [`Manopt.jl`](https://manoptjl.org) to perform optimisation on manifolds in [Julia](https://julialang.org).

For more theoretical background, see e.g. (do Carmo, 1992) for an introduction to Riemannian manifolds
and (Absil, Mahony and Sepulchre, 2008) or (Boumal, 2022) to read more about optimisation thereon.

Let $\mathcal M$ denote a [Riemannian manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)
and let $f\colon \mathcal M → ℝ$ be a cost function.
We aim to compute a point $p^*$ where $f$ is *minimal* or in other words $p^*$ is a *minimizer* of $f$.

We also write this as

``` math
    \operatorname*{arg\,min}_{p ∈ \mathcal M} f(p)
```

and would like to find $p^*$ numerically.
As an example we take the generalisation of the [(arithemtic) mean](https://en.wikipedia.org/wiki/Arithmetic_mean).
In the Euclidean case with$d\in\mathbb N$, that is for $n\in \mathbb N$ data points $y_1,\ldots,y_n \in \mathbb R^d$ the mean

``` math
  \sum_{i=1}^n y_i
```

can not be directly generalised to data $q_1,\ldots,q_n$, since on a manifold we do not have an addition.
But the mean can also be charcterised as

``` math
  \operatorname*{arg\,min}_{x\in\mathbb R^d} \frac{1}{2n}\sum_{i=1}^n \lVert x - y_i\rVert^2
```

and using the Riemannian distance $d_\mathcal M$, this can be written on Riemannian manifolds. We obtain the *Riemannian Center of Mass* (Karcher, 1977)

``` math
  \operatorname*{arg\,min}_{p\in\mathbb R^d}
  \frac{1}{2n} \sum_{i=1}^n d_{\mathcal M}^2(p, q_i)
```

Fortunately the gradient can be computed and is

``` math
  \operatorname*{arg\,min}_{p\in\mathbb R^d} \frac{1}{n} \sum_{i=1}^n -\log_p q_i
```

## Loading the necessary packages

Let’s assume you have already installed both Manotp and Manifolds in Julia (using e.g. `using Pkg; Pkg.add(["Manopt", "Manifolds"])`).
Then we can get started by loading both packages – and `Random` for persistency in this tutorial.

``` julia
using Manopt, Manifolds, Random, LinearAlgebra
Random.seed!(42);
```

Now assume we are on the [Sphere](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/sphere.html)
$\mathcal M = \mathbb S^2$ and we generate some random points “around” some initial point $p$

``` julia
n = 100
σ = π / 8
M = Sphere(2)
p = 1 / sqrt(2) * [1.0, 0.0, 1.0]
data = [exp(M, p,  σ * rand(M; vector_at=p)) for i in 1:n];
```

Now we can define the cost function $f$ and its (Riemannian) gradient $\operatorname{grad} f$
for the Riemannian center of mass:

``` julia
f(M, p) = sum(1 / (2 * n) * distance.(Ref(M), Ref(p), data) .^ 2)
grad_f(M, p) = sum(1 / n * grad_distance.(Ref(M), data, Ref(p)));
```

and just call [`gradient_descent`](https://manoptjl.org/stable/solvers/gradient_descent/).
For a first start, we do not have to provide more than the manifold, the cost, the gradient,
and a startig point, which we just set to the first data point

``` julia
m1 = gradient_descent(M, f, grad_f, data[1])
```

    3-element Vector{Float64}:
     0.6868392795563908
     0.006531600623587405
     0.7267799820108911

In order to get more details, we further add the `debug=` keyword argument, which
act as a [decorator pattern](https://en.wikipedia.org/wiki/Decorator_pattern).

This way we can easily specify a certain debug to be printed.
The goal is to get an output of the form

``` {shell}
# i | Last Change: [...] | F(x): [...] |
```

but where we also want to fix the display format for the change and the cost
numbers (the `[...]`) to have a certain format. Furthermore, the reason why the solver stopped should be printed at the end

These can easily be specified using either a Symbol – using the default format for numbers – or a tuple of a symbol and a format-string in the `debug=` keyword that is avaiable for every solver.
We can also – for illustration reasons – just look at the first 6 steps by setting a [`stopping_criterion=`](https://manoptjl.org/stable/plans/stopping_criteria/)

``` julia
m2 = gradient_descent(M, f, grad_f, data[1];
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"),
        (:Cost, " F(x): %1.11f | "), "\n", :Stop],
    stopping_criterion = StopAfterIteration(6)
  )
```

    Initial  F(x): 0.32487988924 |
    # 1     |Δp|: 1.063609017 | F(x): 0.25232524046 |
    # 2     |Δp|: 0.809858671 | F(x): 0.20966960102 |
    # 3     |Δp|: 0.616665145 | F(x): 0.18546505598 |
    # 4     |Δp|: 0.470841764 | F(x): 0.17121604104 |
    # 5     |Δp|: 0.359345690 | F(x): 0.16300825911 |
    # 6     |Δp|: 0.274597420 | F(x): 0.15818548927 |
    The algorithm reached its maximal number of iterations (6).

    3-element Vector{Float64}:
      0.7533872481682506
     -0.060531070555836286
      0.6547851890466333

See [here](https://manoptjl.org/stable/plans/debug/#Manopt.DebugActionFactory-Tuple%7BSymbol%7D) for the list of available symbols.

!!! info "Technical Detail"
    The `debug=` keyword is actually a list of [`DebugActions`](https://manoptjl.org/stable/plans/debug/#Manopt.DebugAction) added to every iteration, allowing you to write your own ones even. Additionally, `:Stop` is an action added to the end of the solver to display the reason why the solver stopped.

The default stopping criterion for [`gradient_descent`](https://manoptjl.org/stable/solvers/gradient_descent/) is, to either stopwhen the gradient is small (`<1e-9`) or a max number of iterations is reached (as a fallback.
Combining stopping-criteria can be done by `|` or `&`.
We further pass a number `25` to `debug=` to only an output every `25`th iteration:

``` julia
m3 = gradient_descent(M, f, grad_f, data[1];
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"),
        (:Cost, " F(x): %1.11f | "), "\n", :Stop, 25],
    stopping_criterion = StopWhenGradientNormLess(1e-14) | StopAfterIteration(400),
)
```

    Initial  F(x): 0.32487988924 |
    # 25    |Δp|: 0.459715605 | F(x): 0.15145076374 |
    # 50    |Δp|: 0.000551270 | F(x): 0.15145051509 |
    # 75    |Δp|: 0.000000674 | F(x): 0.15145051509 |
    The algorithm reached approximately critical point after 75 iterations; the gradient norm (2.2455867775217738e-15) is less than 1.0e-14.

    3-element Vector{Float64}:
     0.6868392794788665
     0.006531600680779426
     0.7267799820836411

We can finally use another way to determine the stepsize, for example
a little more expensive [`ArmijoLineSeach`](https://manoptjl.org/stable/plans/stepsize/#Manopt.ArmijoLinesearch) than the default [stepsize](https://manoptjl.org/stable/plans/stepsize/) rule used on the Sphere.

``` julia
m4 = gradient_descent(M, f, grad_f, data[1];
    debug=[:Iteration,(:Change, "|Δp|: %1.9f |"),
        (:Cost, " F(x): %1.11f | "), "\n", :Stop, 2],
      stepsize = ArmijoLinesearch(M; contraction_factor=0.999, sufficient_decrease=0.5),
    stopping_criterion = StopWhenGradientNormLess(1e-14) | StopAfterIteration(400),
)
```

    Initial  F(x): 0.32487988924 |
    # 2     |Δp|: 0.001318138 | F(x): 0.15145051509 |
    # 4     |Δp|: 0.000000021 | F(x): 0.15145051509 |
    # 6     |Δp|: 0.000000021 | F(x): 0.15145051509 |
    The algorithm reached approximately critical point after 7 iterations; the gradient norm (2.014814589152431e-15) is less than 1.0e-14.

    3-element Vector{Float64}:
     0.6868392794788668
     0.006531600680779305
     0.7267799820836411

Then we reach approximately the same point as in the previous run, but in far less steps

``` julia
[f(M, m3)-f(M,m4), distance(M, m3, m4)]
```

    2-element Vector{Float64}:
     1.942890293094024e-16
     2.9802322387695312e-8

## Example 2: Computing the median of symmetric positive definite matrices.

For the second example let’s consider the manifold of [$3 × 3$ symmetric positive definite matrices](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/symmetricpositivedefinite.html) and again 100 random points

``` julia
N = SymmetricPositiveDefinite(3)
m = 100
σ = 0.005
q = Matrix{Float64}(I, 3, 3)
data2 = [exp(N, q, σ * rand(N; vector_at=q)) for i in 1:m];
```

Instead of the mean, let’s consider a non-smooth optimisation task:
The median can be generalized to Manifolds as the minimiser of the sum of distances, see e.g. (Bačák, 2014). We define

``` julia
g(N, q) = sum(1 / (2 * m) * distance.(Ref(N), Ref(q), data2))
```

    g (generic function with 1 method)

Since the function is non-smooth, we can not use a gradient-based approach.
But since for every summand the [proximal map](https://manoptjl.org/stable/functions/proximal_maps/#Manopt.prox_distance) is available, we can use the
[cyclic proximal point algorithm (CPPA)](https://manoptjl.org/stable/solvers/cyclic_proximal_point/). We hence define the vector of proximal maps as

``` julia
proxes_g = Function[(N, λ, q) -> prox_distance(N, λ / m, di, q, 1) for di in data2];
```

Besides also looking at a some debug prints, we can also easily record these values. Similarly to `debug=`, `record=` also accepts Symbols, see list [here](https://manoptjl.org/stable/plans/record/#Manopt.RecordFactory-Tuple%7BAbstractManoptSolverState,%20Vector%7D), to indicate things to record. We further set `return_state` to true to obtain not just the (approximate) minimizer.

``` julia
s = cyclic_proximal_point(N, g, proxes_g, data2[1];
  debug=[:Iteration," | ",:Change," | ",(:Cost, "F(x): %1.12f"),"\n", 1000, :Stop,
        ],
        record=[:Iteration, :Change, :Cost, :Iterate],
        return_state=true,
    );
```

    Initial  |  | F(x): 0.004628871966
    # 1000

     | Last Change: 0.005022 | F(x): 0.003271791321
    # 2000

     | Last Change: 0.000013 | F(x): 0.003271774443
    # 3000

     | Last Change: 0.000004 | F(x): 0.003271771311
    # 4000

     | Last Change: 0.000002 | F(x): 0.003271770214
    # 5000

     | Last Change: 0.000001 | F(x): 0.003271769706
    The algorithm reached its maximal number of iterations (5000).

!!!note "Technical Detail"
   The recording is realised by [`RecordActions`](https://manoptjl.org/stable/plans/record/#Manopt.RecordAction) that are (also) executed at every iteration. These can also be individually implemented and added to the `record=` array instead of symbols.

First, the computed median can be accessed as

``` julia
median = get_solver_result(s)
```

    3×3 Matrix{Float64}:
     1.00054      6.1464e-5    0.000343679
     6.1464e-5    1.00007      0.000101879
     0.000343679  0.000101879  1.00044

but we can also look at the recorded values. For simplicity (of output), lets just look
at the recorded values at iteration 42

``` julia
get_record(s)[42]
```

    (42, 7.640883870061361e-6, 0.0032798026498283275, [1.0002117366518994 0.00014357556816643746 0.00028004648707080637; 0.00014357556816643746 0.9999321794912776 0.00029370555482273464; 0.0002800464870708619 0.0002937055548227485 1.000391810942462])

But we can also access whole serieses and see that the cost does not decrease that fast; actually, the CPPA might converge relatively slow. For that we can for
example access the `:Cost` that was recorded every `:Iterate` as well as the (maybe a little boring) `:Iteration`-number in a semilogplot.

``` julia
x = get_record(s, :Iteration, :Iteration)
y = get_record(s, :Iteration, :Cost)
using Plots
plot(x,y,xaxis=:log, label="CPPA Cost")
```

![](Optimize!_files/figure-commonmark/cell-17-output-1.png)

## Literature

Absil, P.-A., Mahony, R. and Sepulchre, R. (2008) *Optimization algorithms on matrix manifolds*. Princeton University Press. Available at: <https://doi.org/10.1515/9781400830244>.

Bačák, M. (2014) “Computing medians and means in Hadamard spaces,” *SIAM Journal on Optimization*, 24(3), pp. 1542–1566. Available at: <https://doi.org/10.1137/140953393>.

Boumal, N. (2022) *An introduction to optimization on smooth manifolds*. Available at: <https://www.nicolasboumal.net/book>.

do Carmo, M.P. (1992) *Riemannian geometry*. Birkhäuser Boston, Inc., Boston, MA (Mathematics: Theory & applications), p. xiv+300. Available at: <https://doi.org/10.1007/978-1-4757-2201-7>.

Karcher, H. (1977) “Riemannian center of mass and mollifier smoothing,” *Communications on Pure and Applied Mathematics*, 30(5), pp. 509–541. Available at: <https://doi.org/10.1002/cpa.3160300502>.
