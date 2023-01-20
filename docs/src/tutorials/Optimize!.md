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
using Manopt, Manifolds, Random
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

This is nice as a first run and agrees with the [`mean`](https://juliamanifolds.github.io/Manifolds.jl/stable/features/statistics.html#Statistics.mean-Tuple%7BAbstractManifold,%20Vararg%7BAny%7D%7D)

``` julia
m2 = mean(M, data)
distance(M, m1, m2)
```

    0.0018507774286825084

On the other hand we do not get much insight into the run of the solber.
We could add some debug output first

``` julia
m1 = gradient_descent(
  M, f, grad_f, data[1];
  debug = [:Iteration, " ", :Cost, " | ", :Change, :Stop, "\n"],
  )
```

    Initial  F(x): 0.324880 | 
    # 1      F(x): 0.252325 | Last Change: 1.063609
    # 2      F(x): 0.209670 | Last Change: 0.809859
    # 3      F(x): 0.185465 | Last Change: 0.616665
    # 4      F(x): 0.171216 | Last Change: 0.470842
    # 5      F(x): 0.163008 | Last Change: 0.359346
    # 6      F(x): 0.158185 | Last Change: 0.274597
    # 7      F(x): 0.155389 | Last Change: 0.209754
    # 8      F(x): 0.153748 | Last Change: 0.160326
    # 9      F(x): 0.152794 | Last Change: 0.122512
    # 10     F(x): 0.152235 | Last Change: 0.093650
    # 11     F(x): 0.151909 | Last Change: 0.071575
    # 12     F(x): 0.151718 | Last Change: 0.054715
    # 13     F(x): 0.151607 | Last Change: 0.041822
    # 14     F(x): 0.151542 | Last Change: 0.031971
    # 15     F(x): 0.151504 | Last Change: 0.024439
    # 16     F(x): 0.151482 | Last Change: 0.018683
    # 17     F(x): 0.151469 | Last Change: 0.014282
    # 18     F(x): 0.151461 | Last Change: 0.010919
    # 19     F(x): 0.151457 | Last Change: 0.008347
    # 20     F(x): 0.151454 | Last Change: 0.006381
    # 21     F(x): 0.151453 | Last Change: 0.004879
    # 22     F(x): 0.151452 | Last Change: 0.003730
    # 23     F(x): 0.151451 | Last Change: 0.002852
    # 24     F(x): 0.151451 | Last Change: 0.002180
    # 25     F(x): 0.151451 | Last Change: 0.001667
    # 26     F(x): 0.151451 | Last Change: 0.001274
    # 27     F(x): 0.151451 | Last Change: 0.000974
    # 28     F(x): 0.151451 | Last Change: 0.000745
    # 29     F(x): 0.151451 | Last Change: 0.000570
    # 30     F(x): 0.151451 | Last Change: 0.000435
    # 31     F(x): 0.151451 | Last Change: 0.000333
    # 32     F(x): 0.151451 | Last Change: 0.000255
    # 33     F(x): 0.151451 | Last Change: 0.000195
    # 34     F(x): 0.151451 | Last Change: 0.000149
    # 35     F(x): 0.151451 | Last Change: 0.000114
    # 36     F(x): 0.151451 | Last Change: 0.000087
    # 37     F(x): 0.151451 | Last Change: 0.000067
    # 38     F(x): 0.151451 | Last Change: 0.000051
    # 39     F(x): 0.151451 | Last Change: 0.000039
    # 40     F(x): 0.151451 | Last Change: 0.000030
    # 41     F(x): 0.151451 | Last Change: 0.000023
    # 42     F(x): 0.151451 | Last Change: 0.000017
    # 43     F(x): 0.151451 | Last Change: 0.000013
    # 44     F(x): 0.151451 | Last Change: 0.000010
    # 45     F(x): 0.151451 | Last Change: 0.000008
    # 46     F(x): 0.151451 | Last Change: 0.000006
    # 47     F(x): 0.151451 | Last Change: 0.000005
    # 48     F(x): 0.151451 | Last Change: 0.000003
    # 49     F(x): 0.151451 | Last Change: 0.000003
    # 50     F(x): 0.151451 | Last Change: 0.000002
    # 51     F(x): 0.151451 | Last Change: 0.000002
    # 52     F(x): 0.151451 | Last Change: 0.000001
    # 53     F(x): 0.151451 | Last Change: 0.000001
    # 54     F(x): 0.151451 | Last Change: 0.000001
    # 55     F(x): 0.151451 | Last Change: 0.000001
    # 56     F(x): 0.151451 | Last Change: 0.000000
    # 57     F(x): 0.151451 | Last Change: 0.000000
    # 58     F(x): 0.151451 | Last Change: 0.000000
    # 59     F(x): 0.151451 | Last Change: 0.000000
    # 60     F(x): 0.151451 | Last Change: 0.000000
    # 61     F(x): 0.151451 | Last Change: 0.000000
    # 62     F(x): 0.151451 | Last Change: 0.000000
    # 63     F(x): 0.151451 | Last Change: 0.000000
    # 64     F(x): 0.151451 | Last Change: 0.000000
    # 65     F(x): 0.151451 | Last Change: 0.000000
    # 66     F(x): 0.151451 | Last Change: 0.000000
    # 67     F(x): 0.151451 | Last Change: 0.000000
    # 68     F(x): 0.151451 | Last Change: 0.000000
    The algorithm reached approximately critical point after 68 iterations; the gradient norm (4.954832225480352e-10) is less than 1.0e-9.

    3-element Vector{Float64}:
     0.6868392795563908
     0.006531600623587405
     0.7267799820108911

## Literature

Absil, P.-A., Mahony, R. and Sepulchre, R. (2008) *Optimization algorithms on matrix manifolds*. Princeton University Press. Available at: <https://doi.org/10.1515/9781400830244>.

Boumal, N. (2022) *An introduction to optimization on smooth manifolds*. Available at: <https://www.nicolasboumal.net/book>.

do Carmo, M.P. (1992) *Riemannian geometry*. Birkhäuser Boston, Inc., Boston, MA (Mathematics: Theory & applications), p. xiv+300. Available at: <https://doi.org/10.1007/978-1-4757-2201-7>.

Karcher, H. (1977) “Riemannian center of mass and mollifier smoothing,” *Communications on Pure and Applied Mathematics*, 30(5), pp. 509–541. Available at: <https://doi.org/10.1002/cpa.3160300502>.
