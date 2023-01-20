Get Started: Optimize!
================

In this tutorial, we will both introduce the basics of optimisation on
manifolds as well as how to use [`Manopt.jl`](https://manoptjl.org) to
perform optimisation on manifolds in [Julia](https://julialang.org).

For more theoretical background, see e.g. (do Carmo 1992) for an
introduction to Riemannian manifolds and (Absil, Mahony, and Sepulchre
2008) or (Boumal 2022) to read more about optimisation thereon.

Let$\mathcal M$ denote a [Riemannian
manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)
and let$f\colon \mathcal M → ℝ$ be a cost function. We aim to compute a
point$p^*$ where$f$ is *minimal* or in other words$p^*$ is a *minimizer*
of$f$.

We also write this as

``` math
    \operatorname*{arg\,min}_{p ∈ \mathcal M} f(p)
```

and would like to find$p^*$ numerically. As an example we take the
generalisation of the [(arithemtic)
mean](https://en.wikipedia.org/wiki/Arithmetic_mean). In the Euclidean
case with$d\in\mathbb N$, that is for$n\in \mathbb N$ data
points$y_1,\ldots,y_n \in \mathbb R^d$ the mean

``` math
  \sum_{i=1}^n y_i
```

can not be directly generalised to data$q_1,\ldots,q_n$, since on a
manifold we do not have an addition. But the mean can also be
charcterised as

``` math
  \operatorname*{arg\,min}_{x\in\mathbb R^d} \frac{1}{2}\sum_{i=1}^n \lVert x - y_i\rVert^2
```

and using the Riemannian distance$d_\mathcal M$, this can be written on
Riemannian manifolds. We obtain the *Riemannian Center of Mass* (Karcher
1977)

``` math
  \operatorname*{arg\,min}_{p\in\mathbb R^d} \sum_{i=1}^n -\log_p q_i
```math

Luckily the gradient can be computed and is

## Loading the necessary packages



::: {.cell execution_count=2}
``` {.julia .cell-code}
using Manopt, Manifolds, Random
Random.seed!(42);
```

:::

Now assume we are on the
[Sphere](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/sphere.html)$\mathcal M = \mathbb S^2$
and we generate some random points “around” some initial point $p$

``` julia
    n = 100
    σ = π / 8
    M = Sphere(2)
    p = 1 / sqrt(2) * [1.0, 0.0, 1.0]
    data = [exp(M, p,  σ * rand(M; vector_at=p)) for i in 1:n];
```

Now we can define the cost function $f$ and its (Riemannian) gradient
$\operatorname{grad} f$ for the Riemannian center of mass:

``` julia
f(M, p) = sum(1 / (2 * n) * distance.(Ref(M), Ref(p), data) .^ 2)
grad_f(M, p) = sum(1 / n * grad_distance.(Ref(M), data, Ref(p)));
```

and just call [`gradient_descent`](). For a first start, we do not have
to provide more than the manifold, the cost, the gradient, and a startig
point, which we just set to the first data point

``` julia
m1 = gradient_descent(M, f, grad_f, data[1])
```

    3-element Vector{Float64}:
     0.6868392795563908
     0.006531600623587405
     0.7267799820108911

## Literature

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-AbsilMahonySepulchre2008" class="csl-entry">

Absil, P.-A., R. Mahony, and R. Sepulchre. 2008. *Optimization
Algorithms on Matrix Manifolds*. Princeton University Press.
<https://doi.org/10.1515/9781400830244>.

</div>

<div id="ref-Boumal2023" class="csl-entry">

Boumal, Nicolas. 2022. *An Introduction to Optimization on Smooth
Manifolds*. <https://www.nicolasboumal.net/book>.

</div>

<div id="ref-doCarmo1992" class="csl-entry">

Carmo, Manfredo Perdigão do. 1992. *Riemannian Geometry*. Mathematics:
Theory & Applications. Birkhäuser Boston, Inc., Boston, MA.
<https://doi.org/10.1007/978-1-4757-2201-7>.

</div>

<div id="ref-Karcher1977" class="csl-entry">

Karcher, Hermann. 1977. “Riemannian Center of Mass and Mollifier
Smoothing.” *Communications on Pure and Applied Mathematics* 30 (5):
509–41. <https://doi.org/10.1002/cpa.3160300502>.

</div>

</div>
