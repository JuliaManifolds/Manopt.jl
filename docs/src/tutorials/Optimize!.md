Get Started: Optimize!
================

In this tutorial, we woill both introduce the basics of optimisation on
manifolds as well as how to use [`Manopt.jl`](https://manoptjl.org) to
perform optimisation on manifolds in [Julia](https://julialang.org).

For more theoretical background, see e.g. (Carmo, 1992) for an
introduction to Riemannian manifolds amnd (Absil *et al.*, 2008) or
(Boumal, 2022) to read more about optimisation thereon.

Let $\mathcal M$ denote a [Riemannian
manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)
and let $f\colon \mathcal M → ℝ$ be a cost function. We aim to compute a
point $p^*$ where $f$ is *minimal* or in other words $p^*$ is a
*minimizer* of $f$.

We also write this as

\begin{equation} \_{p ∈ M} f(p) \end{equation\*}

and would like to find $p^*$ numerically. As an example we take the
generalisation of the [(arithemtic)
mean](https://en.wikipedia.org/wiki/Arithmetic_mean). In the Euclidean
case with $d\in\mathbb N$, that is for $n\in \mathbb N$ data points
$y_1,\ldots,y_n \in \mathbb R^d$ the mean

$$
  \sum_{i=1}^n y_i
$$

can not be directly generalised to data $q_1,\ldots,q_n$, since on a
manifold we do not have an addition. But the mean can also be
charcterised as

$$
  \operatorname*{arg\,min}_{x\in\mathbb R^d} \sum_{i=1}^n \lVert x - y_i\rVert^2
$$

and using the Riemannian distance $d_\mathcal M$, this can be written on
Riemannian manifolds. We obtain the *Riemannian Center of Mass*
(Karcher, 1977)

$$
  \operatorname*{arg\,min}_{p\in\mathbb R^d} \sum_{i=1}^n d_{\mathcal M}^2(p, q_i)
$$

## Loading the necessary packages

``` julia
using Manopt, Manifolds, Random
Random.seed!(42);
```

Now assume we are on the [Sphere]() and hace some points measured

``` julia
    n = 100
    σ = π / 8
    M = Sphere(2)
    p = 1 / sqrt(2) * [1.0, 0.0, 1.0]
    data = [exp(M, p,  σ * rand(M; vector_at=p)) for i in 1:n];
```

## Literature

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-AbsilMahonySepulchre2008" class="csl-entry">

<span class="smallcaps">Absil</span>, P.-A., <span
class="smallcaps">Mahony</span>, R., & <span
class="smallcaps">Sepulchre</span>, R. (2008) *[Optimization algorithms
on matrix manifolds](https://doi.org/10.1515/9781400830244)*. Princeton
University Press.

</div>

<div id="ref-Boumal2023" class="csl-entry">

<span class="smallcaps">Boumal</span>, N. (2022) [An introduction to
optimization on smooth manifolds](https://www.nicolasboumal.net/book).

</div>

<div id="ref-doCarmo1992" class="csl-entry">

<span class="smallcaps">Carmo</span>, M.P. do (1992) *[Riemannian
geometry](https://doi.org/10.1007/978-1-4757-2201-7)*, Mathematics:
Theory & applications. Birkhäuser Boston, Inc., Boston, MA.

</div>

<div id="ref-Karcher1977" class="csl-entry">

<span class="smallcaps">Karcher</span>, H. (1977) [Riemannian center of
mass and mollifier smoothing](https://doi.org/10.1002/cpa.3160300502).
*Communications on Pure and Applied Mathematics*, **30**, 509–541.

</div>

</div>
