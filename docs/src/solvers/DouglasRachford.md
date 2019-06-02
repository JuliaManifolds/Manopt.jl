# [Douglas–Rachford Algorithm](@id DRSolver)

The (Parallel) Douglas–Rachford ((P)DR) Algorithm was generalized to Hadamard
manifolds in [[Bergmann, Persch, Steidl, 2016](#BergmannPerschSteidl2016)].

The aim is to minimize the sum

$F(x) = f(x) + g(x)$

on a manifold, where the two summands have proximal maps
$\operatorname{prox}_{\lambda f}, \operatorname{prox}_{\lambda g}$ that are easy
to evaluate (maybe in closed form or not too costly to approximate).
Further define the Reflection operator at the proximal map as

$\operatorname{refl}_{\lambda f}(x) = \exp_{\operatorname{prox}_{\lambda f}(x)} \bigl( -\log_{\operatorname{prox}_{\lambda f}(x)} x \bigr)$.

Let $\alpha_k\in [0,1]$ with $\sum_{k\in\mathbb N} \alpha_k(1-\alpha_k) = \infty$
and $\lambda > 0$ which might depend on iteration $k$ as well) be given.

Then the (P)DRA algorithm for initial data $x_0\in\mathcal H$ as

## Initialization

Initialize $t_0 = x_0$ and $k=0$

## Iteration

Repeat  until a convergence criterion is reached

1. Compute $s_k = \operatorname{refl}_{\lambda f}\operatorname{refl}_{\lambda g}(t_k)$
2. within that operation store $x_{k+1} = \operatorname{prox}_{\lambda g}(t_k)$ which is the prox the inner reflection reflects at.
3. Compute $t_{k+1} = g(\alpha_k; t_k, s_k)$
4. Set $k = k+1$

## Result

The result is given by the last computed $x_K$.

For the parallel version, the first proximal map is a vectorial version, where
in each component one prox is applied to the corresponding copy of $t_k$ and
the second proximal map corresponds to the indicator function of the set,
where all copies are equal (in $\mathcal H^n$, where $n$ is the number of copies),
leading to the second prox being the Riemannian mean.

## Interface

```@docs
  DouglasRachford
```

## Options

```@docs
DouglasRachfordOptions
```

For specific [`DebugAction`](@ref)s and [`RecordAction`](@ref)s see also
[Cyclic Proximal Point](@ref CPPSolver).

## Literature

```@raw html
<ul>
<li id="BergmannPerschSteidl2016">[<a>Bergmann, Persch, Steidl, 2016</a>]
  Bergmann, R; Persch, J.; Steidl, G.: <emph>A Parallel Douglas–Rachford
  Algorithm for Minimizing ROF-like Functionals on Images with Values in
  Symmetric Hadamard Manifolds.</emph>
  SIAM Journal on Imaging Sciences, Volume 9, Number 3, pp. 901–937, 2016.
  doi: <a href="https://doi.org/10.1137/15M1052858">10.1137/15M1052858</a>,
  arXiv: <a href="https://arxiv.org/abs/1512.02814">1512.02814</a>.
</li>
</ul>
```
