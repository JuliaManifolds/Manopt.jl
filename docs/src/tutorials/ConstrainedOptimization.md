
# How to do Constrained Optimization

This tutorial is a short introduction to using solvers for constraint optimisation in [`Manopt.jl`](https://manoptjl.org).

## Introduction

A constraint optimisation problem is given by

``` math
\tag{P}
\begin{align*}
\operatorname*{arg\,min}_{p\in\mathcal M} & f(p)\\
\text{such that} &\quad g(p) \leq 0\\
&\quad h(p) = 0,\\
\end{align*}
```

where $f\colon \mathcal M → ℝ$ is a cost function, and $g\colon \mathcal M → ℝ^m$ and $h\colon \mathcal M → ℝ^n$ are the inequality and equality constraints, respectively. The $\leq$ and $=$ in (P) are meant elementwise.

This can be seen as a balance between moving constraints into the geometry of a manifold $\mathcal M$ and keeping some, since they can be handled well in algorithms, see (Bergmann and Herzog, 2019), (Liu and Boumal, 2019) for details.

``` julia
using Distributions, LinearAlgebra, Manifolds, Manopt, Random
Random.seed!(12345);
```

In this tutorial we want to look at different ways to specify the problem and its implications. We start with specifying an example problems to illustrayte the different available forms.

We will consider the problem of a Nonnegative PCA, cf. Section 5.1.2 in (Liu and Boumal, 2019):

let $v_0 ∈ ℝ^d$, $\lVert v_0 \rVert=1$ be given spike signal, that is a signal that is sparse with only $s=\lfloor δd \rfloor$ nonzero entries.

``` math
  Z = \sqrt{σ} v_0v_0^{\mathrm{T}}+N,
```

where $\sigma$ is a signal-to-noise ratio and $N$ is a matrix with random entries, where the diagonal entries are distributed with zero mean and standard deviation $1/d$ on the off-diagonals and $2/d$ on the daigonal

``` julia
d = 150; # dimension of v0
σ = 0.1^2; # SNR
δ = 0.1; s = Int(floor(δ * d)); # Sparsity
S = sample(1:d, s; replace=false);
v0 =  [i ∈ S ? 1 / sqrt(s) : 0.0 for i in 1:d];
N = rand(Normal(0, 1 / d), (d, d)); N[diagind(N, 0)] .= rand(Normal(0, 2 / d), d);
Z = Z = sqrt(σ) * v0 * transpose(v0) + N;
```

In order to recover $v_0$ we consider the constrained optimisation problem on the sphere $\mathcal S^{d-1}$ given by

``` math
\begin{align*}
\operatorname*{arg\,min}_{p\in\mathcal S^{d-1}} & -p^{\mathrm{T}}Zp^{\mathrm{T}}\\
\text{such that} &\quad p \geq 0\\
\end{align*}
```

or in the previous notation $f(p) = -p^{\mathrm{T}}Zp^{\mathrm{T}}$ and $g(p) = -p$. We first initialize the manifold under consideration

``` julia
M = Sphere(d - 1)
```

    Sphere(149, ℝ)

## A first Augmented Lagrangian Run

We first defined $f$ and $g$ as usual functions

``` julia
f(M, p) = -transpose(p) * Z * p;
g(M, p) = -p;
```

since $f$ is a functions defined in the embedding $ℝ^d$ as well, we obtain its gradient by projection.

``` julia
grad_f(M, p) = project(M, p, -transpose(Z) * p - Z * p);
```

For the constraints this is a little more involved, since each function $g_i = g(p)_i = p_i$ has to return its own gradient. These are again in the embedding just $\operatorname{grad} g_i(p) = -e_i$ the $i$ th unit vector. We can project these again onto the tangent space at $p$:

``` julia
grad_g(M, p) = project.(
    Ref(M), Ref(p), [[i == j ? -1.0 : 0.0 for j in 1:d] for i in 1:d]
);
```

We further start in a random point:

``` julia
p0 = rand(M);
```

Let’s check a few things for the initial point

``` julia
f(M, p0)
```

    0.0006573926605877413

How much the function g is positive

``` julia
maximum(g(M, p0))
```

    0.20493076865295534

Now as a first method we can just call the [Augmented Lagrangian Method](https://manoptjl.org/stable/solvers/augmented_Lagrangian_method/) with a simple call:

``` julia
@time v1 = augmented_Lagrangian_method(
    M, f, grad_f, p0; g=g, grad_g=grad_g,
    debug=[:Iteration, :Cost, :Stop, " | ", :Change, 50, "\n"],
);
```

    Initial F(x): 0.000657 | 

    # 50    F(x): -0.120344 | Last Change: 1.049035
    # 100   

    F(x): -0.119963 | Last Change: 0.008114
    The value of the variable (ϵ) is smaller than or equal to its threshold (1.0e-6).
    The algorithm performed a step with a change (0.0) less than 1.0e-6.
      2.205239 seconds (9.40 M allocations: 8.190 GiB, 27.54% gc time, 11.29% compilation time: 88% of which was recompilation)

Now we have both a lower function value and the point is nearly within the constraints, … up to numerical inaccuracies

``` julia
f(M, v1)
```

    -0.11929165607642694

``` julia
maximum( g(M, v1) )
```

    -3.5602423483217516e-13

## A faster Augmented Lagrangian Run

Now this is a little slow, so we can modify two things, that we will directly do both – but one could also just change one of these – :

1.  Gradients should be evaluated in place, so for example

``` julia
grad_f!(M, X, p) = project!(M, X, p, -transpose(Z) * p - Z * p);
```

2.  The constraints are currently always evaluated all together, since the function `grad_g` always returns a vector of gradients.
    We first change the constraints function into a vector of functions.
    We further change the gradient *both* into a vector of gradient functions $\operatorname{grad} g_i, i=1,\ldots,d$, *as well as* gradients that are computed in place.

``` julia
g2 = [(M, p) -> -p[i] for i in 1:d];
grad_g2! = [
    (M, X, p) -> project!(M, X, p, [i == j ? -1.0 : 0.0 for j in 1:d]) for i in 1:d
];
```

We obtain

``` julia
@time v2 = augmented_Lagrangian_method(
        M, f, grad_f!, p0; g=g2, grad_g=grad_g2!, evaluation=InplaceEvaluation(),
        debug=[:Iteration, :Cost, :Stop, " | ", :Change, 50, "\n"],
    );
```

    Initial F(x): 0.000657 | 

    # 50    F(x): -0.120343 | Last Change: 1.049289
    # 100   

    F(x): -0.120250 | Last Change: 0.028941

    # 150   F(x): NaN | Last Change: NaN
    # 200   

    F(x): NaN | Last Change: NaN

    # 250   F(x): NaN | Last Change: NaN
    # 300   

    F(x): NaN | Last Change: NaN
    The algorithm reached its maximal number of iterations (300).
     12.231798 seconds (13.97 M allocations: 49.419 GiB, 24.86% gc time, 3.65% compilation time)

As a technical remark: Note that (by default) the change to [`InplaceEvaluation`](https://manoptjl.org/stable/plans/problem/#Manopt.InplaceEvaluation)s affects both the constrained solver as well as the inner solver of the subproblem in each iteration.

``` julia
f(M, v2)
```

    NaN

``` julia
maximum(g(M, v2))
```

    NaN

These are the very similar to the previous values but the solver took much less time and less memory allocations.

## Exact Penalty Method

As a second solver, we have the [Exact Penalty Method](https://manoptjl.org/stable/solvers/exact_penalty_method/), which currenlty is available with two smoothing variants, which make an inner solver for smooth optimisationm, that is by default again \[quasi Newton\] possible:
[`LogarithmicSumOfExponentials`](https://manoptjl.org/stable/solvers/exact_penalty_method/#Manopt.LogarithmicSumOfExponentials)
and [`LinearQuadraticHuber`](https://manoptjl.org/stable/solvers/exact_penalty_method/#Manopt.LinearQuadraticHuber). We compare both here as well. The first smoothing technique is the default, so we can just call

``` julia
@time v3 = exact_penalty_method(
    M, f, grad_f!, p0; g=g2, grad_g=grad_g2!, evaluation=InplaceEvaluation(),
    debug=[:Iteration, :Cost, :Stop, " | ", :Change, 50, "\n"],
);
```

    Initial F(x): 0.000657 | 

    # 50    F(x): -0.119609 | Last Change: 1.030083
    # 100   

    F(x): -0.120341 | Last Change: 0.014608
    The value of the variable (ϵ) is smaller than or equal to its threshold (1.0e-6).
    The algorithm performed a step with a change (2.580956827951785e-8) less than 1.0e-6.
      1.504211 seconds (6.25 M allocations: 3.839 GiB, 25.26% gc time, 31.39% compilation time)

We obtain a similar cost value as for the Augmented Lagrangian Solver above, but here the constraint is actually fulfilled and not just numerically “on the boundary”.

``` julia
f(M, v3)
```

    -0.1203415430414448

``` julia
maximum(g(M, v3))
```

    -3.76010458767384e-6

The second smoothing technique is often beneficial, when we have a lot of constraints (in the above mentioned vectorial manner), since we can avoid several gradient evaluations for the constraint functions here. This leads to a faster iteration time.

``` julia
@time v4 = exact_penalty_method(
    M, f, grad_f!, p0; g=g2, grad_g=grad_g2!,
    evaluation=InplaceEvaluation(),
    smoothing=LinearQuadraticHuber(),
    debug=[:Iteration, :Cost, :Stop, " | ", :Change, 50, "\n"],
);
```

    Initial F(x): 0.000657 | 

    # 50    F(x): -0.120346 | Last Change: 0.008883
    # 100   

    F(x): -0.120344 | Last Change: 0.000161
    The value of the variable (ϵ) is smaller than or equal to its threshold (1.0e-6).
    The algorithm performed a step with a change (5.771194914292421e-8) less than 1.0e-6.
      0.683839 seconds (2.73 M allocations: 687.442 MiB, 6.75% gc time, 59.86% compilation time)

For the result we see the same behaviour as for the other smoothing.

``` julia
f(M, v4)
```

    -0.12034370942770528

``` julia
maximum(g(M, v4))
```

    2.3337667948104105e-8

## Comparing to the unconstraint solver

We can compare this to the *global* optimum on the sphere, which is the unconstraint optimisation problem; we can just use Quasi Newton.

Note that this is much faster, since every iteration of the algorithms above does a quasi-Newton call as well.

``` julia
@time w1 = quasi_Newton(
    M, f, grad_f!, p0; evaluation=InplaceEvaluation()
);
```

      0.062299 seconds (154.22 k allocations: 32.308 MiB, 93.42% compilation time: 100% of which was recompilation)

``` julia
f(M, w1)
```

    -0.1340215368738348

But for sure here the constraints here are not fulfilled and we have veru positive entries in $g(w_1)$

``` julia
maximum(g(M, w1))
```

    0.2719372729760305

## Literature

Bergmann, R. and Herzog, R. (2019) “Intrinsic formulation of KKT conditions and constraint qualifications on smooth manifolds,” *SIAM Journal on Optimization*, 29(4), pp. 2423–2444. Available at: <https://doi.org/10.1137/18M1181602>.

Liu, C. and Boumal, N. (2019) “Simple algorithms for optimization on Riemannian manifolds with constraints,” *Applied Mathematics & Optimization* \[Preprint\]. Available at: <https://doi.org/10.1007/s00245-019-09564-3>.
