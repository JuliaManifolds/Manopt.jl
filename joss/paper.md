---
title: 'Manopt.jl: Optimization on Manifolds in Julia'
tags:
  - Julia
  - Riemannian manifolds
  - optimization
  - numerical analysis
authors:
  - name: Ronny Bergmann
    orcid: 0000-0003-0872-7098
    affiliation: 1
affiliations:
 - name: Norwegian University of Science and Technology, Department of Mathematical Sciences, Trondheim, Norway
   index: 1
date: 23 December 2020
bibliography: bibliography.bib

---

# Summary

`Manopt.jl` provides a set of optimization algorithms for optimization problems given on a Riemannian manifold $\mathcal M$.
Based on a generic optimization framework together with the interface `ManifoldsBase.jl` for Riemannian manifolds, classical and recently developed methods are provided in an efficient implementation. Algorithms include the least requiring Particle Swarm and Nelder Mead algorithms as well as a classical gradient or stochastic gradient descent. Furthermore, quasi Newton methods like a Riemannian L-BFGS `HuangGallivanAbsil:2015:1` and nonsmooth optimization algorithms like a Cyclic Proximal Point Algorithm `@Bacak:2014:1`, Douglas-Rachford ' @BergmannPerschSteidl:2016:1` and Chambolle-Pock algorithm `@BergmannHerzogSilvaLouzeiroTenbrinckVidal-Nunez-2020` are provided together with several basic cost functions, gradients and proximal maps.s

# Statement of Need

In many applications and optimization tasks, nonlinear data appears naturally.
For example when data on the sphere is measured, diffusion data can be captures as a signal or multivariate data of symmetric positive definite matrices or orientations like they appear for EBSD data. Another example are fixed rank matrices, as they appear in dictionary learning, which might be incomplete.

Working on these data, for example denoising, inpainting, or performing matrix completion can be phrased as an optimization problem

$$\text{Minimize}\quad f(x) \quad \text{where } x\in\mathcal M, $$

where the optimization problem is phrased on a Riemannian manifold $\mathcal M$.
Further examples can be found in `@AbsilMahonySepulchre:2008:1`, `Boumal:2020:1`.

Using the interface from `ManifoldsBase-jl` the algorithms implemented in the optimization framework can be used with any manifold from `Manifolds.jl`, a library of manifolds.

`Manopt.jl` provides a low level entry to optimization on manifolds while also providing efficient implementations, that can easily be extended to cover own manifolds.

# Functionality

`Manopt.jl` provides a comprehensive framework for optimization on Riemannian manifolds,
including a generic way to specify a step size and a stopping criterion as well as enhance the algorithm with debug and recording capabilities.
Based on this interface a variety

An optimization problem in `Manopt.jl` consists of a `Problem p` and `Options o`,
The `Problem` consists of all static information like the cost function and a potential gradient of the optimization task. The `Options` specify the type of algorithm and the details. For example by default most options specify that the exponential map, which generalizes the notion of addition to the manifold should be used and the algorithm steps are performed following an acceleration free curve on the manifold. This might not be known in closed form for some manifolds and hence also arbitrary retractions can be specified for this instead. This yields approximate algorithms that are numerically more efficient.
Similarly, tangent vectors at different points are identified by vector transport, which by default is the parallel transport.
By providing always a default, the start for a user can start right away and modify these settings to improve speed or specify the retraction to their needs.

Using a decorator pattern, the `Options` can be encapsulated in `DebugOptions` or `RecordOptions` which either print or record arbitrary data stored within the `Options`. This enables to investigate how the optimization is performed in detail and use the algorithms from within this package also for numerical analysis.

# Example

`Manopt.jl` is registered in the general Julia registry and can hence be installed typing `]add Manopt` in Julia REPL.
Given the `Sphere` from `Manifolds.jl` and a set of unit vectors $p_1,...,p_N\in\mathbb R^3$, where $N$ is the number of data points.
we can compute the generalization of the mean, called the Riemannian Center of Mass, which is defined as the minimizer of the squared distances to the given data

```math
\text{Minimize}_{x\in\mathcal M} \displaystyle\sum_{k=1}^Nd_{\mathcal M}(x, p_k)^2,
```

where $d_{\mathcal M}$ denotes the Riemannian distance. For the sphere this distance is given by the length of the shorter great arc connecting the two points.

```julia
using Manopt, Manifolds, LinearAlgebra
M = Sphere(2)
pts = [ normalize(rand(3)) for _ in 1:100 ]

F(y) = sum(1/(2*n) * distance.(Ref(M), pts, Ref(y)).^2)
gradF(y) = sum(1/n grad_distance.(Ref(M), pts, Ref(y)))

xMean = gradient_descent(M, F, gradF, pts[1])
```

In order to print the iteration, the current iterate, change and cost every $50$th iteration as well as the stopping reason and record iteration number, change and cost, these can be specified as optional parameters. These can then be easily accessed using the `get_record` function.

```julia
o = gradient_descent(M, F, gradF, pts[1],
    debug=[:Iteration, " | ", :x, " | ", :Change, " | ", :Cost, "\n", 50, :Stop],
    record=[:Iteration, :Change, :Cost],
)
xMean3 = get_solver_result(o)
values = get_record(o)
```

# Other software

There are two projects that are similar to `Manopt.jl`, though less flexible for example concerning the stopping criteria, which are [`Manopt`](https://manopt.org) in Matlab and [`pymanopt`](https://pymanopt.org) in Python.
Similarly [`ROPTLIB`](https://www.math.fsu.edu/~whuang2/Indices/index_ROPTLIB.html) is a package for optimization on Manifolds in C++.
While all three packages cover some algorithm, most are less flexible for example in stating the stopping criterion, which is fixed to mainly maximal number of iterations or a small gradient. Most prominently, `Manopt.jl` is the first package that also covers methods for high-performance and high-dimensional nonsmooth optimization on manifolds.

# Quality control and contributions

`Manopt.jl` uses GitHub Actions and a continuous integration testing with the most recent version of Julia all supported versions of Julia
 on macOS, Linux and Windows. The tests cover most of the library and can also be run locally in Julia REPL with `]test Manopt`
Support and submission of contributions to the library are handled through the GitHub repository via issues or by pull requests

# Requirements

`Manopt.jl` is based on `ManifoldsBase.jl`.
This package is available in the general Julia registry. Using it together with `Manifolds.jl` is recommended, since this package provides a library of manifolds.

# References