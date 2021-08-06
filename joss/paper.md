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
date: 22 July 2021
bibliography: bibliography.bib

---

# Summary

[`Manopt.jl`](https://manoptjl.org) provides a set of optimization algorithms for optimization problems given on a Riemannian manifold $\mathcal M$.
Based on a generic optimization framework together with the interface `ManifoldsBase.jl` for Riemannian manifolds, classical and recently developed methods are provided in an efficient implementation. Algorithms include the derivative free Particle Swarm and Nelder Mead algorithms as well as a classical gradient, conjugate gradient and stochastic gradient descent. Furthermore, quasi Newton methods like a Riemannian L-BFGS [@HuangGallivanAbsil:2015:1] and nonsmooth optimization algorithms like a Cyclic Proximal Point Algorithm [@Bacak:2014:1], a (parallel) Douglas-Rachford algorithm [@BergmannPerschSteidl:2016:1] and a Chambolle-Pock algorithm [@BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez:2021:1] are provided together with several basic cost functions, gradients and proximal maps as well as debug and record capabilities.

# Statement of Need

In many applications and optimization tasks, nonlinear data appears naturally.
For example when data on the sphere is measured, diffusion data can be captured as a signal or even multivariate data of symmetric positive definite matrices, and orientations like they appear for electron backscattered diffraction (EBSD) data. Another example are fixed rank matrices, appearing in dictionary learning.
Working on these data, for example doing data interpolation, data approximation, denoising, inpainting, or performing matrix completion, can usually phrased as an optimization problem

$$ \text{Minimize}\quad f(x) \quad \text{where } x\in\mathcal M, $$

where the optimization problem is phrased on a Riemannian manifold $\mathcal M$.

A main challenge of these algorithms is that, compared to the (classical) Euclidean case, there is no addition available. For example on the unit sphere $\mathbb S^2$ of unit vectors in $\mathbb R^3$, adding two vectors of unit lengths yields a vector that is not of unit norm.
The resolution is to generalize the notion of a shortest path from the straight line to what is called a (shortest) geodesic, or acceleration free curves.
In the same sense, other features and properties have to be rephrased and generalized, when performing optimization on a Riemannian manifold.
Algorithms to perform the optimization can still often be stated in the generic form, i.e. on an arbitrary Riemannian manifold $\mathcal M$.

Further examples and a thorough introduction can be found in [@AbsilMahonySepulchre:2008:1], [@Boumal:2020:1].

For a user facing an optimization problem on a manifold, there are two obstacles to the actual numerical optimization: on the one hand, a suitable implementation of the manifold at hand is required, for example how to evaluate the above mentioned geodesics. On the other hand, an implementation of the optimization algorithm that employs said methods from the manifold, such that the algorithm can be applied to the cost function $f$ a user already has.

Using the interface for manifolds, `ManifoldsBase.jl`, the algorithms are implemented in the optimization framework can therefore be used with any manifold from `Manifolds.jl` [@AxenBaranBergmannRzecki:2021:1], a library of efficiently implemented Riemannian manifolds.
`Manopt.jl` provides a low level entry to optimization on manifolds while also providing efficient implementations, that can easily be extended to cover own manifolds.

# Functionality

`Manopt.jl` provides a comprehensive framework for optimization on Riemannian manifolds and a variety of algorithms using this framework.
The framework includes a generic way to specify a step size and a stopping criterion as well as enhance the algorithm with debug and recording capabilities.
Each of the algorithms has a high level interface to make it easy to use the algorithms directly.

An optimization task in `Manopt.jl` consists of a `Problem p` and `Options o`,
The `Problem` consists of all static information like the cost function and a potential gradient of the optimization task. The `Options` specify the type of algorithm and the settings and data required to run the algorithm. For example by default most options specify that the exponential map, which generalizes the notion of addition to the manifold should be used and the algorithm steps are performed following an acceleration free curve on the manifold. This might not be known in closed form for some manifolds and hence also more generally arbitrary retractions can be specified for this instead. This yields approximate algorithms that are numerically more efficient.
Similarly, tangent vectors at different points are identified by a vector transport, which by default is the parallel transport.
By providing always a default, a user can start right away without thinking about these details. They can then modify these settings to improve speed or accuracy by specifying other retractions or vector transport to their needs.

The main methods to implement for an own solver are the `initialize_solver!(p,o)` which should fill the data in the options with an initial state. The second method to implement is the `step_solver!(p,o,i)` performing the $i$th iteration.

Using a decorator pattern, the `Options` can be encapsulated in `DebugOptions` and `RecordOptions` which print and record arbitrary data stored within the `Options`, respectively. This enables to investigate how the optimization is performed in detail and use the algorithms from within this package also for numerical analysis.

In the current version `Manopt.jl` version 0.3.12 the following algorithms are available

* Alternating Gradient Descent
* Chambolle-Pock [@BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez:2021:1]
* Conjugate Gradient Descent, including eight update rules
* Cyclic Proximal Point [@Bacak:2014:1]
* (parallel) Douglas–Rachford [@BergmannPerschSteidl:2016:1]
* Gradient Descent, including direction update rules including Momentum, Average, and a Nestorv-type one
* Nelder-Mead
* Particle Swarm Optimization [@BorckmansIshtevaAbsil2010]
* Quasi-Newton, with the BFGS, DFP, Broyden and a symmetric rank 1 update, their inverse updates as well as a limited memory variant of (inverse) BFGS [@HuangGallivanAbsil:2015:1]
* Stochastic Gradient Descent
* Subgradient Method
* Trust Regions, with inner Steihaug-Toint TCG solver [@AbsilBakerGallivan2006]

# Example

`Manopt.jl` is registered in the general Julia registry and can hence be installed typing `]add Manopt` in Julia REPL.
Given the `Sphere` from `Manifolds.jl` and a set of unit vectors $p_1,...,p_N\in\mathbb R^3$, where $N$ is the number of data points.
we can compute the generalization of the mean, called the Riemannian Center of Mass [@Karcher:1977:1], which is defined as the minimizer of the squared distances to the given data

$$ \operatorname*{arg\,min}_{x\in\mathcal M}\quad \displaystyle\sum_{k=1}^Nd_{\mathcal M}(x, p_k)^2, $$

where $d_{\mathcal M}$ denotes length of a shortest geodesic connecting the the two points in the arguments. It is called the Riemannian distance. For the sphere this distance is given by the length of the shorter great arc connecting the two points.

```julia
using Manopt, Manifolds, LinearAlgebra, Random
Random.seed!(42)
M = Sphere(2)
n = 100
pts = [ normalize(rand(3)) for _ in 1:n ]

F(M, y) = sum(1/(2*n) * distance.(Ref(M), pts, Ref(y)).^2)
gradF(M, y) = sum(1/n * grad_distance.(Ref(M), pts, Ref(y)))

x_mean = gradient_descent(M, F, gradF, pts[1])
```

Both the data `pts` and the resulting mean are shown in the following figure.

![100 random points `pts` and the result from the gradient descent to compute the `x_mean` (orange).](src/img/MeanIllustr.png)

In order to print the current iteration number, change and cost every iteration as well as the stopping reason, you can provide an `debug` keyword with the corresponding symbols interleaved with strings. The Symbol `:Stop` indicates the stopping reason should be printed in the end. The last integer in this array introduces that only every $i$th iteration a debug is printed.
While `:x` could be used to also print the current iterate, this usually takes up too much space.
It might be more reasonable to record these data.
The `record` keyword can be used for this, for example to record the current iterate `:x`,  the `:Change` from one iterate to the next and the current function value or `:Cost`.
To access the recorded values, set `return_options` to `true`, to obtain not only the resulting value as in the example before, but the whole `Options` structure.
Then the values can be accessed using the `get_record` function.
Just calling `get_record` returns an array of tuples, where each tuple stores the values of one iteration.
To obtain an array of values for one recorded value,
use the access per symbol, i.e. from the `Iteration`s we want to access the recorded iterates `:x` as follows:

```julia
o = gradient_descent(M, F, gradF, pts[1],
    debug=[:Iteration, " | ", :Change, " | ", :Cost, "\n", :Stop],
    record=[:x, :Change, :Cost],
    return_options=true
)
x_mean_2 = get_solver_result(o) #get solver result
all_values = get_record(o) # get a tuple of recorded data per iteration
iterates = get_record(o, :Iteration, :x) # get just iterates recorded per iteration
```

The debug output of this example looks as follows:
```
Initial |  | F(x): 0.26445609908711865
# 1 | Last Change: 0.5335127457059914 | F(x): 0.1164202416096971
# 2 | Last Change: 0.021122280232099756 | F(x): 0.11618855750050769
# 3 | Last Change: 0.0008160276261172897 | F(x): 0.11618821164343343
# 4 | Last Change: 3.1665020130539125e-5 | F(x): 0.11618821112258158
# 5 | Last Change: 1.2341904187709075e-6 | F(x): 0.11618821112179054
# 6 | Last Change: 5.575503985246929e-8 | F(x): 0.11618821112178938
# 7 | Last Change: 2.9802322387695312e-8 | F(x): 0.11618821112178936
The algorithm reached approximately critical point after 7 iterations;
    the gradient norm (1.9001130414808935e-9) is less than 1.0e-8.
```


# Related research and software

There are two projects that are most similar to `Manopt.jl` are [`Manopt`](https://manopt.org) [@manopt] in Matlab and [`pymanopt`](https://pymanopt.org) [@pymanopt] in Python.
Similarly [`ROPTLIB`](https://www.math.fsu.edu/~whuang2/Indices/index_ROPTLIB.html) [@HuangAbsilGallivanHand:2018:1] is a package for optimization on Manifolds in C++.
While all three packages cover some algorithms, most are less flexible for example in stating the stopping criterion, which is fixed to mainly maximal number of iterations or a small gradient. Most prominently, `Manopt.jl` is the first package that also covers methods for high-performance and high-dimensional nonsmooth optimization on manifolds.

The Riemannian Chambolle-Pock algorithm presented in [@BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez:2021:1] was developed using Manopt.jl. Based on this theory and algorithm, a higher order algorithm was introduced in [@DiepeveenLellmann:2021:1]. Optimized examples from [@BergmannGousenbourger:2018:2] performing data interpolation and approximation with manifold-valued Bézier curves, are also included in `Manopt.jl`.

# References