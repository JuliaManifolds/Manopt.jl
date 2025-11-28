---
title: 'Manopt.jl: Optimization on Manifolds in Julia'
tags:
  - Julia
  - Riemannian manifolds
  - optimization
  - numerical analysis
authors:
  - name: Ronny Bergmann
    orcid: 0000-0001-8342-7218
    affiliation: 1
affiliations:
 - name: Norwegian University of Science and Technology, Department of Mathematical Sciences, Trondheim, Norway
   index: 1
date: 22 July 2021
bibliography: bibliography.bib

---

# Summary

[`Manopt.jl`](https://manoptjl.org) provides a set of optimization algorithms for optimization problems given on a Riemannian manifold $\mathcal M$.
Based on a generic optimization framework, together with the interface [`ManifoldsBase.jl`](https://github.com/JuliaManifolds/ManifoldsBase.jl) for Riemannian manifolds, classical and recently developed methods are provided in an efficient implementation. Algorithms include the derivative-free Particle Swarm and Nelder–Mead algorithms, as well as classical gradient, conjugate gradient and stochastic gradient descent. Furthermore, quasi-Newton methods like a Riemannian L-BFGS [@HuangGallivanAbsil:2015:1] and nonsmooth optimization algorithms like a Cyclic Proximal Point Algorithm [@Bacak:2014:1], a (parallel) Douglas-Rachford algorithm [@BergmannPerschSteidl:2016:1] and a Chambolle-Pock algorithm [@BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez:2021:1] are provided, together with several basic cost functions, gradients and proximal maps as well as debug and record capabilities.

# Statement of Need

In many applications and optimization tasks, non-linear data appears naturally.
For example, when data on the sphere is measured [@GousenbourgerMassartMusolasAbsilJacquesHendrickxMarzouk:2017], diffusion data can be captured as a signal or even multivariate data of symmetric positive definite matrices [@ValkonenBrediesKnoll2013], and orientations like they appear for electron backscattered diffraction (EBSD) data [@BachmannHielscherSchaeben2011]. Another example are fixed rank matrices, appearing in matrix completion [@Vandereycken:2013:1].
Working on these data, for example doing data interpolation and approximation [@BergmannGousenbourger:2018:2], denoising [@LellmannStrekalovskiyKoetterCremers:2013:1; @BergmannFitschenPerschSteidl:2018], inpainting [@BergmannChanHielscherPerschSteidl:2016], or performing matrix completion [@GaoAbsil:2021], can usually be phrased as an optimization problem

$$ \text{Minimize}\quad f(x) \quad \text{where } x\in\mathcal M, $$

where the optimization problem is phrased on a Riemannian manifold $\mathcal M$.

A main challenge of these algorithms is that, compared to the (classical) Euclidean case, there is no addition available. For example, on the unit sphere $\mathbb S^2$ of unit vectors in $\mathbb R^3$, adding two vectors of unit length yields a vector that is not of unit norm.
The solution is to generalize the notion of a shortest path from the straight line to what is called a (shortest) geodesic, or acceleration-free curve.
Similarly, other features and properties also have to be rephrased and generalized when performing optimization on a Riemannian manifold.
Algorithms to perform the optimization can still often be stated in a generic way, i.e. on an arbitrary Riemannian manifold $\mathcal M$.
Further examples and a thorough introduction can be found in @AbsilMahonySepulchre:2008:1; @Boumal:2020:1.

For a user facing an optimization problem on a manifold, there are two obstacles to the actual numerical optimization: firstly, a suitable implementation of the manifold at hand is required, for example how to evaluate the above-mentioned geodesics; and secondly, an implementation of the optimization algorithm that employs said methods from the manifold, such that the algorithm can be applied to the cost function $f$ a user already has.

Using the interface for manifolds from the `ManifoldsBase.jl` package, the algorithms are implemented in the optimization framework. They can then be used with any manifold from [`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/) [@AxenBaranBergmannRzecki:2021:1], a library of efficiently-implemented Riemannian manifolds.
`Manopt.jl` provides a low-bar entry to optimization on manifolds, while also providing efficient implementations, that can easily be extended to cover  manifolds specified by the user.

# Functionality

`Manopt.jl` provides a comprehensive framework for optimization on Riemannian manifolds and a variety of algorithms using this framework.
The framework includes a generic way to specify a step size and a stopping criterion, as well as enhance the algorithm with debug and recording capabilities.
Each of the algorithms has a high-level interface to make it easy to use the algorithms directly.

An optimization task in `Manopt.jl` consists of a `Problem p` and `Options o`.
The `Problem` consists of all static information, like the cost function and a potential gradient of the optimization task. The `Options` specify the type of algorithm and the settings and data required to run the algorithm. For example, by default most options specify that the exponential map, which generalizes the notion of addition to the manifold, should be used and the algorithm steps are performed following an acceleration-free curve on the manifold. This might not be known in closed form for some manifolds, e.g. the [`Spectrahedron`](https://juliamanifolds.github.io/Manifolds.jl/v0.7/) does not have -- to the best of the author's knowledge -- a closed-form expression for the exponential map; hence more general arbitrary *retractions* can be specified for this instead.
Retractions are first-order approximations for the exponential map. They provide an alternative to the acceleration-free form, if no closed form solution is known. Otherwise, a retraction might also be chosen, when their evaluation is computationally cheaper than to use the exponential map, especially if their approximation error can be stated; see e.g. @BendokatZimmermann:2021.

Similarly, tangent vectors at different points are identified by a vector transport, which by default is the parallel transport.
By always providing a default, a user can start immediately, without thinking about these details. They can then modify these settings to improve speed or accuracy by specifying other retractions or vector transport to their needs.

The main methods to implement for a user-defined solver are `initialize_solver!(p,o)`, which fills the data in the options with an initial state, and `step_solver!(p,o,i)`, which performs the $i$th iteration.

Using a decorator pattern, `Options` can be encapsulated in `DebugOptions` and `RecordOptions`, which print and record arbitrary data stored within `Options`, respectively. This enables to investigate how the optimization is performed in detail and use the algorithms from within this package also for numerical analysis.

In the current version 0.3.17 of `Manopt.jl` the following algorithms are available:

* Alternating Gradient Descent ([`alternating_gradient_descent`](https://manoptjl.org/v0.3/solvers/alternating_gradient_descent.html))
* Chambolle-Pock ([`ChambollePock`](https://manoptjl.org/v0.3/solvers/ChambollePock.html)) [@BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez:2021:1]
* Conjugate Gradient Descent ([`conjugate_gradient_descent`](https://manoptjl.org/v0.3/solvers/conjugate_gradient_descent.html)), which includes eight direction update rules using the `coefficient` keyword:
  [`SteepestDirectionUpdateRule`](https://manoptjl.org/v0.3/solvers/conjugate_gradient_descent.html#Manopt.SteepestDirectionUpdateRule),   [`ConjugateDescentCoefficient`](https://manoptjl.org/v0.3/solvers/conjugate_gradient_descent.html#Manopt.ConjugateDescentCoefficient). [`DaiYuanCoefficientRule`](https://manoptjl.org/v0.3/solvers/conjugate_gradient_descent.html#Manopt.DaiYuanCoefficientRule), [`FletcherReevesCoefficient`](https://manoptjl.org/v0.3/solvers/conjugate_gradient_descent.html#Manopt.FletcherReevesCoefficient), [`HagerZhangCoefficient`](https://manoptjl.org/v0.3/solvers/conjugate_gradient_descent.html#Manopt.HagerZhangCoefficient), [`HeestenesStiefelCoefficient`](https://manoptjl.org/v0.3/solvers/conjugate_gradient_descent.html#Manopt.HeestenesStiefelCoefficient), [`LiuStoreyCoefficient`](https://manoptjl.org/v0.3/solvers/conjugate_gradient_descent.html#Manopt.LiuStoreyCoefficient), and [`PolakRibiereCoefficient`](https://manoptjl.org/v0.3/solvers/conjugate_gradient_descent.html#Manopt.PolakRibiereCoefficient)
* Cyclic Proximal Point ([`cyclic_proximal_point`](https://manoptjl.org/v0.3/solvers/cyclic_proximal_point.html)) [@Bacak:2014:1]
* (parallel) Douglas—Rachford ([`DouglasRachford`](https://manoptjl.org/v0.3/solvers/DouglasRachford.html)) [@BergmannPerschSteidl:2016:1]
* Gradient Descent ([`gradient_descent`](https://manoptjl.org/v0.3/solvers/gradient_descent.html)), including direction update rules ([`IdentityUpdateRule`](https://manoptjl.org/v0.3/solvers/gradient_descent.html#Manopt.IdentityUpdateRule) for the classical gradient descent) to perform [`MomentumGradient`](https://manoptjl.org/v0.3/solvers/gradient_descent.html#Manopt.MomentumGradient), [`AverageGradient`](https://manoptjl.org/v0.3/solvers/gradient_descent.html#Manopt.AverageGradient), and [`Nesterov`](https://manoptjl.org/v0.3/solvers/gradient_descent.html#Manopt.Nesterov) types
* Nelder-Mead ([`NelderMead`](https://manoptjl.org/v0.3/solvers/NelderMead.html))
* Particle-Swarm Optimization ([`particle_swarm`](https://manoptjl.org/v0.3/solvers/particle_swarm.html)) [@BorckmansIshtevaAbsil2010]
* Quasi-Newton ([`quasi_Newton`](https://manoptjl.org/v0.3/solvers/quasi_Newton.html)), with [`BFGS`](https://manoptjl.org/v0.3/solvers/quasi_Newton.html#Manopt.BFGS), [`DFP`](https://manoptjl.org/v0.3/solvers/quasi_Newton.html#Manopt.DFP), [`Broyden`](https://manoptjl.org/v0.3/solvers/quasi_Newton.html#Manopt.Broyden) and a symmetric rank 1 ([`SR1`](https://manoptjl.org/v0.3/solvers/quasi_Newton.html#Manopt.SR1)) update, their inverse updates as well as a limited memory variant of (inverse) BFGS  (using the `memory` keyword) [@HuangGallivanAbsil:2015:1]
* Stochastic Gradient Descent ([`stochastic_gradient_descent`](https://manoptjl.org/v0.3/solvers/stochastic_gradient_descent.html))
* Subgradient Method ([`subgradient_method`](https://manoptjl.org/v0.3/solvers/subgradient.html))
* Trust Regions ([`trust_regions`](https://manoptjl.org/v0.3/solvers/trust_regions.html)), with inner Steihaug-Toint ([`truncated_conjugate_gradient_descent`](https://manoptjl.org/v0.3/solvers/truncated_conjugate_gradient_descent.html)) solver [@AbsilBakerGallivan2006]

# Example

`Manopt.jl` is registered in the general Julia registry and can hence be installed typing `]add Manopt` in the Julia REPL.
Given the [`Sphere`](https://juliamanifolds.github.io/Manifolds.jl/v0.7/manifolds/sphere.html) from `Manifolds.jl` and a set of unit vectors $p_1,...,p_N\in\mathbb R^3$, where $N$ is the number of data points,
we can compute the generalization of the mean, called the Riemannian Center of Mass [@Karcher:1977:1], defined as the minimizer of the squared distances to the given data – a property that the mean in vector spaces fulfills:

$$ \operatorname*{arg\,min}_{x\in\mathcal M}\quad \displaystyle\sum_{k=1}^N d_{\mathcal M}(x, p_k)^2, $$

where $d_{\mathcal M}$ denotes the length of a shortest geodesic connecting the points specified by its two arguments;
this is called the Riemannian distance. For the sphere this [`distance`](https://juliamanifolds.github.io/Manifolds.jl/v0.7/manifolds/sphere.html#ManifoldsBase.distance-Tuple{AbstractSphere,%20Any,%20Any}) is given by the length of the shorter great arc connecting the two points.

```julia
using Manopt, Manifolds, LinearAlgebra, Random
Random.seed!(42)
M = Sphere(2)
n = 40
p = 1/sqrt(3) .* ones(3)
B = DefaultOrthonormalBasis()
pts = [ exp(M, p, get_vector(M, p, 0.425*randn(2), B)) for _ in 1:n ]

F(M, y) = sum(1/(2*n) * distance.(Ref(M), pts, Ref(y)).^2)
gradF(M, y) = sum(1/n * grad_distance.(Ref(M), pts, Ref(y)))

x_mean = gradient_descent(M, F, gradF, pts[1])
```

The resulting `x_mean` minimizes the (Riemannian) distances squared, but is especially a point of unit norm.
This should be compared to `mean(pts)`, which computes the mean in the embedding of the sphere, $\mathbb R^3$, and yields a point “inside” the sphere,
since its norm is approximately `0.858`. But even projecting this back onto the sphere yields a point that does not fulfill the property of minimizing the squared distances.

In the following figure the data `pts` (teal) and the resulting mean (orange) as well as the projected Euclidean mean (small, cyan) are shown.

![40 random points `pts` and the result from the gradient descent to compute the `x_mean` (orange) compared to a projection of their (Euclidean) mean onto the sphere (cyan).](src/img/MeanIllustr.png)

In order to print the current iteration number, change and cost every iteration as well as the stopping reason, you can provide a `debug` keyword with the corresponding symbols interleaved with strings. The Symbol `:Stop` indicates that the reason for stopping reason should be printed at the end. The last integer in this array specifies that debugging information should be printed only every $i$th iteration.
While `:x` could be used to also print the current iterate, this usually takes up too much space.

It might be more reasonable to *record* these data instead.
The `record` keyword can be used for this, for example to record the current iterate `:x`,  the `:Change` from one iterate to the next and the current function value or `:Cost`.
To access the recorded values, set `return_state` to `true`, to obtain not only the resulting value as in the example before, but the whole `Options` structure.
Then the values can be accessed using the `get_record` function.
Just calling `get_record` returns an array of tuples, where each tuple stores the values of one iteration.
To obtain an array of values for one recorded value,
use the access per symbol, i.e. from the `Iteration`s we want to access the recorded iterates `:x` as follows:

```julia
o = gradient_descent(M, F, gradF, pts[1],
    debug=[:Iteration, " | ", :Change, " | ", :Cost, "\n", :Stop],
    record=[:x, :Change, :Cost],
    return_state=true
)
x_mean_2 = get_solver_result(o) # the solver result
all_values = get_record(o) # a tuple of recorded data per iteration
iterates = get_record(o, :Iteration, :x) # iterates recorded per iteration
```

The debugging output of this example looks as follows:

```
Initial |  | F(x): 0.20638171781316278
# 1 | Last Change: 0.22025631624261213 | F(x): 0.18071614247165613
# 2 | Last Change: 0.014654955252636971 | F(x): 0.1805990319857418
# 3 | Last Change: 0.0013696682667046617 | F(x): 0.18059800144857607
# 4 | Last Change: 0.00013562945413135856 | F(x): 0.1805979913344784
# 5 | Last Change: 1.3519139571830234e-5 | F(x): 0.1805979912339798
# 6 | Last Change: 1.348534506171897e-6 | F(x): 0.18059799123297982
# 7 | Last Change: 1.3493575361575816e-7 | F(x): 0.1805979912329699
# 8 | Last Change: 2.580956827951785e-8 | F(x): 0.18059799123296988
# 9 | Last Change: 2.9802322387695312e-8 | F(x): 0.18059799123296993
The algorithm reached approximately critical point after 9 iterations;
    the gradient norm (1.3387605239861564e-9) is less than 1.0e-8.
```

For more details on more algorithms to compute the mean and other statistical functions on manifolds like the median
see [https://juliamanifolds.github.io/Manifolds.jl/v0.7/features/statistics.html](https://juliamanifolds.github.io/Manifolds.jl/v0.7/features/statistics.html).

# Related research and software

The two projects that are most similar to `Manopt.jl` are [`Manopt`](https://manopt.org) [@manopt] in Matlab and [`pymanopt`](https://pymanopt.org) [@pymanopt] in Python.
Similarly [`ROPTLIB`](https://www.math.fsu.edu/~whuang2/Indices/index_ROPTLIB.html) [@HuangAbsilGallivanHand:2018:1] is a package for optimization on Manifolds in C++.
While all three packages cover some algorithms, most are less flexible, for example in stating the stopping criterion, which is fixed to mainly the maximal number of iterations or a small gradient. Most prominently, `Manopt.jl` is the first package that also covers methods for high-performance and high-dimensional nonsmooth optimization on manifolds.

The Riemannian Chambolle-Pock algorithm presented in @BergmannHerzogSilvaLouzeiroTenbrinckVidalNunez:2021:1 was developed using `Manopt.jl`. Based on this theory and algorithm, a higher-order algorithm was introduced in @DiepeveenLellmann:2021:1. Optimized examples from @BergmannGousenbourger:2018:2 performing data interpolation and approximation with manifold-valued Bézier curves are also included in `Manopt.jl`.

# References
