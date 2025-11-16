# About

Manopt.jl inherited its name from [Manopt](https://manopt.org), a Matlab toolbox for optimization on manifolds.
This Julia package was started and is currently maintained by [Ronny Bergmann](https://ronnybergmann.net/).

## Contributors

Thanks to the following contributors to `Manopt.jl`:

* [Constantin Ahlmann-Eltze](https://const-ae.name) implemented the [gradient and differential `check` functions](helpers/checks.md)
* [Renée Dornig](https://github.com/r-dornig) implemented the [particle swarm](solvers/particle_swarm.md), the [Riemannian Augmented Lagrangian Method](solvers/augmented_Lagrangian_method.md), the [Exact Penalty Method](solvers/exact_penalty_method.md), as well as the [`NonmonotoneLinesearch`](@ref). These solvers are also the first one with modular/exchangable sub solvers.
* [Willem Diepeveen](https://www.maths.cam.ac.uk/person/wd292) implemented the [primal-dual Riemannian semismooth Newton](solvers/primal_dual_semismooth_Newton.md) solver.
* [Hajg Jasa](https://www.ntnu.edu/employees/hajg.jasa) implemented the [convex bundle method](solvers/convex_bundle_method.md) and the [proximal bundle method](solvers/proximal_bundle_method.md) and a default subsolver each of them.
* Even Stephansen Kjemsås contributed to the implementation of the [Frank Wolfe Method](solvers/FrankWolfe.md) solver.
* Mathias Ravn Munkvold contributed most of the implementation of the [Adaptive Regularization with Cubics](solvers/adaptive-regularization-with-cubics.md) solver as well as its [Lanczos](@ref arc-Lanczos) subsolver
* [Sander Engen Oddsen](https://github.com/oddsen) contributed to the implementation of the [LTMADS](solvers/mesh_adaptive_direct_search.md) solver.
* [Jonas Püschel](https://www.uni-augsburg.de/de/fakultaet/mntf/math/prof/numa/team/jonas-pueschel/) contributed [restart rules for the conjugate gradient solver](@ref cg-restart).
* [Tom-Christian Riemer](https://www.tu-chemnitz.de/mathematik/wire/mitarbeiter.php) implemented the [trust regions](solvers/trust_regions.md) and [quasi Newton](solvers/quasi_Newton.md) solvers as well as the [truncated conjugate gradient descent](solvers/truncated_conjugate_gradient_descent.md) subsolver.
* [Markus A. Stokkenes](https://www.linkedin.com/in/markus-a-stokkenes-b41bba17b/) contributed most of the implementation of the [Interior Point Newton Method](solvers/interior_point_Newton.md) as well as its default [Conjugate Residual](solvers/conjugate_residual.md) subsolver
* [Laura Weigl](https://num.math.uni-bayreuth.de/en/team/laura-weigl/index.php) implemented the [Vector bundle Newton Method](solvers/vectorbundle_newton.md).
* [Manuel Weiss](https://scoop.iwr.uni-heidelberg.de/author/manuel-weiß/) implemented most of the [conjugate gradient update rules](@ref cg-coeffs)

as well as various [contributors](https://github.com/JuliaManifolds/Manopt.jl/graphs/contributors) providing small extensions, finding small bugs and mistakes and fixing them by opening [PR](https://github.com/JuliaManifolds/Manopt.jl/pulls)s. Thanks to all of you.

If you want to contribute a manifold or algorithm or have any questions, visit
the [GitHub repository](https://github.com/JuliaManifolds/Manopt.jl/)
to clone/fork the repository or open an issue.

## Work using Manopt.jl

* [ExponentialFamilyProjection.jl](https://github.com/ReactiveBayes/ExponentialFamilyProjection.jl) package uses `Manopt.jl` to project arbitrary functions onto the closest exponential family distributions. The package also integrates with [`RxInfer.jl`](https://github.com/ReactiveBayes/RxInfer.jl) to enable Bayesian inference in a larger set of probabilistic models.
* [Caesar.jl](https://github.com/JuliaRobotics/Caesar.jl) within non-Gaussian factor graph inference algorithms

If you are missing a package, that uses `Manopt.jl`, please [open an issue](https://github.com/JuliaManifolds/Manopt.jl/issues/new).
It would be great to collect anything and anyone using Manopt.jl in this list.

## Further packages

`Manopt.jl` belongs to the Manopt family:

*  [manopt.org](https://www.manopt.org) The Matlab version of Manopt, see also their :octocat: [GitHub repository](https://github.com/NicolasBoumal/manopt)
* [pymanopt.org](https://www.pymanopt.org/) The Python version of Manopt providing also several AD backends, see also their :octocat: [GitHub repository](https://github.com/pymanopt/pymanopt)

but there are also more packages providing tools on manifolds in other languages

* [Jax Geometry](https://github.com/ComputationalEvolutionaryMorphometry/jaxgeometry) (Python/Jax) for differential geometry and stochastic dynamics with deep learning
* [Geomstats](https://geomstats.github.io) (Python with several backends) focusing on statistics and machine learning :octocat: [GitHub repository](https://github.com/geomstats/geomstats)
* [Geoopt](https://geoopt.readthedocs.io/en/latest/) (Python & PyTorch) Riemannian ADAM & SGD. :octocat: [GitHub repository](https://github.com/geoopt/geoopt)
* [McTorch](https://github.com/mctorch/mctorch) (Python & PyToch) Riemannian SGD, Adagrad, ASA & CG.
* [ROPTLIB](https://www.math.fsu.edu/~whuang2/papers/ROPTLIB.htm) (C++) a Riemannian OPTimization LIBrary :octocat: [GitHub repository](https://github.com/whuang08/ROPTLIB)
* [TF Riemopt](https://github.com/master/tensorflow-riemopt) (Python & TensorFlow) Riemannian optimization using TensorFlow
