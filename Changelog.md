# Changelog

All notable Changes to the Julia package `Manopt.jl` will be documented in this file. The file was started with Version `0.4`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.41] - 02/11/2023

### Changed

– `trust_regions` is now more flexible and the sub solver (Steinhaug-Toint tCG by default)
  can now be exchanged.
- `adaptive_regularization_with_cubics` is now more flexible as well, where it previously was a bit too
  much tightened to the Lanczos solver as well.
- Unified documentation notation and bumped dependencies to use DocumenterCitations 1.3

## [0.4.40] – 24/10/2023

### Added

* add a `--help` argument to `docs/make.jl` to document all availabel command line arguments
* add a `--exclude-tutorials` argument to `docs/make.jl`. This way, when quarto is not available
  on a computer, the docs can still be build with the tutorials not being added to the menu
  such that documenter does not expect them to exist.

### Changes

* Bump dependencies to `ManifoldsBase.jl` 0.15 and `Manifolds.jl` 0.9
* move the ARC CG subsolver to the main package, since `TangentSpace` is now already
  available from `ManifoldsBase`.

## [0.4.39] – 09/10/2023

### Changes

* also use the pair of a retraction and the inverse retraction (see last update)
  to perform the relaxation within the Douglas-Rachford algorithm.

## [0.4.38] – 08/10/2023

### Changes

* avoid allocations when calling `get_jacobian!` within the Levenberg-Marquard Algorithm.

### Fixed

* Fix a lot of typos in the documentation

## [0.4.37] – 28/09/2023

### Changes

* add more of the Riemannian Levenberg-Marquard algorithms parameters as keywords, so they
  can be changed on call
* generalize the internal reflection of Douglas-Rachford, such that is also works with an
  arbitrary pair of a reflection and an inverse reflection.

## [0.4.36] – 20/09/2023

### Fixed

* Fixed a bug that caused non-matrix points and vectors to fail when working with approcimate

## [0.4.35] – 14/09/2023

### Added

* The access to functions of the objective is now unified and encapsulated in proper `get_`
  functions.

## [0.4.34] – 02/09/2023

### Added

* an `ManifoldEuclideanGradientObjetive` to allow the cost, gradient, and Hessian and other
  first or second derivative based elements to be Euclidean and converted when needed.
* a keyword `objective_type=:Euclidean` for all solvers, that specifies that an Objective shall be created of the above type

## [0.4.33] - 24/08/2023

### Added

* `ConstantStepsize` and `DecreasingStepsize` now have an additional field `type::Symbol` to assess whether the
  step-size should be relatively (to the gradient norm) or absolutely constant.

## [0.4.32] - 23/08/2023

### Added

* The adaptive regularization with cubics (ARC) solver.

## [0.4.31] - 14/08/2023

### Added

* A `:Subsolver` keyword in the `debug=` keyword argument, that activates the new `DebugWhenActive``
  to de/activate subsolver debug from the main solvers `DebugEvery`.

## [0.4.30] - 03/08/2023

### Changed

* References in the documentation are now rendered using [DocumenterCitations.jl](https://github.com/JuliaDocs/DocumenterCitations.jl)
* Asymptote export now also accepts a size in pixel instead of its default `4cm` size and `render` can be deactivated setting it to `nothing`.

## [0.4.29] - 12/07/2023

### Fixed

* fixed a bug, where `cyclic_proximal_point` did not work with decorated objectives.

## [0.4.28] - 24/06/2023

### Changed

* `max_stepsize` was specialized for `FixedRankManifold` to follow Matlab Manopt.

## [0.4.27] - 15/06/2023

### Added

* The `AdaptiveWNGrad` stepsize is now available as a new stepsize functor.

### Fixed

* Levenberg-Marquardt now possesses its parameters `initial_residual_values` and
  `initial_jacobian_f` also as keyword arguments, such that their default initialisations
  can be adapted, if necessary

## [0.4.26] - 11/06/2023

### Added

* simplify usage of gradient descent as sub solver in the DoC solvers.
* add a `get_state` function
* document `indicates_convergence`.

## [0.4.25] - 05/06/2023

### Fixed

* Fixes an allocation bug in the difference of convex algorithm

## [0.4.24] - 04/06/2023

### Added

* another workflow that deletes old PR renderings from the docs to keep them smaller in overall size.

### Changes

* bump dependencies since the extension between Manifolds.jl and ManifoldsDiff.jl has been moved to Manifolds.jl

## [0.4.23] - 04/06/2023

### Added

* More details on the Count and Cache tutorial

### Changed

* loosen constraints slightly

## [0.4.22] - 31/05/2023

### Added

* A tutorial on how to implement a solver

## [0.4.21] - 22/05/2023

### Added

* A `ManifoldCacheObjective` as a decorator for objectives to cache results of calls,
  using LRU Caches as a weak dependency. For now this works with cost and gradient evaluations
* A `ManifoldCountObjective` as a decorator for objectives to enable counting of calls to for example the cost and the gradient
* adds a `return_objective` keyword, that switches the return of a solver to a tuple `(o, s)`,
  where `o` is the (possibly decorated) objective, and `s` is the “classical” solver return (state or point).
  This way the counted values can be accessed and the cache can be reused.
* change solvers on the mid level (form `solver(M, objective, p)`) to also accept decorated objectives

### Changed
* Switch all Requires weak dependencies to actual weak dependencies starting in Julia 1.9


## [0.4.20] - 11/05/2023

### Changed

* the default tolerances for the numerical `check_` functions were loosened a bit,
  such that `check_vector` can also be changed in its tolerances.

## [0.4.19] - 07/05/2023

### Added

* the sub solver for `trust_regions` is now customizable, i.e. can be exchanged.

### Changed

* slightly changed the definitions of the solver states for ALM and EPM to be type stable

## [0.4.18] - 04/05/2023

### Added

* A function `check_Hessian(M, f, grad_f, Hess_f)` to numerically check the (Riemannian) Hessian of a function `f`

## [0.4.17] - 28/04/2023

### Added

* A new interface of the form `alg(M, objective, p0)` to allow to reuse
  objectives without creating `AbstractManoptSolverState`s and calling `solve!`. This especially still allows for any decoration of the objective and/or the state using e.g. `debug=`, or `record=`.

### Changed

* All solvers now have the initial point `p` as an optional parameter making it more accessible to first time users, e.g. `gradient_descent(M, f, grad_f)`

### Fixed

* Unified the framework to work on manifold where points are represented by numbers for several solvers

## [0.4.16] - 18/04/2023

### Fixed

* the inner products used in `truncated_gradient_descent` now also work thoroughly on complex
  matrix manifolds

## [0.4.15] - 13/04/2023

### Changed

* `trust_regions(M, f, grad_f, hess_f, p)` now has the Hessian `hess_f` as well as
  the start point `p0` as an optional parameter and approximate it otherwise.
* `trust_regions!(M, f, grad_f, hess_f, p)` has the Hessian as an optional parameter
  and approximate it otherwise.

### Removed

* support for `ManifoldsBase.jl` 0.13.x, since with the definition of `copy(M,p::Number)`,
  in 0.14.4, we now use that instead of defining it ourselves.

## [0.4.14] - 06/04/2023

### Changed
* `particle_swarm` now uses much more in-place operations

### Fixed
* `particle_swarm` used quite a few `deepcopy(p)` commands still, which were replaced by `copy(M, p)`

## [0.4.13] - 09/04/2023

### Added

* `get_message` to obtain messages from sub steps of a solver
* `DebugMessages` to display the new messages in debug
* safeguards in Armijo linesearch and L-BFGS against numerical over- and underflow that report in messages

## [0.4.12] - 04/04/2023

### Added

* Introduce the [Difference of Convex Algorithm](https://manoptjl.org/stable/solvers/difference_of_convex/#DCASolver) (DCA)
  `difference_of_convex_algorithm(M, f, g, ∂h, p0)`
* Introduce the [Difference of Convex Proximal Point Algorithm](https://manoptjl.org/stable/solvers/difference_of_convex/#DCPPASolver) (DCPPA)
  `difference_of_convex_proximal_point(M, prox_g, grad_h, p0)`
* Introduce a `StopWhenGradientChangeLess` stopping criterion

## [0.4.11] - 27/04/2023

### Changed

* adapt tolerances in tests to the speed/accuracy optimized distance on the sphere in `Manifolds.jl` (part II)

## [0.4.10] - 26/04/2023

### Changed

* adapt tolerances in tests to the speed/accuracy optimized distance on the sphere in `Manifolds.jl`

## [0.4.9] – 03/03/2023

### Added

* introduce a wrapper that allows line searches from [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl)
  to be used within Manopt.jl, introduce the [manoptjl.org/stable/extensions/](https://manoptjl.org/stable/extensions/)
  page to explain the details.

## [0.4.8] - 21/02/2023

### Added

* a `status_summary` that displays the main parameters within several structures of Manopt,
  most prominently a solver state

### Changed

* Improved storage performance by introducing separate named tuples for points and vectors
* changed the `show` methods of `AbstractManoptSolverState`s to display their `state_summary
* Move tutorials to be rendered with Quarto into the documentation.

## [0.4.7] - 14/02/2023

### Changed

* Bump [compat] entry of ManifoldDiff to also include 0.3

## [0.4.6] - 03/02/2023

### Fixed

* Fixed a few stopping criteria even indicated to stop before the algorithm started.

## [0.4.5] - 24/01/2023

### Changed

* the new default functions that include `p` are used where possible
* a first step towards faster storage handling

## [0.4.4] - 20/01/2023

### Added

* Introduce `ConjugateGradientBealeRestart` to allow CG restarts using Beale‘s rule

### Fixed

* fix a type in `HestenesStiefelCoefficient`


## [0.4.3] - 17/01/2023

### Fixed

* the CG coefficient `β` can now be complex
* fix a bug in `grad_distance`

## [0.4.2] - 16/01/2023

### Changed

* the usage of `inner` in linesearch methods, such that they work well with
  complex manifolds as well


## [0.4.1] - 15/01/2023

### Fixed

* a `max_stepsize` per manifold to avoid leaving the injectivity radius,
  which it also defaults to

## [0.4.0] - 10/01/2023

### Added

* Dependency on `ManifoldDiff.jl` and a start of moving actual derivatives, differentials,
  and gradients there.
* `AbstractManifoldObjective` to store the objective within the `AbstractManoptProblem`
* Introduce a `CostGrad` structure to store a function that computes the cost and gradient
  within one function.

### Changed

* `AbstractManoptProblem` replaces `Problem`
* the problem now contains a
* `AbstractManoptSolverState` replaces `Options`
* `random_point(M)` is replaced by `rand(M)` from `ManifoldsBase.jl
* `random_tangent(M, p)` is replaced by `rand(M; vector_at=p)`