# Changelog

All notable Changes to the Julia package `Manopt.jl` will be documented in this file. The file was started with Version `0.4`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

Formally one could consider this version breaking, since a few functions
have been moved, that in earlier versions (0.3.x) have been used in example scripts.
These examples are now available again within [ManoptExamples.jl](), and with their
“reappearance” the corresponding costs, gradients, (adjoint) differentials and proximal maps
have been moved as well.
Actually, we do not consider this breaking, since we do not expect these were used.
We still document each and every of the functions below. They have partly been renamed,
and their documentation and testing has been extendede.

### Changed

* Bumped and added dependencies on all 3 Project.toml files, the main one, the docs/, an the tutorials/ one.
* `artificial_S2_lemniscate` is now available as [`ManoptExample.Lemniscate`]() – and works on arbitrary manifolds now.
* `artificial_S2_composite_bezier_curve` is now available as [`ManoptExamples.artificial_S2_composite_Bezier_curve`]()
* `adjoint_differential_forward_logs` is available as [`ManoptExamples.adjoint_differential_forward_logs`]()
* `adjoint:differential_bezier_control` is available as [`ManoptExamples.adjoint_differential_Bezier_control_points`]
* `cost_acceleration_bezier` is avilable as [`ManoptExamples.acceleration_Bezier`]()
* `cost_L2_acceleration_bezier` is available as [`ManoptExamples.L2_acceleration_Bezier`]()
* `costIntrICTV12` is available as [`ManoptExamples.Intrinsic_infimal_convolution_TV12`]()
* `costL2TV` is available as [`ManoptExamples.L2_Total_Variation`]()
* `costL2TV12` is available as [`ManoptExamples.L2_Total_Variation_1_2`]()
* `costL2TV2` is available as [`ManoptExamples.L2_second_order_Total_Variation`]()
* `costTV` is available as [`ManoptExamples.Total_Variation`]()
* `costTV2` is available as [`ManoptExamples.second_order_Total_Variation`]()
* `de_casteljau` is available as [`ManoptExamples.de_Casteljau`]()
* `differential_forward_logs` is available as [`ManoptExamples.differential_forward_logs`]()
* `differential_bezier_control` is available as [`ManoptExamples.differential_Bezier_control_points`]
* `forward_logs` is available as [`ManoptExamples.forward_logs`]()
* `get_bezier_degree` is available as [`ManoptExamples.get_Bezier_degree`]()
* `get_Bezier_inner_points` is available as [`ManoptExamples.get_Bezier_inner_points`]()
* `get_bezier_junction_tangent_vectors` is available as [`ManoptExamples.get_Bezier_junction_tangent_vectors`]()
* `get_bezier_junctions` is available as [`ManoptExamples.get_Bezier_junctions`]()
* `get_bezier_points` is available as [`ManoptExamples.get_Bezier_points`]()
* `get_bezier_segments` is available as [`ManoptExamples.get_Bezier_segments`]()
* `grad_acceleration_bezier` is available as [`ManoptExamples.grad_acceleration_Bezier`]()
* `grad_L2_acceleration_bezier` is available as [`ManoptExamples.grad_L2_acceleration_Bezier`]()
* `grad_Intrinsic_infimal_convolution_TV12` is available as [`ManoptExamples.Intrinsic_infimal_convolution_TV12``]()
* `grad_TV` is available as [`ManoptExamples.grad_Total_Variation`]()
* `costIntrICTV12` is now available as [`ManoptExamples.Intrinsic_infimal_convolution_TV12`]()
* `project_collaborative_TV` is available as [`ManoptExamples.project_collaborative_TV`]()
* `prox_parallel_TV` is available as [`ManoptExamples.prox_parallel_TV`]()
* `grad_TV2` is available as [`ManoptExamples.prox_second_order_Total_Variation`]()
* `prox_TV` is available as [`ManoptExamples.prox_Total_Variation`]()

## [0.4.43] - November 19, 2023

### Added

* vale.sh as a CI to keep track of a consistent documenttion

## [0.4.42] - November 6, 2023

### Added

* add `Manopt.JuMP_Optimizer` implementing JuMP's solver interface

## [0.4.41] - November 2, 2023

### Changed

* `trust_regions` is now more flexible and the sub solver (Steihaug-Toint tCG by default)
  can now be exchanged.
* `adaptive_regularization_with_cubics` is now more flexible as well, where it previously was a bit too
  much tightened to the Lanczos solver as well.
* Unified documentation notation and bumped dependencies to use DocumenterCitations 1.3

## [0.4.40] - October 24, 2023

### Added

* add a `--help` argument to `docs/make.jl` to document all available command line arguments
* add a `--exclude-tutorials` argument to `docs/make.jl`. This way, when quarto is not available
  on a computer, the docs can still be build with the tutorials not being added to the menu
  such that documenter does not expect them to exist.

### Changes

* Bump dependencies to `ManifoldsBase.jl` 0.15 and `Manifolds.jl` 0.9
* move the ARC CG subsolver to the main package, since `TangentSpace` is now already
  available from `ManifoldsBase`.

## [0.4.39] - October 9, 2023

### Changes

* also use the pair of a retraction and the inverse retraction (see last update)
  to perform the relaxation within the Douglas-Rachford algorithm.

## [0.4.38] - October 8, 2023

### Changes

* avoid allocations when calling `get_jacobian!` within the Levenberg-Marquard Algorithm.

### Fixed

* Fix a lot of typos in the documentation

## [0.4.37] - September 28, 2023

### Changes

* add more of the Riemannian Levenberg-Marquard algorithms parameters as keywords, so they
  can be changed on call
* generalize the internal reflection of Douglas-Rachford, such that is also works with an
  arbitrary pair of a reflection and an inverse reflection.

## [0.4.36] -  September 20, 2023

### Fixed

* Fixed a bug that caused non-matrix points and vectors to fail when working with approximate

## [0.4.35] -  September 14, 2023

### Added

* The access to functions of the objective is now unified and encapsulated in proper `get_` functions.

## [0.4.34] -  September 02, 2023

### Added

* an `ManifoldEuclideanGradientObjective` to allow the cost, gradient, and Hessian and other
  first or second derivative based elements to be Euclidean and converted when needed.
* a keyword `objective_type=:Euclidean` for all solvers, that specifies that an Objective shall be created of the above type

## [0.4.33] - August 24, 2023

### Added

* `ConstantStepsize` and `DecreasingStepsize` now have an additional field `type::Symbol` to assess whether the
  step-size should be relatively (to the gradient norm) or absolutely constant.

## [0.4.32] - August 23, 2023

### Added

* The adaptive regularization with cubics (ARC) solver.

## [0.4.31] - August 14, 2023

### Added

* A `:Subsolver` keyword in the `debug=` keyword argument, that activates the new `DebugWhenActive``
  to de/activate subsolver debug from the main solvers `DebugEvery`.

## [0.4.30] - August 3, 2023

### Changed

* References in the documentation are now rendered using [DocumenterCitations.jl](https://github.com/JuliaDocs/DocumenterCitations.jl)
* Asymptote export now also accepts a size in pixel instead of its default `4cm` size and `render` can be deactivated setting it to `nothing`.

## [0.4.29] - July 12, 2023

### Fixed

* fixed a bug, where `cyclic_proximal_point` did not work with decorated objectives.

## [0.4.28] - June 24, 2023

### Changed

* `max_stepsize` was specialized for `FixedRankManifold` to follow Matlab Manopt.

## [0.4.27] - June 15, 2023

### Added

* The `AdaptiveWNGrad` stepsize is now available as a new stepsize functor.

### Fixed

* Levenberg-Marquardt now possesses its parameters `initial_residual_values` and
  `initial_jacobian_f` also as keyword arguments, such that their default initialisations
  can be adapted, if necessary

## [0.4.26] - June 11, 2023

### Added

* simplify usage of gradient descent as sub solver in the DoC solvers.
* add a `get_state` function
* document `indicates_convergence`.

## [0.4.25] - June 5, 2023

### Fixed

* Fixes an allocation bug in the difference of convex algorithm

## [0.4.24] - June 4, 2023

### Added

* another workflow that deletes old PR renderings from the docs to keep them smaller in overall size.

### Changes

* bump dependencies since the extension between Manifolds.jl and ManifoldsDiff.jl has been moved to Manifolds.jl

## [0.4.23] - June 4, 2023

### Added

* More details on the Count and Cache tutorial

### Changed

* loosen constraints slightly

## [0.4.22] - May 31, 2023

### Added

* A tutorial on how to implement a solver

## [0.4.21] - May 22, 2023

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


## [0.4.20] - May 11, 2023

### Changed

* the default tolerances for the numerical `check_` functions were loosened a bit,
  such that `check_vector` can also be changed in its tolerances.

## [0.4.19] - May 7, 2023

### Added

* the sub solver for `trust_regions` is now customizable and can now be exchanged.

### Changed

* slightly changed the definitions of the solver states for ALM and EPM to be type stable

## [0.4.18] - May 4, 2023

### Added

* A function `check_Hessian(M, f, grad_f, Hess_f)` to numerically check the (Riemannian) Hessian of a function `f`

## [0.4.17] - April 28, 2023

### Added

* A new interface of the form `alg(M, objective, p0)` to allow to reuse
  objectives without creating `AbstractManoptSolverState`s and calling `solve!`. This especially still allows for any decoration of the objective and/or the state using `debug=`, or `record=`.

### Changed

* All solvers now have the initial point `p` as an optional parameter making it more accessible to first time users, `gradient_descent(M, f, grad_f)` is equivalent to `gradient_descent(M, f, grad_f, rand(M))`

### Fixed

* Unified the framework to work on manifold where points are represented by numbers for several solvers

## [0.4.16] - April 18, 2023

### Fixed

* the inner products used in `truncated_gradient_descent` now also work thoroughly on complex
  matrix manifolds

## [0.4.15] - April 13, 2023

### Changed

* `trust_regions(M, f, grad_f, hess_f, p)` now has the Hessian `hess_f` as well as
  the start point `p0` as an optional parameter and approximate it otherwise.
* `trust_regions!(M, f, grad_f, hess_f, p)` has the Hessian as an optional parameter
  and approximate it otherwise.

### Removed

* support for `ManifoldsBase.jl` 0.13.x, since with the definition of `copy(M,p::Number)`,
  in 0.14.4, we now use that instead of defining it ourselves.

## [0.4.14] - April 06, 2023

### Changed
* `particle_swarm` now uses much more in-place operations

### Fixed
* `particle_swarm` used quite a few `deepcopy(p)` commands still, which were replaced by `copy(M, p)`

## [0.4.13] - April 09, 2023

### Added

* `get_message` to obtain messages from sub steps of a solver
* `DebugMessages` to display the new messages in debug
* safeguards in Armijo line search and L-BFGS against numerical over- and underflow that report in messages

## [0.4.12] - April 4, 2023

### Added

* Introduce the [Difference of Convex Algorithm](https://manoptjl.org/stable/solvers/difference_of_convex/#DCASolver) (DCA)
  `difference_of_convex_algorithm(M, f, g, ∂h, p0)`
* Introduce the [Difference of Convex Proximal Point Algorithm](https://manoptjl.org/stable/solvers/difference_of_convex/#DCPPASolver) (DCPPA)
  `difference_of_convex_proximal_point(M, prox_g, grad_h, p0)`
* Introduce a `StopWhenGradientChangeLess` stopping criterion

## [0.4.11] - March 27, 2023

### Changed

* adapt tolerances in tests to the speed/accuracy optimized distance on the sphere in `Manifolds.jl` (part II)

## [0.4.10] - March 26, 2023

### Changed

* adapt tolerances in tests to the speed/accuracy optimized distance on the sphere in `Manifolds.jl`

## [0.4.9] - March 3, 2023

### Added

* introduce a wrapper that allows line searches from [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl)
  to be used within Manopt.jl, introduce the [manoptjl.org/stable/extensions/](https://manoptjl.org/stable/extensions/)
  page to explain the details.

## [0.4.8] - February 21, 2023

### Added

* a `status_summary` that displays the main parameters within several structures of Manopt,
  most prominently a solver state

### Changed

* Improved storage performance by introducing separate named tuples for points and vectors
* changed the `show` methods of `AbstractManoptSolverState`s to display their `state_summary
* Move tutorials to be rendered with Quarto into the documentation.

## [0.4.7] - February 14, 2023

### Changed

* Bump `[compat]` entry of ManifoldDiff to also include 0.3

## [0.4.6] - February 3, 2023

### Fixed

* Fixed a few stopping criteria even indicated to stop before the algorithm started.

## [0.4.5] - January 24, 2023

### Changed

* the new default functions that include `p` are used where possible
* a first step towards faster storage handling

## [0.4.4] - January 20, 2023

### Added

* Introduce `ConjugateGradientBealeRestart` to allow CG restarts using Beale‘s rule

### Fixed

* fix a type in `HestenesStiefelCoefficient`


## [0.4.3] - January 17, 2023

### Fixed

* the CG coefficient `β` can now be complex
* fix a bug in `grad_distance`

## [0.4.2] - January 16, 2023

### Changed

* the usage of `inner` in line search methods, such that they work well with
  complex manifolds as well


## [0.4.1] - January 15, 2023

### Fixed

* a `max_stepsize` per manifold to avoid leaving the injectivity radius,
  which it also defaults to

## [0.4.0] - January 10, 2023

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
