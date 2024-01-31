# Changelog

All notable Changes to the Julia package `Manopt.jl` will be documented in this file. The file was started with Version `0.4`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.x] - dd/mm/2024

### Added

* `convex_bundle_method` optimization algorithm for non-smooth geodesically convex functions on Hadamard manifolds.
* `prox_bundle_method` optimization algorithm for non-smooth functions.
* `StopWhenSubgradientNormLess`, `StopWhenBundleLess`, and 
  `StopWhenProxBundleLess` stopping criteria.

## [unreleased]

### Added

* Allow the `message=` of the `DebugIfEntry` debug action to contain a format element to print the field in the message as well.

## [0.4.51] January 30, 2024

### Added

* A `StopWhenSubgradientNormLess` stopping criterion for subgradient-based optimization.

## [0.4.50] January 26, 2024

### Fixed

* Fix Quasi Newton on complex manifolds.

## [0.4.49] January 18, 2024

### Added

* A `StopWhenEntryChangeLess` to be able to stop on arbitrary small changes of specific fields
* generalises `StopWhenGradientNormLess` to accept arbitrary `norm=` functions
* refactor the default in `particle_swarm` to no longer “misuse” the iteration change check,
  but actually the new one one the `:swarm` entry

## [0.4.48] January 16, 2024

### Fixed

* fixes an imprecision in the interface of `get_iterate` that sometimes led to the swarm of `particle_swarm` being returned as the iterate.
* refactor `particle_swarm` in naming and access functions to avoid this also in the future.
  To access the whole swarm, one now should use `get_manopt_parameter(pss, :Population)`

## [0.4.47] January 6, 2024

### Fixed

* fixed a bug, where the retraction set in `check_Hessian` was not passed on to the optional inner `check_gradient` call, which could lead to unwanted side effects, see [#342](https://github.com/JuliaManifolds/Manopt.jl/issues/342).

## [0.4.46] January 1, 2024

### Changed

* An error is thrown when a line search from `LineSearches.jl` reports search failure.
* Changed default stopping criterion in ALM algorithm to mitigate an issue occurring when step size is very small.
* Default memory length in default ALM subsolver is now capped at manifold dimension.
* Replaced CI testing on Julia 1.8 with testing on Julia 1.10.

### Fixed

* A bug in `LineSearches.jl` extension leading to slower convergence.
* Fixed a bug in L-BFGS related to memory storage, which caused significantly slower convergence.

## [0.4.45] December 28, 2023

### Added

* Introduce `sub_kwargs` and `sub_stopping_criterion` for `trust_regions` as noticed in [#336](https://github.com/JuliaManifolds/Manopt.jl/discussions/336)

### Changed

* `WolfePowellLineSearch`, `ArmijoLineSearch` step sizes now allocate less
* `linesearch_backtrack!` is now available
* Quasi Newton Updates can work inplace of a direction vector as well.
* Faster `safe_indices` in L-BFGS.

## [0.4.44] December 12, 2023

Formally one could consider this version breaking, since a few functions
have been moved, that in earlier versions (0.3.x) have been used in example scripts.
These examples are now available again within [ManoptExamples.jl](https://juliamanifolds.github.io/ManoptExamples.jl/stable/), and with their
“reappearance” the corresponding costs, gradients, differentials, adjoint differentials, and proximal maps
have been moved there as well.
This is not considered breaking, since the functions were only used in the old, removed examples.
We still document each and every of the moved functions below. They have been partly renamed,
and their documentation and testing has been extended.

### Changed

* Bumped and added dependencies on all 3 Project.toml files, the main one, the docs/, an the tutorials/ one.
* `artificial_S2_lemniscate` is available as [`ManoptExample.Lemniscate`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.Lemniscate-Tuple{Number}) – and works on arbitrary manifolds now.
* `artificial_S1_signal` is available as [`ManoptExample.artificial_S1_signal`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.artificial_S1_signal)
* `artificial_S1_slope_signal` is available as [`ManoptExamples.artificial_S1_slope_signal`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.artificial_S1_slope_signal)
* `artificial_S2_composite_bezier_curve` is available as [`ManoptExamples.artificial_S2_composite_Bezier_curve`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.artificial_S2_composite_Bezier_curve-Tuple{})
* `artificial_S2_rotation_image` is available as [`ManoptExamples.artificial_S2_rotation_image`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.artificial_S2_rotation_image)
* `artificial_S2_whirl_image` is available as [`ManoptExamples.artificial_S2_whirl_image`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.artificial_S2_whirl_image)
* `artificial_S2_whirl_patch` is available as [`ManoptExamples.artificial_S2_whirl_path`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.artificial_S2_whirl_patch)
* `artificial_SAR_image` is available as [`ManoptExamples.artificial_SAR_image`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.artificialIn_SAR_image-Tuple{Integer})
* `artificial_SPD_image` is available as [`ManoptExamples.artificial_SPD_image`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.artificial_SPD_image)
* `artificial_SPD_image2` is available as [`ManoptExamples.artificial_SPD_image`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.artificial_SPD_image2)
* `adjoint_differential_forward_logs` is available as [`ManoptExamples.adjoint_differential_forward_logs`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.adjoint_differential_forward_logs-Union{Tuple{TPR},%20Tuple{TSize},%20Tuple{TM},%20Tuple{𝔽},%20Tuple{ManifoldsBase.PowerManifold{𝔽,%20TM,%20TSize,%20TPR},%20Any,%20Any}}%20where%20{𝔽,%20TM,%20TSize,%20TPR})
* `adjoint:differential_bezier_control` is available as [`ManoptExamples.adjoint_differential_Bezier_control_points`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.adjoint_differential_Bezier_control_points-Tuple{ManifoldsBase.AbstractManifold,%20AbstractVector{%3C:ManoptExamples.BezierSegment},%20AbstractVector,%20AbstractVector})
* `BezierSegment` is available as [`ManoptExamples.BeziérSegment`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.BezierSegment)
* `cost_acceleration_bezier` is avilable as [`ManoptExamples.acceleration_Bezier`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.acceleration_Bezier-Union{Tuple{P},%20Tuple{ManifoldsBase.AbstractManifold,%20AbstractVector{P},%20AbstractVector{%3C:Integer},%20AbstractVector{%3C:AbstractFloat}}}%20where%20P)
* `cost_L2_acceleration_bezier` is available as [`ManoptExamples.L2_acceleration_Bezier`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.L2_acceleration_Bezier-Union{Tuple{P},%20Tuple{ManifoldsBase.AbstractManifold,%20AbstractVector{P},%20AbstractVector{%3C:Integer},%20AbstractVector{%3C:AbstractFloat},%20AbstractFloat,%20AbstractVector{P}}}%20where%20P)
* `costIntrICTV12` is available as [`ManoptExamples.Intrinsic_infimal_convolution_TV12`]()
* `costL2TV` is available as [`ManoptExamples.L2_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.L2_Total_Variation-NTuple{4,%20Any})
* `costL2TV12` is available as [`ManoptExamples.L2_Total_Variation_1_2`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.L2_Total_Variation_1_2-Tuple{ManifoldsBase.PowerManifold,%20Vararg{Any,%204}})
* `costL2TV2` is available as [`ManoptExamples.L2_second_order_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.L2_second_order_Total_Variation-Tuple{ManifoldsBase.PowerManifold,%20Any,%20Any,%20Any})
* `costTV` is available as [`ManoptExamples.Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.Total_Variation)
* `costTV2` is available as [`ManoptExamples.second_order_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.second_order_Total_Variation)
* `de_casteljau` is available as [`ManoptExamples.de_Casteljau`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.de_Casteljau-Tuple{ManifoldsBase.AbstractManifold,%20Vararg{Any}})
* `differential_forward_logs` is available as [`ManoptExamples.differential_forward_logs`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.differential_forward_logs-Tuple{ManifoldsBase.PowerManifold,%20Any,%20Any})
* `differential_bezier_control` is available as [`ManoptExamples.differential_Bezier_control_points`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.differential_Bezier_control_points-Tuple{ManifoldsBase.AbstractManifold,%20AbstractVector{%3C:ManoptExamples.BezierSegment},%20AbstractVector,%20AbstractVector{%3C:ManoptExamples.BezierSegment}})
* `forward_logs` is available as [`ManoptExamples.forward_logs`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.forward_logs-Union{Tuple{TPR},%20Tuple{TSize},%20Tuple{TM},%20Tuple{𝔽},%20Tuple{ManifoldsBase.PowerManifold{𝔽,%20TM,%20TSize,%20TPR},%20Any}}%20where%20{𝔽,%20TM,%20TSize,%20TPR})
* `get_bezier_degree` is available as [`ManoptExamples.get_Bezier_degree`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.get_Bezier_degree-Tuple{ManifoldsBase.AbstractManifold,%20ManoptExamples.BezierSegment})
* `get_bezier_degrees` is available as [`ManoptExamples.get_Bezier_degrees`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.get_Bezier_degrees-Tuple{ManifoldsBase.AbstractManifold,%20AbstractVector{%3C:ManoptExamples.BezierSegment}})
* `get_Bezier_inner_points` is available as [`ManoptExamples.get_Bezier_inner_points`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.get_Bezier_inner_points-Tuple{ManifoldsBase.AbstractManifold,%20AbstractVector{%3C:ManoptExamples.BezierSegment}})
* `get_bezier_junction_tangent_vectors` is available as [`ManoptExamples.get_Bezier_junction_tangent_vectors`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.get_Bezier_junction_tangent_vectors-Tuple{ManifoldsBase.AbstractManifold,%20AbstractVector{%3C:ManoptExamples.BezierSegment}})
* `get_bezier_junctions` is available as [`ManoptExamples.get_Bezier_junctions`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.get_Bezier_junctions)
* `get_bezier_points` is available as [`ManoptExamples.get_Bezier_points`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.get_Bezier_points)
* `get_bezier_segments` is available as [`ManoptExamples.get_Bezier_segments`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.get_Bezier_segments-Union{Tuple{P},%20Tuple{ManifoldsBase.AbstractManifold,%20Vector{P},%20Any},%20Tuple{ManifoldsBase.AbstractManifold,%20Vector{P},%20Any,%20Symbol}}%20where%20P)
* `grad_acceleration_bezier` is available as [`ManoptExamples.grad_acceleration_Bezier`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.grad_acceleration_Bezier-Tuple{ManifoldsBase.AbstractManifold,%20AbstractVector,%20AbstractVector{%3C:Integer},%20AbstractVector})
* `grad_L2_acceleration_bezier` is available as [`ManoptExamples.grad_L2_acceleration_Bezier`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.grad_L2_acceleration_Bezier-Union{Tuple{P},%20Tuple{ManifoldsBase.AbstractManifold,%20AbstractVector{P},%20AbstractVector{%3C:Integer},%20AbstractVector,%20Any,%20AbstractVector{P}}}%20where%20P)
* `grad_Intrinsic_infimal_convolution_TV12` is available as [`ManoptExamples.Intrinsic_infimal_convolution_TV12``](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.grad_intrinsic_infimal_convolution_TV12-Tuple{ManifoldsBase.AbstractManifold,%20Vararg{Any,%205}})
* `grad_TV` is available as [`ManoptExamples.grad_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.grad_Total_Variation)
* `costIntrICTV12` is available as [`ManoptExamples.Intrinsic_infimal_convolution_TV12`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.Intrinsic_infimal_convolution_TV12-Tuple{ManifoldsBase.AbstractManifold,%20Vararg{Any,%205}})
* `project_collaborative_TV` is available as [`ManoptExamples.project_collaborative_TV`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.project_collaborative_TV)
* `prox_parallel_TV` is available as [`ManoptExamples.prox_parallel_TV`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.prox_parallel_TV)
* `grad_TV2` is available as [`ManoptExamples.prox_second_order_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.grad_second_order_Total_Variation)
* `prox_TV` is available as [`ManoptExamples.prox_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.prox_Total_Variation)
* `prox_TV2` is available as [`ManopExamples.prox_second_order_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.prox_second_order_Total_Variation-Union{Tuple{T},%20Tuple{ManifoldsBase.AbstractManifold,%20Any,%20Tuple{T,%20T,%20T}},%20Tuple{ManifoldsBase.AbstractManifold,%20Any,%20Tuple{T,%20T,%20T},%20Int64}}%20where%20T)

## [0.4.43] – November 19, 2023

### Added

* vale.sh as a CI to keep track of a consistent documenttion

## [0.4.42] - November 6, 2023
## [0.4.x] - dd/mm/2023

### Added

* `convex_bundle_method` optimization algorithm for non-smooth geodesically convex functions on Hadamard manifolds.
* `prox_bundle_method` optimization algorithm for non-smooth functions.
* `StopWhenSubgradientNormLess`, `StopWhenBundleLess`, and 
  `StopWhenProxBundleLess` stopping criteria.

## [Unreleased]

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

* The `AdaptiveWNGrad` stepsize is available as a new stepsize functor.

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
