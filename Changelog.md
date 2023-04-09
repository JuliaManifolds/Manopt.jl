# Changelog

All notable Changes to the Julia package `Manopt.jl` will be documented in this file. The file was started with Version `0.4`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

## {0.4.0] - 10/01/2023

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