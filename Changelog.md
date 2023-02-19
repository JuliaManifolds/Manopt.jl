# Changelog

All notable Changes to the Julia package `Manopt.jl` will be documented in this file. The file was started with Version `0.4`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.8]

### Added

* started this Changelog (checking the last few patches backwards)

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

* the usage of `inner` in linesearch methods, such that they work well with complex manifolds as well


## [0.4.1] - 15/01/2023

### Fixed

* a `max_stepsize` per manifold to avoid leaving the injectivity radius, which it also defaults to

## {0.4.0] - 10/01/2023

### Added

* Dependency on `ManifoldDiff.jl` and a start of moving actual derivatives, differentials, and gradients there.
* `AbstractManifoldObjective` to store the objective within the `AbstractManoptProblem`
* Introduce a `CostGrad` structure to store a function that computes the cost and gradient within one function.

### Changed

* `AbstractManoptProblem` replaces `Problem`
* the problem now contains a
* `AbstractManoptSolverState` replaces `Options`
* `random_point(M)` is replaced by `rand(M)` from `ManifoldsBase.jl
* `random_tangent(M, p)` is replaced by `rand(M; vector_at=p)`