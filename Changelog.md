# Changelog

All notable Changes to the Julia package `Manopt.jl` are documented in this file.
The file was started with Version `0.4`.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

* Change the construction of the product manifold in `interior_point_newton` from `×` to `ProductManifold`, so that the algorithm also work on Product manifolds `M`, where it now correctly wraps `M` instead of extending it.
* unifies the doc strings for constrained problems
* fixes a few typos in the doc strings of matrix update formulae within the quasi-Newton and CG solver.
* unified the interfaces for line search related functions, especially,
  * `linesearch_backtrack(M, F, p, X, s, decrease, contract, η, f0; kwargs...)` now has `lf0=` and `gradient=` keyword arguments instead of positional ones for `X` and the last `f0`, respectively. It additionally has a `Dlf0=` keyword argument to pass the evaluated differential instead of the gradient, which otherwise defaults to calling the inner product.
* refactor the nonmonotone linesearch stepsize to have an initial guess that can be set. For now it still afterwards performs the Barzilein Borwein initial guess,
so a constant initial guess is recommended here for not.
* covers one last line in `proximal_gradient_plan`

## [0.5.27] November 11, 2025

### Added

* In `WolfePowellLinesearchStepsize`, two new keyword arguments `stop_increasing_at_step=` and `stop_decreasing_at_step=` were added to limit the number of increase/decrease steps in the initial bracketing phase for s_plus and s_minus, respectively. (resolves (#495))
* refactor `get_message` to only allocate a string when it is asked to deliver one, not every time a message is actually stored. This makes the message system align more with `get_reason`.

## [0.5.26] November 5, 2025

### Added

* a `vectorbundle_newton` solver to find zeros of equations defined on vector bundles.

### Fixed

* fixes a few inconsistencies regarding `get_embedding`, which now consistently uses a point type as positional second argument.

### Changed

* fixed a few typos in the documentation strings of a few solvers.
* fixed a typo in the documentation of `LevenbergMarquardt`.
* fixed a bug in an internal tex command to print sums in the documentation.
* fixed the use of `mesh_adaptive_direct_search` on manifolds with irrational injectivity radius.
* improved the `CONTRIBUTING.md` to reflect the new code formatter we use, as mentioned in (#527).

## [0.5.25] October 9, 2025

### Changed

* Bumped dependencies of all JuliaManifolds ecosystem packages to be consistent with ManifoldsBase.jl 2.0 and Manifolds.jl 0.11

## [0.5.24] October 6, 2025

### Added

* `CubicBracketingLinesearch` step size
* fallback in `proximal_gradient_plan`to use the norm of the inverse retraction if the distance is not available.

## [0.5.23] September 14, 2025

### Added

* `HybridCoefficient(args...)` conjugate gradient parameters.
* a function `has_converged(sc)` function for any `StoppingCriterion` to indicate that it _both_ has stopped and the reason is a convergence certificate.
  Note that compared to the static evaluation of `indicates_convergence(sc)`, which is independent of the state of the criterion,
  this is the dynamic variant to be used _after_ a solver has stopped.
* a `has_converged(::AbstractManoptSolverState)` function to check whether the solver has converged.

### Changed

* formerly a stopping criterion could be activated at certain iterations with `sc > 5`, `sc >= 5`, `sc == 5`, `sc <= 5`, and `sc < 5`.
  This caused too many issues with invalidations, so it has been reduced and moved to `sc ⩼ 5`, `sc ≟ 5`, `sc ⩻ 5` for the cases 1, 3, and 5, respectively,
  cf. (#509).
* Refine the `JuMP` extension and add an allocation-free cost and gradient callback for JuMP interface (#498)

## [0.5.22] September 09, 2025

### Added

* a `keywords_accepted(f, mode=:warn; kwargs...)` function that verifies that all keywords are accepted by a certain function.
* an internal function `calls_with_kwargs(f)` to indicate which functions `f` passes `kwargs...` to.
* a `KeywordsErrorMode` preference parameter to control how keywords that are not used/allowed should be treated. Values are `"none"`, `"warn"` (default), and `"error"`.
* Add Distance over Gradients (RDoG) stepsize: `DistanceOverGradientsStepsize` and factory `DistanceOverGradients`, a learning‑rate‑free, curvature‑aware stepsize with `show`/`repr` and tests on Euclidean, Sphere, and Hyperbolic manifolds.

### Fixed

* the typo in the name `AdaptiveRgularizationWithCubicsModelObjective` is fixed to `AdaptiveRegularizationWithCubicsModelObjective`.

## [0.5.21] September 5, 2025

### Added

* a system to track keywords, warning when unused ones are passed and a static way to explore possible keywords.
* a `warm_start_factor` field to `ProximalGradientMethodBacktrackingStepsize` to allow to scale the stepsize in the backtracking procedure.
* a `gradient=` keyword in several `Stepsize`s, such that one can avoid to internally avoid computing the gradient again.
* used the ``gradient=` keyword in
  * `alternating_gradient_descent`
  * `conjugate_gradient`
  * `Frank_Wolfe_method`
  * `gradient_descent`
  * `interior_point_newton`
  * `quasi_Newton`
  * `projected_gradient_method`
* a `restart_condition` functor to `conjugate_gradient_descent`, which allows the algorithm to restart if the search direction is sub-par (#492)
* two literature references

### Changed

* remodelled the docs for the extensions a bit, added `JuMP` to the DocumenterInterlinks.
* the internal `VectorizedManifold` within that extension is now called `ManifoldSet`
* the internal `ArrayShape` within that extensionis not called `ManifoldPointArrayShape`
* Switch to using [Runic.jl](https://github.com/fredrikekre/Runic.jl) as code formatter

### Fixed

* Fixed some math rendering in the docs, especially avoid `raw` strings and interpolate math symbols more often.

### Fixed

* Fixed allocations in the callbacks of the JuMP interface so that the solver can query the cost and gradient without allocating.

## [0.5.20] July 8, 2025

### Added

* a `DebugWarnIfStepsizeCollapsed` DebugAction and a related `:WarnStepsize` symbol for the debug dictionary. This is to be used in conjunction with the `ProximalGradientMethodBacktracking` stepsize to warn if the backtracking procedure of the `proximal_gradient_method` hit the stepsize length threshold without converging.

### Changed

* bumped dependencies.

### Fixed

* Fixed a few typos in the docs.

## [0.5.19] July 4, 2025

### Added

* a function `get_differential` and `get_differential_function` for first order objectives.
* a `ParentEvaluationType` to indicate that a certain objective inherits it evaluation from the parent (wrapping) objective
* a new `AllocatingInplaceEvaluation` that is used for the functions that offer both variants simultaneously.
* a `differential=` keyword for providing a faster way of computing `inner(M, p, grad_f(p), X)`, introduced to the algorithms `conjugate_gradient_descent`, `gradient_descent`, `Frank_Wolfe_method`, `quasi_Newton`

### Changed

* the `ManifoldGradientObjective` and the `ManifoldCostGradientObjective` are now merely
  a const special cases of the `ManifoldFirstOrderObjective`, since this type might now
  also represent a differential or other combinations of cost, grad, and differential, where they are computed together.
* the `AbstractManifoldGradientObjective` is renamed to `AbstractManifoldFirstOrderObjective`, since the
 second function might now also represent a differential.

### Fixed

* fixes a small bug where calling `mesh_adaptive_direct_search` with a start point in some cases did not initialise the state correctly with that start point.
* The `HestenesStiefelCoefficient` now also always returns a real value, similar
  the other coefficient rules. To the best of our knowledge, this might have been a bug previously.

## [0.5.18] June 18, 2025

### Added

* Introduce the algorithm `proximal_gradient_method` along
  with `ManifoldProximalGradientObjective`, `ProximalGradientMethodState`, as well as an experimental `ProximalGradientMethodAcceleration`.
* Add `ProximalGradientMethodBacktracking` stepsize.
* Add `StopWhenGradientMappingNormLess` stopping criterion.
* Introduce a `StopWhenRepeated` stopping criterion that stops when the given stopping criterion has indicated to stop `n` times (consecutively, if `consecutive=true`).
* Introduce a `StopWhenCriterionWithIterationCondition` stopping criterion that stops when a given stopping criterion has been satisfied together with a certain iteration condition. This can the generated even with shortcuts like `sc > 5`
* Introduce a `DebugCallback` that allows to add a callback function to the debug system
* Introduce a `callback=` keyword to all solvers.
* Added back functions `estimate_sectional_curvature`, `ζ_1`, `ζ_2`, `close_point` from `convex_bundle_method`; the function call can stay the same as before since there is a curvature estimation fallback
* Add back some fields and arguments such as `p_estimate`, `ϱ`, `α`, from `ConvexBundleMethodState`

### Changed

* make the `GradientDescentState` a bit more tolerant to ignore keywords it does not use.

## [0.5.17] June 3, 2025

### Added

* Introduce a `StopWhenCostChangeLess` stopping criterion that stops when the cost function changes less than a given value.

## [0.5.16] May 7, 2025

### Fixed

* fixes a bug in the `LineSearches.jl` extension, where two (old) `retract!`s were still
present; they were changed to `retact_fused!`.

## [0.5.15] May 6, 2025

### Fixed

* CMA-ES no longer errors when the covariance matrix has nonpositive eigenvalues due to numerical issues.

## [0.5.14] May 5, 2025

### Added

* `linear_subsolver!` is added as a keyword argument to the Levenberg-Marquardt interface.

### Changed

* adapt to using `default_basis` where appropriate.
* the tutorials are now rendered with `quarto` using the [`QuartoNotebookRunner.jl`](https://github.com/PumasAI/QuartoNotebookRunner.jl) and are hence purely julia based.

## [0.5.13] April 25, 2025

### Added

* Allow setting `AbstractManifoldObjective` through JuMP

### Changed

* Remove dependency on `ManoptExamples.jl` which yielded a circular dependency, though only through extras
* Unify dummy types and several test functions into the `ManoptTestSuite` subpackage.

### Fixed

* A scaling error that appeared only when calling `get_cost_function` on the new `ScaledManifoldObjective`.
* Documentation issues for quasi-Newton solvers.
* fixes a scaling error in quasi newton
* Fixes printing of JuMP models containg Manopt solver.

## [0.5.12] April 13, 2025

### Added

* a `ScaledManifoldObjective` to easier build scaled versions of objectives,
  especially turn maximisation problems into minimisation ones using a scaling of `-1`.
* Introduce a `ManifoldConstrainedSetObjective`
* Introduce a `projected_gradient_method`

## [0.5.11] April 8, 2025

### Added

* Configurable subsolver for the linear subproblem in Levenberg-Marquardt. The default subsolver is now also robust to numerical issues that may cause Cholesky decomposition to fail.

## [0.5.10] April 4, 2025

### Fixed

* a proper implementation of the preconditioning for `quasi_Newton`, that can be used instead
  of or in combination with the initial scaling.

## [0.5.9] March 24, 2025

### Added

* add a `PreconditionedDirection` variant to the `direction` gradient processor
  keyword argument and its corresponding `PreconditionedDirectionRule`
* make the preconditioner available in quasi Newton.
* in `gradient_descent` and `conjugate_gradient_descent` the rule can be added anyway.

### Fixed

* the links in the AD tutorial are fixed and moved to using `extref`

## [0.5.8] February 28, 2025

### Fixed

* fixed a small bug in the `NonmonotoneLinesearchStepsize` hwn the injectivity radius is an irrational number.
* fixed a small bug in `check_gradient` where `eps` might have been called on complex types.
* fixed a bug in several gradient based solvers like `quasi_newton`, such that they properly work with the combined cost grad objective.
* fixes a few typos in the docs.

## [0.5.7] February 20, 20265

### Added

* Adds a mesh adaptive direct search algorithm (MADS), using the LTMADS variant with a lower triangular (LT) random matrix in the mesh generation.

## [0.5.6] February 10, 2025

### Changed

* bump dependencies of all JuliaManifolds ecosystem packages to be consistent with ManifoldsBase 1.0

## [0.5.5] January 4, 2025

### Added

* the Levenberg-Marquardt algorithm internally uses a `VectorGradientFunction`, which allows
 to use a vector of gradients of a function returning all gradients as well for the algorithm
* The `VectorGradientFunction` now also have a `get_jacobian` function

### Changed

* Minimum Julia version is now 1.10 (the LTS which replaced 1.6)
* The vectorial functions had a bug where the original vector function for the mutating case
  was not always treated as mutating.

### Removed

* The geodesic regression example, first because it is not correct, second because it should become part of ManoptExamples.jl once it is correct.

## [0.5.4] December 11, 2024

### Added

* An automated detection whether the tutorials are present
   if not an also no quarto run is done, an automated `--exclude-tutorials` option is added.
* Support for ManifoldDiff 0.4
* icons upfront external links when they link to another package or Wikipedia.

## [0.5.3] October 18, 2024

### Added

* `StopWhenChangeLess`, `StopWhenGradientChangeLess` and `StopWhenGradientLess` can now use the new idea (ManifoldsBase.jl 0.15.18) of different outer norms on manifolds with components like power and product manifolds and all others that support this from the `Manifolds.jl` Library, like `Euclidean`

### Changed

* stabilize `max_stepsize` to also work when `injectivity_radius` dos not exist.
  It however would warn new users, that activate tutorial mode.
* Start a `ManoptTestSuite` sub package to store dummy types and common test helpers in.

## [0.5.2] October 5, 2024

### Added

* three new symbols to easier state to record the `:Gradient`, the `:GradientNorm`, and the `:Stepsize`.

### Changed

* fix a few typos in the documentation
* improved the documentation for the initial guess of [`ArmijoLinesearchStepsize`](https://manoptjl.org/stable/plans/stepsize/#Manopt.ArmijoLinesearch).

## [0.5.1] September 4, 2024

### Changed

* slightly improves the test for the `ExponentialFamilyProjection` text on the about page.

### Added

* the `proximal_point` method.

## [0.5.0] August 29, 2024

This breaking update is mainly concerned with improving a unified experience through all solvers
and some usability improvements, such that for example the different gradient update rules are easier to specify.

In general this introduces a few factories, that avoid having to pass the manifold to keyword arguments

### Added

* A `ManifoldDefaultsFactory` that postpones the creation/allocation of manifold-specific fields in for example direction updates, step sizes and stopping criteria. As a rule of thumb, internal structures, like a solver state should store the final type. Any high-level interface, like the functions to start solvers, should accept such a factory in the appropriate places and call the internal `_produce_type(factory, M)`, for example before passing something to the state.
* a `documentation_glossary.jl` file containing a glossary of often used variables in fields, arguments, and keywords, to print them in a unified manner. The same for usual sections, text, and math notation that is often used within the doc-strings.

### Changed

* Any `Stepsize` now has a `Stepsize` struct used internally as the original `struct`s before. The newly exported terms aim to fit `stepsize=...` in naming and create a `ManifoldDefaultsFactory` instead, so that any stepsize can be created without explicitly specifying the manifold.
  * `ConstantStepsize` is no longer exported, use `ConstantLength` instead. The length parameter is now a positional argument following the (optional) manifold. Besides that `ConstantLength` works as before,just that omitting the manifold fills the one specified in the solver now.
  * `DecreasingStepsize` is no longer exported, use `DecreasingLength` instead. `ConstantLength` works as before,just that omitting the manifold fills the one specified in the solver now.
  * `ArmijoLinesearch` is now called `ArmijoLinesearchStepsize`. `ArmijoLinesearch` works as before,just that omitting the manifold fills the one specified in the solver now.
  * `WolfePowellLinesearch` is now called `WolfePowellLinesearchStepsize`, its constant `c_1` is now unified with Armijo and called `sufficient_decrease`, `c_2` was renamed to `sufficient_curvature`. Besides that, `WolfePowellLinesearch` works as before, just that omitting the manifold fills the one specified in the solver now.
  * `WolfePowellBinaryLinesearch` is now called `WolfePowellBinaryLinesearchStepsize`, its constant `c_1` is now unified with Armijo and called `sufficient_decrease`, `c_2` was renamed to `sufficient_curvature`. Besides that, `WolfePowellBinaryLinesearch` works as before, just that omitting the manifold fills the one specified in the solver now.
  * `NonmonotoneLinesearch` is now called `NonmonotoneLinesearchStepsize`. `NonmonotoneLinesearch` works as before, just that omitting the manifold fills the one specified in the solver now.
  * `AdaptiveWNGradient` is now called `AdaptiveWNGradientStepsize`. Its second positional argument, the gradient function was only evaluated once for the `gradient_bound` default, so it has been replaced by the keyword `X=` accepting a tangent vector. The last positional argument `p` has also been moved to a keyword argument. Besides that, `AdaptiveWNGradient` works as before, just that omitting the manifold fills the one specified in the solver now.
* Any `DirectionUpdateRule` now has the `Rule` in its name, since the original name is used to create the `ManifoldDefaultsFactory` instead. The original constructor now no longer requires the manifold as a parameter, that is later done in the factory. The `Rule` is, however, also no longer exported.
  * `AverageGradient` is now called `AverageGradientRule`. `AverageGradient` works as before, but the manifold as its first parameter is no longer necessary and `p` is now a keyword argument.
  * The `IdentityUpdateRule` now accepts a manifold optionally for consistency, and you can use `Gradient()` for short as well as its factory. Hence `direction=Gradient()` is now available.
  * `MomentumGradient` is now called `MomentumGradientRule`. `MomentumGradient` works as before, but the manifold as its first parameter is no longer necessary and `p` is now a keyword argument.
  * `Nesterov` is now called `NesterovRule`. `Nesterov` works as before, but the manifold as its first parameter is no longer necessary and `p` is now a keyword argument.
  * `ConjugateDescentCoefficient` is now called `ConjugateDescentCoefficientRule`. `ConjugateDescentCoefficient` works as before, but can now use the factory in between
  * the `ConjugateGradientBealeRestart` is now called `ConjugateGradientBealeRestartRule`. For the `ConjugateGradientBealeRestart` the manifold is now a first parameter, that is not necessary and no longer the `manifold=` keyword.
  * `DaiYuanCoefficient` is now called `DaiYuanCoefficientRule`. For the `DaiYuanCoefficient` the manifold as its first parameter is no longer necessary and the vector transport has been unified/moved to the `vector_transport_method=` keyword.
  * `FletcherReevesCoefficient` is now called `FletcherReevesCoefficientRule`. `FletcherReevesCoefficient` works as before, but can now use the factory in between
  * `HagerZhangCoefficient` is now called `HagerZhangCoefficientRule`. For the `HagerZhangCoefficient` the manifold as its first parameter is no longer necessary and the vector transport has been unified/moved to the `vector_transport_method=` keyword.
  * `HestenesStiefelCoefficient` is now called `HestenesStiefelCoefficientRule`. For the `HestenesStiefelCoefficient` the manifold as its first parameter is no longer necessary and the vector transport has been unified/moved to the `vector_transport_method=` keyword.
  * `LiuStoreyCoefficient` is now called `LiuStoreyCoefficientRule`. For the `LiuStoreyCoefficient` the manifold as its first parameter is no longer necessary and the vector transport has been unified/moved to the `vector_transport_method=` keyword.
  * `PolakRibiereCoefficient` is now called `PolakRibiereCoefficientRule`. For the `PolakRibiereCoefficient` the manifold as its first parameter is no longer necessary and the vector transport has been unified/moved to the `vector_transport_method=` keyword.
  * the `SteepestDirectionUpdateRule` is now called `SteepestDescentCoefficientRule`. The `SteepestDescentCoefficient` is equivalent, but creates the new factory temporarily.
  * `AbstractGradientGroupProcessor` is now called `AbstractGradientGroupDirectionRule`
    * the `StochasticGradient` is now called `StochasticGradientRule`. The `StochasticGradient` is equivalent, but creates the new factory temporarily, so that the manifold is not longer necessary.
  * the `AlternatingGradient` is now called `AlternatingGradientRule`.
  The `AlternatingGradient` is equivalent, but creates the new factory temporarily, so that the manifold is not longer necessary.
* `quasi_Newton` had a keyword `scale_initial_operator=` that was inconsistently declared (sometimes boolean, sometimes real) and was unused.
  It is now called `initial_scale=1.0` and scales the initial (diagonal, unit) matrix within the approximation of the Hessian additionally to the $\frac{1}{\lVert g_k\rVert}$ scaling with the norm of the oldest gradient for the limited memory variant. For the full matrix variant the initial identity matrix is now scaled with this parameter.
* Unify doc strings and presentation of keyword arguments
  * general indexing, for example in a vector, uses `i`
  * index for inequality constraints is unified to `i` running from `1,...,m`
  * index for equality constraints is unified to `j` running from `1,...,n`
  * iterations are using now `k`
* `get_manopt_parameter` has been renamed to `get_parameter` since it is internal,
  so internally that is clear; accessing it from outside hence reads anyway `Manopt.get_parameter`
* `set_manopt_parameter!` has been renamed to `set_parameter!` since it is internal,
  so internally that is clear; accessing it from outside hence reads `Manopt.set_parameter!`
* changed the `stabilize::Bool=` keyword in `quasi_Newton` to the more flexible `project!=`
  keyword, this is also more in line with the other solvers. Internally the same is done
  within the `QuasiNewtonLimitedMemoryDirectionUpdate`. To adapt,
  * the previous `stabilize=true` is now set with `(project!)=embed_project!` in general,
    and if the manifold is represented by points in the embedding, like the sphere, `(project!)=project!` suffices
  * the new default is `(project!)=copyto!`, so by default no projection/stabilization is performed.
* the positional argument `p` (usually the last or the third to last if sub solvers existed) has been moved to a keyword argument `p=` in all State constructors
* in `NelderMeadState` the `population` moved from positional to keyword argument as well,
* the way to initialise sub solvers in the solver states has been unified In the new variant
  * the `sub_problem` is always a positional argument; namely the last one
  * if the `sub_state` is given as a optional positional argument after the problem, it has to be a manopt solver state
  * you can provide the new `ClosedFormSolverState(e::AbstractEvaluationType)` for the state
    to indicate that the `sub_problem` is a closed form solution (function call) and how it
    has to be called
  * if you do not provide the `sub_state` as positional, the keyword `evaluation=` is used
    to generate the state `ClosedFormSolverState`.
  * when previously `p` and eventually `X` where positional arguments, they are now moved
    to keyword arguments of the same name for start point and tangent vector.
  * in detail
    * `AdaptiveRegularizationState(M, sub_problem [, sub_state]; kwargs...)` replaces
      the (unused) variant to only provide the objective; both `X` and `p` moved to keyword arguments.
    * `AugmentedLagrangianMethodState(M, objective, sub_problem; evaluation=...)` was added
    * `AugmentedLagrangianMethodState(M, objective, sub_problem, sub_state; evaluation=...)` now has `p=rand(M)` as keyword argument instead of being the second positional one
    * `ExactPenaltyMethodState(M, sub_problem; evaluation=...)` was added and `ExactPenaltyMethodState(M, sub_problem, sub_state; evaluation=...)` now has `p=rand(M)` as keyword argument instead of being the second positional one
    * `DifferenceOfConvexState(M, sub_problem; evaluation=...)` was added and `DifferenceOfConvexState(M, sub_problem, sub_state; evaluation=...)` now has `p=rand(M)` as keyword argument instead of being the second positional one
    * `DifferenceOfConvexProximalState(M, sub_problem; evaluation=...)` was added and `DifferenceOfConvexProximalState(M, sub_problem, sub_state; evaluation=...)` now has `p=rand(M)` as keyword argument instead of being the second positional one
  * bumped `Manifolds.jl`to version 0.10; this mainly means that any algorithm working on a product manifold and requiring `ArrayPartition` now has to explicitly do `using RecursiveArrayTools`.

### Fixed

* the `AverageGradientRule` filled its internal vector of gradients wrongly or mixed it up in parallel transport. This is now fixed.

### Removed

* the `convex_bundle_method` and its `ConvexBundleMethodState` no longer accept the keywords `k_size`, `p_estimate` nor `ϱ`, they are superseded by just providing `k_max`.
* the `truncated_conjugate_gradient_descent(M, f, grad_f, hess_f)` has the Hessian now
   a mandatory argument. To use the old variant,
   provide `ApproxHessianFiniteDifference(M, copy(M, p), grad_f)` to `hess_f` directly.
* all deprecated keyword arguments and a few function signatures were removed:
  * `get_equality_constraints`, `get_equality_constraints!`, `get_inequality_constraints`, `get_inequality_constraints!` are removed. Use their singular forms and set the index to `:` instead.
  * `StopWhenChangeLess(ε)` is removed, use ``StopWhenChangeLess(M, ε)` instead to fill for example the retraction properly used to determine the change
* In the `WolfePowellLinesearch` and  `WolfeBinaryLinesearch`the `linesearch_stopsize=` keyword is replaced by `stop_when_stepsize_less=`
* `DebugChange` and `RecordChange` had a `manifold=` and a `invretr` keyword that were replaced by the first positional argument `M` and `inverse_retraction_method=`, respectively
* in the `NonlinearLeastSquaresObjective` and `LevenbergMarquardt` the `jacB=` keyword is now called `jacobian_tangent_basis=`
* in `particle_swarm` the `n=` keyword is replaced by `swarm_size=`.
* `update_stopping_criterion!` has been removed and unified with `set_parameter!`. The code adaptions are
  * to set a parameter of a stopping criterion, just replace `update_stopping_criterion!(sc, :Val, v)` with `set_parameter!(sc, :Val, v)`
  * to update a stopping criterion in a solver state, replace the old `update_stopping_criterion!(state, :Val, v)` tat passed down to the stopping criterion by the explicit pass down with `set_parameter!(state, :StoppingCriterion, :Val, v)`

## [0.4.69] August 3, 2024

### Changed

* Improved performance of Interior Point Newton Method.

## [0.4.68] August 2, 2024

### Added

* an Interior Point Newton Method, the `interior_point_newton`
* a `conjugate_residual` Algorithm to solve a linear system on a tangent space.
* `ArmijoLinesearch` now allows for additional `additional_decrease_condition` and `additional_increase_condition` keywords to add further conditions to accept additional conditions when to accept an decreasing or increase of the stepsize.
* add a `DebugFeasibility` to have a debug print about feasibility of points in constrained optimisation employing the new `is_feasible` function
* add a `InteriorPointCentralityCondition` that can be added for step candidates within the line search of `interior_point_newton`
* Add Several new functors
  * the `LagrangianCost`, `LagrangianGradient`, `LagrangianHessian`, that based on a constrained objective allow to construct the Hessian objective of its Lagrangian
  * the `CondensedKKTVectorField` and its `CondensedKKTVectorFieldJacobian`, that are being used to solve a linear system within `interior_point_newton`
  * the `KKTVectorField` as well as its `KKTVectorFieldJacobian` and ``KKTVectorFieldAdjointJacobian`
  * the `KKTVectorFieldNormSq` and its `KKTVectorFieldNormSqGradient` used within the Armijo line search of `interior_point_newton`
* New stopping criteria
  * A `StopWhenRelativeResidualLess` for the `conjugate_residual`
  * A `StopWhenKKTResidualLess` for the `interior_point_newton`

## [0.4.67] July 25, 2024

### Added

* `max_stepsize` methods for `Hyperrectangle`.

### Fixed

* a few typos in the documentation
* `WolfePowellLinesearch` no longer uses `max_stepsize` with invalid point by default.

## [0.4.66] June 27, 2024

### Changed

* Remove functions `estimate_sectional_curvature`, `ζ_1`, `ζ_2`, `close_point` from `convex_bundle_method`
* Remove some unused fields and arguments such as `p_estimate`, `ϱ`, `α`, from `ConvexBundleMethodState` in favor of jut `k_max`
* Change parameter `R` placement in `ProximalBundleMethodState` to fifth position

## [0.4.65] June 13, 2024

### Changed

* refactor stopping criteria to not store a `sc.reason` internally, but instead only
  generate the reason (and hence allocate a string) when actually asked for a reason.

## [0.4.64] June 4, 2024

### Added

* Remodel the constraints and their gradients into separate `VectorGradientFunctions`
  to reduce code duplication and encapsulate the inner model of these functions and their gradients
* Introduce a `ConstrainedManoptProblem` to model different ranges for the gradients in the
  new `VectorGradientFunction`s beyond the default `NestedPowerRepresentation`
* introduce a `VectorHessianFunction` to also model that one can provide the vector of Hessians
  to constraints
* introduce a more flexible indexing beyond single indexing, to also include arbitrary ranges
  when accessing vector functions and their gradients and hence also for constraints and
  their gradients.

### Changed

* Remodel `ConstrainedManifoldObjective` to store an `AbstractManifoldObjective`
  internally instead of directly `f` and `grad_f`, allowing also Hessian objectives
  therein and implementing access to this Hessian
* Fixed a bug that Lanczos produced NaNs when started exactly in a minimizer, since the algorithm initially divides by the gradient norm.

### Deprecated

* deprecate `get_grad_equality_constraints(M, o, p)`, use `get_grad_equality_constraint(M, o, p, :)`
  from the more flexible indexing instead.

## [0.4.63] May 11, 2024

### Added

* `:reinitialize_direction_update` option for quasi-Newton behavior when the direction is not a descent one. It is now the new default for `QuasiNewtonState`.
* Quasi-Newton direction update rules are now initialized upon start of the solver with the new internal function `initialize_update!`.

### Fixed

* ALM and EPM no longer keep a part of the quasi-Newton subsolver state between runs.

### Changed

* Quasi-Newton solvers: `:reinitialize_direction_update` is the new default behavior in case of detection of non-descent direction instead of `:step_towards_negative_gradient`. `:step_towards_negative_gradient` is still available when explicitly set using the `nondescent_direction_behavior` keyword argument.

## [0.4.62] May 3, 2024

### Changed

* bumped dependency of ManifoldsBase.jl to 0.15.9 and imported their numerical verify functions. This changes the `throw_error` keyword used internally to a `error=` with a symbol.

## [0.4.61] April 27, 2024

### Added

* Tests use `Aqua.jl` to spot problems in the code
* introduce a feature-based list of solvers and reduce the details in the alphabetical list
* adds a `PolyakStepsize`
* added a `get_subgradient` for `AbstractManifoldGradientObjectives` since their gradient is a special case of a subgradient.

### Fixed

* `get_last_stepsize` was defined in quite different ways that caused ambiguities. That is now internally a bit restructured and should work nicer.
  Internally this means that the interim dispatch on `get_last_stepsize(problem, state, step, vars...)` was removed. Now the only two left are `get_last_stepsize(p, s, vars...)` and the one directly checking `get_last_stepsize(::Stepsize)` for stored values.
* the accidentally exported `set_manopt_parameter!` is no longer exported

### Changed

* `get_manopt_parameter` and `set_manopt_parameter!` have been revised and better documented,
  they now use more semantic symbols (with capital letters) instead of direct field access
  (lower letter symbols). Since these are not exported, this is considered an internal, hence non-breaking change.
  * semantic symbols are now all nouns in upper case letters
  * `:active` is changed to `:Activity`

## [0.4.60] April 10, 2024

### Added

* `RecordWhenActive` to allow records to be deactivated during runtime, symbol `:WhenActive`
* `RecordSubsolver` to record the result of a subsolver recording in the main solver, symbol `:Subsolver`
* `RecordStoppingReason` to record the reason a solver stopped
* made the `RecordFactory` more flexible and quite similar to `DebugFactory`, such that it is now also easy to specify recordings at the end of solver runs. This can especially be used to record final states of sub solvers.

### Changed

* being a bit more strict with internal tools and made the factories for record non-exported, so this is the same as for debug.

### Fixed

* The name `:Subsolver` to generate `DebugWhenActive` was misleading, it is now called `:WhenActive` referring to “print debug only when set active, that is by the parent (main) solver”.
* the old version of specifying `Symbol => RecordAction` for later access was ambiguous, since
it could also mean to store the action in the dictionary under that symbol. Hence the order for access
was switched to `RecordAction => Symbol` to resolve that ambiguity.

## [0.4.59] April 7, 2024

### Added

* A Riemannian variant of the CMA-ES (Covariance Matrix Adaptation Evolutionary Strategy) algorithm, `cma_es`.

### Fixed

* The constructor dispatch for `StopWhenAny` with `Vector` had incorrect element type assertion which was fixed.

## [0.4.58] March 18, 2024

### Added

* more advanced methods to add debug to the beginning of an algorithm, a step, or the end of
  the algorithm with `DebugAction` entries at `:Start`, `:BeforeIteration`, `:Iteration`, and
  `:Stop`, respectively.
* Introduce a Pair-based format to add elements to these hooks, while all others ar
  now added to :Iteration (no longer to `:All`)
* (planned) add an easy possibility to also record the initial stage and not only after the first iteration.

### Changed

* Changed the symbol for the `:Step` dictionary to be `:Iteration`, to unify this with the symbols used in recording,
  and removed the `:All` symbol. On the fine granular scale, all but `:Start` debugs are now reset on init.
  Since these are merely internal entries in the debug dictionary, this is considered non-breaking.
* introduce a `StopWhenSwarmVelocityLess` stopping criterion for `particle_swarm` replacing
  the current default of the swarm change, since this is a bit more effective to compute

### Fixed

* fixed the outdated documentation of `TruncatedConjugateGradientState`, that now correctly
  state that `p` is no longer stored, but the algorithm runs on `TpM`.
* implemented the missing `get_iterate` for `TruncatedConjugateGradientState`.

## [0.4.57] March 15, 2024

### Changed

* `convex_bundle_method` uses the `sectional_curvature` from `ManifoldsBase.jl`.
* `convex_bundle_method` no longer has the unused `k_min` keyword argument.
* `ManifoldsBase.jl` now is running on Documenter 1.3, `Manopt.jl` documentation now uses [DocumenterInterLinks](https://github.com/JuliaDocs/DocumenterInterLinks.jl) to refer to sections and functions from `ManifoldsBase.jl`

### Fixed

* fixes a type that when passing `sub_kwargs` to `trust_regions` caused an error in the decoration of the sub objective.

## [0.4.56] March 4, 2024

### Added

* The option `:step_towards_negative_gradient` for `nondescent_direction_behavior` in quasi-Newton solvers does no longer emit a warning by default. This has been moved to a `message`, that can be accessed/displayed with `DebugMessages`
* `DebugMessages` now has a second positional argument, specifying whether all messages, or just the first (`:Once`) should be displayed.

## [0.4.55] March 3, 2024

### Added

* Option `nondescent_direction_behavior` for quasi-Newton solvers.
  By default it checks for non-descent direction which may not be handled well by
  some stepsize selection algorithms.

### Fixed

* unified documentation, especially function signatures further.
* fixed a few typos related to math formulae in the doc strings.

## [0.4.54] February 28, 2024

### Added

* `convex_bundle_method` optimization algorithm for non-smooth geodesically convex functions
* `proximal_bundle_method` optimization algorithm for non-smooth functions.
* `StopWhenSubgradientNormLess`, `StopWhenLagrangeMultiplierLess`, and stopping criteria.

### Fixed

* Doc strings now follow a [vale.sh](https://vale.sh) policy. Though this is not fully working,
  this PR improves a lot of the doc strings concerning wording and spelling.

## [0.4.53] February 13, 2024

### Fixed

* fixes two storage action defaults, that accidentally still tried to initialize a `:Population` (as modified back to `:Iterate` 0.4.49).
* fix a few typos in the documentation and add a reference for the subgradient method.

## [0.4.52] February 5, 2024

### Added

* introduce an environment persistent way of setting global values with the `set_manopt_parameter!` function using [Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl).
* introduce such a value named `:Mode` to enable a `"Tutorial"` mode that shall often provide more warnings and information for people getting started with optimisation on manifolds

## [0.4.51] January 30, 2024

### Added

* A `StopWhenSubgradientNormLess` stopping criterion for subgradient-based optimization.
* Allow the `message=` of the `DebugIfEntry` debug action to contain a format element to print the field in the message as well.

## [0.4.50] January 26, 2024

### Fixed

* Fix Quasi Newton on complex manifolds.

## [0.4.49] January 18, 2024

### Added

* A `StopWhenEntryChangeLess` to be able to stop on arbitrary small changes of specific fields
* generalises `StopWhenGradientNormLess` to accept arbitrary `norm=` functions
* refactor the default in `particle_swarm` to no longer “misuse” the iteration change,
  but actually the new one the `:swarm` entry

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
* Quasi Newton Updates can work in-place of a direction vector as well.
* Faster `safe_indices` in L-BFGS.

## [0.4.44] December 12, 2023

Formally one could consider this version breaking, since a few functions
have been moved, that in earlier versions (0.3.x) have been used in example scripts.
These examples are now available again within [ManoptExamples.jl](https://juliamanifolds.github.io/ManoptExamples.jl/stable/), and with their
“reappearance” the corresponding costs, gradients, differentials, adjoint differentials, and proximal maps
have been moved there as well.
This is not considered breaking, since the functions were only used in the old, removed examples.
Each and every moved function is still documented. They have been partly renamed,
and their documentation and testing has been extended.

### Changed

* Bumped and added dependencies on all 3 Project.toml files, the main one, the docs/, an the tutorials/ one.
* `artificial_S2_lemniscate` is available as [`ManoptExample.Lemniscate`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/data/#ManoptExamples.Lemniscate-Tuple{Number}) and works on arbitrary manifolds now.
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
* `cost_acceleration_bezier` is available as [`ManoptExamples.acceleration_Bezier`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.acceleration_Bezier-Union{Tuple{P},%20Tuple{ManifoldsBase.AbstractManifold,%20AbstractVector{P},%20AbstractVector{%3C:Integer},%20AbstractVector{%3C:AbstractFloat}}}%20where%20P)
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
* `grad_Intrinsic_infimal_convolution_TV12` is available as [`ManoptExamples.Intrinsic_infimal_convolution_TV12`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.grad_intrinsic_infimal_convolution_TV12-Tuple{ManifoldsBase.AbstractManifold,%20Vararg{Any,%205}})
* `grad_TV` is available as [`ManoptExamples.grad_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.grad_Total_Variation)
* `costIntrICTV12` is available as [`ManoptExamples.Intrinsic_infimal_convolution_TV12`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.Intrinsic_infimal_convolution_TV12-Tuple{ManifoldsBase.AbstractManifold,%20Vararg{Any,%205}})
* `project_collaborative_TV` is available as [`ManoptExamples.project_collaborative_TV`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.project_collaborative_TV)
* `prox_parallel_TV` is available as [`ManoptExamples.prox_parallel_TV`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.prox_parallel_TV)
* `grad_TV2` is available as [`ManoptExamples.prox_second_order_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.grad_second_order_Total_Variation)
* `prox_TV` is available as [`ManoptExamples.prox_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.prox_Total_Variation)
* `prox_TV2` is available as [`ManopExamples.prox_second_order_Total_Variation`](https://juliamanifolds.github.io/ManoptExamples.jl/stable/objectives/#ManoptExamples.prox_second_order_Total_Variation-Union{Tuple{T},%20Tuple{ManifoldsBase.AbstractManifold,%20Any,%20Tuple{T,%20T,%20T}},%20Tuple{ManifoldsBase.AbstractManifold,%20Any,%20Tuple{T,%20T,%20T},%20Int64}}%20where%20T)

## [0.4.43] November 19, 2023

### Added

* vale.sh as a CI to keep track of a consistent documentation

## [0.4.42] November 6, 2023

### Added

* add `Manopt.JuMP_Optimizer` implementing JuMP's solver interface

## [0.4.41] November 2, 2023

### Changed

* `trust_regions` is now more flexible and the sub solver (Steihaug-Toint tCG by default)
  can now be exchanged.
* `adaptive_regularization_with_cubics` is now more flexible as well, where it previously was a bit too
  much tightened to the Lanczos solver as well.
* Unified documentation notation and bumped dependencies to use DocumenterCitations 1.3

## [0.4.40] October 24, 2023

### Added

* add a `--help` argument to `docs/make.jl` to document all available command line arguments
* add a `--exclude-tutorials` argument to `docs/make.jl`. This way, when quarto is not available
  on a computer, the docs can still be build with the tutorials not being added to the menu
  such that documenter does not expect them to exist.

### Changes

* Bump dependencies to `ManifoldsBase.jl` 0.15 and `Manifolds.jl` 0.9
* move the ARC CG subsolver to the main package, since `TangentSpace` is now already
  available from `ManifoldsBase`.

## [0.4.39] October 9, 2023

### Changes

* also use the pair of a retraction and the inverse retraction (see last update)
  to perform the relaxation within the Douglas-Rachford algorithm.

## [0.4.38] October 8, 2023

### Changes

* avoid allocations when calling `get_jacobian!` within the Levenberg-Marquard Algorithm.

### Fixed

* Fix a lot of typos in the documentation

## [0.4.37] September 28, 2023

### Changes

* add more of the Riemannian Levenberg-Marquard algorithms parameters as keywords, so they
  can be changed on call
* generalize the internal reflection of Douglas-Rachford, such that is also works with an
  arbitrary pair of a reflection and an inverse reflection.

## [0.4.36]  September 20, 2023

### Fixed

* Fixed a bug that caused non-matrix points and vectors to fail when working with approximate

## [0.4.35]  September 14, 2023

### Added

* The access to functions of the objective is now unified and encapsulated in proper `get_` functions.

## [0.4.34]  September 02, 2023

### Added

* an `ManifoldEuclideanGradientObjective` to allow the cost, gradient, and Hessian and other
  first or second derivative based elements to be Euclidean and converted when needed.
* a keyword `objective_type=:Euclidean` for all solvers, that specifies that an Objective shall be created of the new type

## [0.4.33] August 24, 2023

### Added

* `ConstantStepsize` and `DecreasingStepsize` now have an additional field `type::Symbol` to assess whether the
  step-size should be relatively (to the gradient norm) or absolutely constant.

## [0.4.32] August 23, 2023

### Added

* The adaptive regularization with cubics (ARC) solver.

## [0.4.31] August 14, 2023

### Added

* A `:Subsolver` keyword in the `debug=` keyword argument, that activates the new `DebugWhenActive``
  to de/activate subsolver debug from the main solvers`DebugEvery`.

## [0.4.30] August 3, 2023

### Changed

* References in the documentation are now rendered using [DocumenterCitations.jl](https://github.com/JuliaDocs/DocumenterCitations.jl)
* Asymptote export now also accepts a size in pixel instead of its default `4cm` size and `render` can be deactivated setting it to `nothing`.

## [0.4.29] July 12, 2023

### Fixed

* fixed a bug, where `cyclic_proximal_point` did not work with decorated objectives.

## [0.4.28] June 24, 2023

### Changed

* `max_stepsize` was specialized for `FixedRankManifold` to follow Matlab Manopt.

## [0.4.27] June 15, 2023

### Added

* The `AdaptiveWNGrad` stepsize is available as a new stepsize functor.

### Fixed

* Levenberg-Marquardt now possesses its parameters `initial_residual_values` and
  `initial_jacobian_f` also as keyword arguments, such that their default initialisations
  can be adapted, if necessary

## [0.4.26] June 11, 2023

### Added

* simplify usage of gradient descent as sub solver in the DoC solvers.
* add a `get_state` function
* document `indicates_convergence`.

## [0.4.25] June 5, 2023

### Fixed

* Fixes an allocation bug in the difference of convex algorithm

## [0.4.24] June 4, 2023

### Added

* another workflow that deletes old PR renderings from the docs to keep them smaller in overall size.

### Changes

* bump dependencies since the extension between Manifolds.jl and ManifoldsDiff.jl has been moved to Manifolds.jl

## [0.4.23] June 4, 2023

### Added

* More details on the Count and Cache tutorial

### Changed

* loosen constraints slightly

## [0.4.22] May 31, 2023

### Added

* A tutorial on how to implement a solver

## [0.4.21] May 22, 2023

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

## [0.4.20] May 11, 2023

### Changed

* the default tolerances for the numerical `check_` functions were loosened a bit,
  such that `check_vector` can also be changed in its tolerances.

## [0.4.19] May 7, 2023

### Added

* the sub solver for `trust_regions` is now customizable and can now be exchanged.

### Changed

* slightly changed the definitions of the solver states for ALM and EPM to be type stable

## [0.4.18] May 4, 2023

### Added

* A function `check_Hessian(M, f, grad_f, Hess_f)` to numerically verify the (Riemannian) Hessian of a function `f`

## [0.4.17] April 28, 2023

### Added

* A new interface of the form `alg(M, objective, p0)` to allow to reuse
  objectives without creating `AbstractManoptSolverState`s and calling `solve!`. This especially still allows for any decoration of the objective and/or the state using `debug=`, or `record=`.

### Changed

* All solvers now have the initial point `p` as an optional parameter making it more accessible to first time users, `gradient_descent(M, f, grad_f)` is equivalent to `gradient_descent(M, f, grad_f, rand(M))`

### Fixed

* Unified the framework to work on manifold where points are represented by numbers for several solvers

## [0.4.16] April 18, 2023

### Fixed

* the inner products used in `truncated_gradient_descent` now also work thoroughly on complex
  matrix manifolds

## [0.4.15] April 13, 2023

### Changed

* `trust_regions(M, f, grad_f, hess_f, p)` now has the Hessian `hess_f` as well as
  the start point `p0` as an optional parameter and approximate it otherwise.
* `trust_regions!(M, f, grad_f, hess_f, p)` has the Hessian as an optional parameter
  and approximate it otherwise.

### Removed

* support for `ManifoldsBase.jl` 0.13.x, since with the definition of `copy(M,p::Number)`,
  in 0.14.4, that one is used instead of defining it ourselves.

## [0.4.14] April 06, 2023

### Changed

* `particle_swarm` now uses much more in-place operations

### Fixed

* `particle_swarm` used quite a few `deepcopy(p)` commands still, which were replaced by `copy(M, p)`

## [0.4.13] April 09, 2023

### Added

* `get_message` to obtain messages from sub steps of a solver
* `DebugMessages` to display the new messages in debug
* safeguards in Armijo line search and L-BFGS against numerical over- and underflow that report in messages

## [0.4.12] April 4, 2023

### Added

* Introduce the [Difference of Convex Algorithm](https://manoptjl.org/stable/solvers/difference_of_convex/#DCASolver) (DCA)
  `difference_of_convex_algorithm(M, f, g, ∂h, p0)`
* Introduce the [Difference of Convex Proximal Point Algorithm](https://manoptjl.org/stable/solvers/difference_of_convex/#DCPPASolver) (DCPPA)
  `difference_of_convex_proximal_point(M, prox_g, grad_h, p0)`
* Introduce a `StopWhenGradientChangeLess` stopping criterion

## [0.4.11] March 27, 2023

### Changed

* adapt tolerances in tests to the speed/accuracy optimized distance on the sphere in `Manifolds.jl` (part II)

## [0.4.10] March 26, 2023

### Changed

* adapt tolerances in tests to the speed/accuracy optimized distance on the sphere in `Manifolds.jl`

## [0.4.9] March 3, 2023

### Added

* introduce a wrapper that allows line searches from [LineSearches.jl](https://github.com/JuliaNLSolvers/LineSearches.jl)
  to be used within Manopt.jl, introduce the [manoptjl.org/stable/extensions/](https://manoptjl.org/stable/extensions/)
  page to explain the details.

## [0.4.8] February 21, 2023

### Added

* a `status_summary` that displays the main parameters within several structures of Manopt,
  most prominently a solver state

### Changed

* Improved storage performance by introducing separate named tuples for points and vectors
* changed the `show` methods of `AbstractManoptSolverState`s to display their `state_summary
* Move tutorials to be rendered with Quarto into the documentation.

## [0.4.7] February 14, 2023

### Changed

* Bump `[compat]` entry of ManifoldDiff to also include 0.3

## [0.4.6] February 3, 2023

### Fixed

* Fixed a few stopping criteria even indicated to stop before the algorithm started.

## [0.4.5] January 24, 2023

### Changed

* the new default functions that include `p` are used where possible
* a first step towards faster storage handling

## [0.4.4] January 20, 2023

### Added

* Introduce `ConjugateGradientBealeRestart` to allow CG restarts using Beale‘s rule

### Fixed

* fix a type in `HestenesStiefelCoefficient`

## [0.4.3] January 17, 2023

### Fixed

* the CG coefficient `β` can now be complex
* fix a bug in `grad_distance`

## [0.4.2] January 16, 2023

### Changed

* the usage of `inner` in line search methods, such that they work well with
  complex manifolds as well

## [0.4.1] January 15, 2023

### Fixed

* a `max_stepsize` per manifold to avoid leaving the injectivity radius,
  which it also defaults to

## [0.4.0] January 10, 2023

### Added

* Dependency on `ManifoldDiff.jl` and a start of moving actual derivatives, differentials,
  and gradients there.
* `AbstractManifoldObjective` to store the objective within the `AbstractManoptProblem`
* Introduce a `CostGrad` structure to store a function that computes the cost and gradient
  within one function.
* started a `changelog.md` to thoroughly keep track of changes

### Changed

* `AbstractManoptProblem` replaces `Problem`
* the problem now contains a
* `AbstractManoptSolverState` replaces `Options`
* `random_point(M)` is replaced by `rand(M)` from `ManifoldsBase.jl
* `random_tangent(M, p)` is replaced by `rand(M; vector_at=p)`
