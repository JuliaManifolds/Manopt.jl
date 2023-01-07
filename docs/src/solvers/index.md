
# [Solvers](@id SolversSection)

```@meta
CurrentModule = Manopt
```

Solvers can be applied to [`AbstractManoptProblem`](@ref)s with solver
specific [`AbstractManoptSolverState`](@ref).

# List of Algorithms

The following algorithms are currently available

| Solver  | Function   | State |
----------|--------|-------------------|
[Alternating Gradient Descent](@ref AlternatingGradientDescentSolver) | [`alternating_gradient_descent`](@ref) | [`AlternatingGradientDescentState`](@ref) |
[Chambolle-Pock](@ref ChambollePockSolver) | [`ChambollePock`](@ref) | [`TwoManifoldProblem`](@ref), [`ChambollePockState`](@ref) |
[Conjugate Gradient Descent](@ref CGSolver) | [`cyclic_proximal_point`](@ref) | [`CyclicProximalPointState`](@ref) |
[Cyclic Proximal Point](@ref CPPSolver) | [`conjugate_gradient_descent`](@ref) |  [`ConjugateGradientDescentState`](@ref) |
[Douglas–Rachford](@ref DRSolver) | [`DouglasRachford`](@ref) | [`DouglasRachfordState`](@ref) |
[Exact Penalty Method](@ref ExactPenaltySolver) | [`exact_penalty_method`](@ref) |  [`ExactPenaltyMethodState`](@ref) |
[Frank-Wolfe algorithm](@ref FrankWolfe) | [`Frank_Wolfe_method`](@ref) | [`FrankWolfeState`](@ref) |
[Gradient Descent](@ref GradientDescentSolver) | [`gradient_descent`](@ref) |   [`GradientDescentState`](@ref) |
[Levenberg-Marquardt](@ref) | [`LevenbergMarquardt`](@ref) | [`LevenbergMarquardtState`](@ref) |
[Nelder-Mead](@ref NelderMeadSolver) | [`NelderMead`](@ref) | [`NelderMeadState`](@ref) |
[Augmented Lagrangian Method](@ref AugmentedLagrangianSolver) | [`augmented_Lagrangian_method`](@ref)|  [`AugmentedLagrangianMethodState`](@ref) |
[Particle Swarm](@ref ParticleSwarmSolver) | [`particle_swarm`](@ref) | [`ParticleSwarmState`](@ref) |
[Primal-dual Riemannian semismooth Newton Algorithm](@ref PDRSSNSolver) | [`primal_dual_semismooth_Newton`](@ref) | [`TwoManifoldProblem`](@ref), [`PrimalDualSemismoothNewtonState`](@ref) |
[Quasi-Newton Method](@ref quasiNewton) | [`quasi_Newton`](@ref)| [`QuasiNewtonState`](@ref) |
[Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) | [`truncated_conjugate_gradient_descent`](@ref) |
[Subgradient Method](@ref SubgradientSolver) | [`subgradient_method`](@ref) | [`SubGradientMethodState`](@ref) |
[Stochastic Gradient Descent](@ref StochasticGradientDescentSolver) | [`stochastic_gradient_descent`](@ref) | [`StochasticGradientDescentState`](@ref) |
[The Riemannian Trust-Regions Solver](@ref trust_regions) | [`trust_regions`](@ref) | [`TrustRegionsState`](@ref) |

Note that the solvers (their [`AbstractManoptSolverState`](@ref), to be precise) can also be decorated to enhance your algorithm by general additional properties, see [debug output](@ref DebugSection) and [recording values](@ref RecordSection). This is done using the `debug=` and `record=` keywords in the function calls. Similarly, since 0.4 we provide a (simple) [caching of the objective function](@ref CacheSection) using the `cache=` keyword in any of the function calls..

## Technical Details

 The main function a solver calls is

```@docs
solve!(p::AbstractManoptProblem, s::AbstractManoptSolverState)
```

which is a framework that you in general should not change or redefine.
It uses the following methods, which also need to be implemented on your own
algorithm, if you want to provide one.

```@docs
initialize_solver!
step_solver!
get_solver_result
get_solver_return
stop_solver!(p::AbstractManoptProblem, s::AbstractManoptSolverState, Any)
```
