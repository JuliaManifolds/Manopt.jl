
# [Solvers](@id SolversSection)

```@meta
CurrentModule = Manopt
```

Solvers can be applied to [`AbstractManoptProblem`](@ref)s with solver
specific [`AbstractManoptSolverState`](@ref).

# List of Algorithms

The following algorithms are currently available

| Solver  | File   | Problem & Option  |
----------|--------|-------------------|
[Alternating Gradient Descent](@ref AlternatingGradientDescentSolver) | `alterating_gradient_descent.jl` | [`AlternatingGradientProblem`](@ref), [`AlternatingGradientDescentState`](@ref)
[Chambolle-Pock](@ref ChambollePockSolver) | `Chambolle-Pock.jl` | [`PrimalDualProblem`](@ref), [`ChambollePockState`](@ref)
[Conjugate Gradient Descent](@ref CGSolver) | `cyclic_proximal_point.jl` | [`ProximalProblem`](@ref), [`CyclicProximalPointState`](@ref)
[Cyclic Proximal Point](@ref CPPSolver) | `conjugate_gradient_descent.jl` | [`DefaultManoptProblem`](@ref), [`ConjugateGradientDescentState`](@ref)
[Douglas–Rachford](@ref DRSolver) | `DouglasRachford.jl` | [`ProximalProblem`](@ref), [`DouglasRachfordState`](@ref)
[Exact Penalty Method](@ref ExactPenaltySolver) | `exact_penalty_method.jl`| [`ConstrainedProblem`](@ref), [`ExactPenaltyMethodState`](@ref)
[Frank-Wolfe algorithm](@ref FrankWolfe) | `FrankWolfe.jl` |  [`DefaultManoptProblem`](@ref), [`FrankWolfeState`](@ref)
[Gradient Descent](@ref GradientDescentSolver) | `gradient_descent.jl` |  [`DefaultManoptProblem`](@ref), [`GradientDescentState`](@ref)
[Nelder-Mead](@ref NelderMeadSolver) | `NelderMead.jl` | [`CostProblem`](@ref), [`NelderMeadState`](@ref)
[Augmented Lagrangian Method](@ref AugmentedLagrangianSolver) | `augmented_Lagrangian_method.jl`| [`ConstrainedProblem`](@ref), [`AugmentedLagrangianMethodState`](@ref)
[Particle Swarm](@ref ParticleSwarmSolver) | `particle_swarm.jl` | [`CostProblem`](@ref), [`ParticleSwarmState`](@ref)
[Primal-dual Riemannian semismooth Newton Algorithm](@ref PDRSSNSolver) | | [`PrimalDualSemismoothNewtonProblem`](@ref) | [`PrimalDualSemismoothNewtonState`](@ref)
[Quasi-Newton Method](@ref quasiNewton) | `quasi_newton.jl`| [`DefaultManoptProblem`](@ref), [`QuasiNewtonState`](@ref)
[Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) | `truncated_conjugate_gradient_descent.jl` | [`HessianProblem`](@ref)
[Subgradient Method](@ref SubgradientSolver) | `subgradient_method.jl` | [`SubDefaultManoptProblem`](@ref), [`SubGradientMethodState`](@ref)
[Stochastic Gradient Descent](@ref StochasticGradientDescentSolver) | `stochastic_gradient_descent.jl` | [`StochasticDefaultManoptProblem`](@ref), [`StochasticGradientDescentState`](@ref)
[`TruncatedConjugateGradientState`](@ref)
[The Riemannian Trust-Regions Solver](@ref trust_regions) | `trust_regions.jl` | [`HessianProblem`](@ref), [`TrustRegionsState`](@ref)

Note that the solvers (or their [`AbstractSolverState`](@ref), to be precise) can also be decorated to enhance your algorithm by general additional properties, see [debug output](@ref DebugSection) and [recording values](@ref RecordSection).

## Technical Details

 The main function a solver calls is

```@docs
solve!(p::AbstractManoptProblem, o::AbstractSolverState)
```

which is a framework that you in general should not change or redefine.
It uses the following methods, which also need to be implemented on your own
algorithm, if you want to provide one.

```@docs
initialize_solver!
step_solver!
get_solver_result
get_solver_return
stop_solver!(p::AbstractManoptProblem, o::AbstractSolverState, i::Int)
```
