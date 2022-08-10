
# [Solvers](@id SolversSection)

```@meta
CurrentModule = Manopt
```

Solvers can be applied to [`Problem`](@ref)s with solver
specific [`Options`](@ref).

# List of Algorithms

The following algorithms are currently available

| Solver  | File   | Problem & Option  |
----------|--------|-------------------|
[Alternating Gradient Descent](@ref AlternatingGradientDescentSolver) | `alterating_gradient_descent.jl` | [`AlternatingGradientProblem`](@ref), [`AlternatingGradientDescentOptions`](@ref)
[Chambolle-Pock](@ref ChambollePockSolver) | `Chambolle-Pock.jl` | [`PrimalDualProblem`](@ref), [`ChambollePockOptions`](@ref)
[Cyclic Proximal Point](@ref CPPSolver) | `cyclic_proximal_point.jl` | [`ProximalProblem`](@ref), [`CyclicProximalPointOptions`](@ref)
[Douglas–Rachford](@ref DRSolver) | `DouglasRachford.jl` | [`ProximalProblem`](@ref), [`DouglasRachfordOptions`](@ref)
[Gradient Descent](@ref GradientDescentSolver) | `gradient_descent.jl` |  [`GradientProblem`](@ref), [`GradientDescentOptions`](@ref)
[Frank-Wolfe algorithm](@ref FrankWolfe) | `FrankWolfe.jl` |  [`GradientProblem`](@ref), [`FrankWolfeOptions`](@ref)
[Nelder-Mead](@ref NelderMeadSolver) | `NelderMead.jl` | [`CostProblem`](@ref), [`NelderMeadOptions`](@ref)
[Particle Swarm](@ref ParticleSwarmSolver) | `particle_swarm.jl` | [`CostProblem`](@ref), [`ParticleSwarmOptions`](@ref)
[Primal-dual Riemannian semismooth Newton Algorithm](@ref PDRSSNSolver) | | [`PrimalDualSemismoothNewtonProblem`](@ref) | [`PrimalDualSemismoothNewtonOptions`](@ref)
[Quasi-Newton Method](@ref quasiNewton) | `quasi_newton.jl`| [`GradientProblem`](@ref), [`QuasiNewtonOptions`](@ref)
[Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) | `truncated_conjugate_gradient_descent.jl` | [`HessianProblem`](@ref), [Subgradient Method](@ref SubgradientSolver) | `subgradient_method.jl` | [`SubGradientProblem`](@ref), [`SubGradientMethodOptions`](@ref)
[`TruncatedConjugateGradientOptions`](@ref)
[The Riemannian Trust-Regions Solver](@ref trust_regions) | `trust_regions.jl` | [`HessianProblem`](@ref), [`TrustRegionsOptions`](@ref)

Note that the solvers (or their [`Options`](@ref) to be precise) can also be decorated to enhance your algorithm by general additional properties, see [debug output](@ref DebugSection) and [recording values](@ref RecordSection).

## Technical Details

 The main function a solver calls is

```@docs
solve(p::Problem, o::Options)
```

which is a framework, that you in general should not change or redefine.
It uses the following methods, which also need to be implemented on your own
algorithm, if you want to provide one.

```@docs
initialize_solver!
step_solver!
get_solver_result
stop_solver!(p::Problem, o::Options, i::Int)
```
