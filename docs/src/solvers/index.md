
# [Solvers](@id SolversSection)

```@meta
CurrentModule = Manopt
```

Solvers can be applied to [`AbstractManoptProblem`](@ref)s with solver
specific [`AbstractManoptSolverState`](@ref).

# List of Algorithms

The following algorithms are currently available

| Solver   | Function & State    | Objective   |
|:---------|:----------------|:---------|
[Alternating Gradient Descent](@ref AlternatingGradientDescentSolver) | [`alternating_gradient_descent`](@ref) [`AlternatingGradientDescentState`](@ref) | ``f=(f_1,\ldots,f_n)``, ``\operatorname{grad} f_i`` |
[Chambolle-Pock](@ref ChambollePockSolver) | [`ChambollePock`](@ref), [`ChambollePockState`](@ref) (using [`TwoManifoldProblem`](@ref)) | ``f=F+G(Λ\cdot)``, ``\operatorname{prox}_{σ F}``, ``\operatorname{prox}_{τ G^*}``, ``Λ`` |
[Conjugate Gradient Descent](@ref CGSolver) | [`conjugate_gradient_descent`](@ref), [`ConjugateGradientDescentState`](@ref) | ``f``, ``\operatorname{grad} f``
[Cyclic Proximal Point](@ref CPPSolver) | [`cyclic_proximal_point`](@ref), [`CyclicProximalPointState`](@ref) | ``f=\sum f_i``, ``\operatorname{prox}_{\lambda f_i}`` |
[Difference of Convex Algorithm](@ref DCASolver) | [`difference_of_convex_algorithm`](@ref), [`DifferenceOfConvexState`](@ref) | ``f=g-h``, ``∂h``, and e.g. ``g``, ``\operatorname{grad} g`` |
[Difference of Convex Proximal Point](@ref DCPPASolver) | [`difference_of_convex_proximal_point`](@ref), [`DifferenceOfConvexProximalState`](@ref) | ``f=g-h``, ``∂h``, and e.g. ``g``, ``\operatorname{grad} g`` |
[Douglas–Rachford](@ref DRSolver) | [`DouglasRachford`](@ref), [`DouglasRachfordState`](@ref) | ``f=\sum f_i``, ``\operatorname{prox}_{\lambda f_i}`` |
[Exact Penalty Method](@ref ExactPenaltySolver) | [`exact_penalty_method`](@ref), [`ExactPenaltyMethodState`](@ref) | ``f``, ``\operatorname{grad} f``, ``g``, ``\operatorname{grad} g_i``, ``h``, ``\operatorname{grad} h_j`` |
[Frank-Wolfe algorithm](@ref FrankWolfe) | [`Frank_Wolfe_method`](@ref), [`FrankWolfeState`](@ref) | sub-problem solver |
[Gradient Descent](@ref GradientDescentSolver) | [`gradient_descent`](@ref), [`GradientDescentState`](@ref) | ``f``, ``\operatorname{grad} f`` |
[Levenberg-Marquardt](@ref) | [`LevenbergMarquardt`](@ref), [`LevenbergMarquardtState`](@ref) | ``f = \sum_i f_i`` ``\operatorname{grad} f_i`` (Jacobian)|
[Nelder-Mead](@ref NelderMeadSolver) | [`NelderMead`](@ref), [`NelderMeadState`](@ref) | ``f``
[Augmented Lagrangian Method](@ref AugmentedLagrangianSolver) | [`augmented_Lagrangian_method`](@ref), [`AugmentedLagrangianMethodState`](@ref) | ``f``, ``\operatorname{grad} f``, ``g``, ``\operatorname{grad} g_i``, ``h``, ``\operatorname{grad} h_j`` |
[Particle Swarm](@ref ParticleSwarmSolver) | [`particle_swarm`](@ref), [`ParticleSwarmState`](@ref) | ``f`` |
[Primal-dual Riemannian semismooth Newton Algorithm](@ref PDRSSNSolver) | [`primal_dual_semismooth_Newton`](@ref),  [`PrimalDualSemismoothNewtonState`](@ref) (using [`TwoManifoldProblem`](@ref)) | ``f=F+G(Λ\cdot)``, ``\operatorname{prox}_{σ F}`` & diff., ``\operatorname{prox}_{τ G^*}`` & diff., ``Λ``
[Quasi-Newton Method](@ref quasiNewton) | [`quasi_Newton`](@ref), [`QuasiNewtonState`](@ref) | ``f``, ``\operatorname{grad} f`` |
[Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) | [`truncated_conjugate_gradient_descent`](@ref), [`TruncatedConjugateGradientState`](@ref) | ``f``, ``\operatorname{grad} f``, ``\operatorname{Hess} f`` |
[Subgradient Method](@ref SubgradientSolver) | [`subgradient_method`](@ref), [`SubGradientMethodState`](@ref) | ``f``, ``∂ f`` |
[Stochastic Gradient Descent](@ref StochasticGradientDescentSolver) | [`stochastic_gradient_descent`](@ref), [`StochasticGradientDescentState`](@ref) | ``f = \sum_i f_i``, ``\operatorname{grad} f_i`` |
[The Riemannian Trust-Regions Solver](@ref trust_regions) | [`trust_regions`](@ref), [`TrustRegionsState`](@ref) | ``f``, ``\operatorname{grad} f``, ``\operatorname{Hess} f`` |

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

## API for solvers

this is a short overview of the different types of high-level functions are usually
available for a solver. Let's assume the solver is called `new_solver` and requires
a cost `f` and some first order information `df` as well as a starting point `p` on `M`.
`f` and `df` form the objective together called `obj`.

Then there are basically two different variants to call

### The easy to access call

```
new_solver(M, f, df, p=rand(M); kwargs...)
new_solver!(M, f, df, p; kwargs...)
```

Where the start point should be optional.
Keyword arguments include the type of evaluation, decorators like `debug=` or `record=`
as well as algorithm specific ones.
If you provide an immutable point `p` or the `rand(M)` point is immutable, like on the `Circle()` this method should turn the point into a mutable one as well.

The third variant works in place of `p`, so it is mandatory.

This first interface would set up the objective and pass all keywords on the the
objective based call.

### The objective-based call

```
new_solver(M, obj, p=rand(M); kwargs...)
new_solver!(M, obj, p; kwargs...)
```

Here the objective would be created beforehand, e.g. to compare different solvers on the
same objective, and for the first variant the start point is optional.
Keyword arguments include decorators like `debug=` or `record=`
as well as algorithm specific ones.

this variant would generate the `problem` and the `state` and check validity of all provided
keyword arguments that affect the state.
Then it would call the iterate process.

### The manual call

If you generate the corresponding `problem` and `state` as the previous step does, you can
also use the third (lowest level) and just call

```
solve!(problem, state)
```