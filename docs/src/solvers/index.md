
# List of available solvers

```@meta
CurrentModule = Manopt
```

Solvers can be applied to [`AbstractManoptProblem`](@ref)s with solver
specific [`AbstractManoptSolverState`](@ref).

# List of algorithms

The following algorithms are currently available

| Solver   | Function & State    | Objective   |
|:---------|:----------------|:---------|
[Alternating Gradient Descent](@ref solver-alternating-gradient-descent) | [`alternating_gradient_descent`](@ref) [`AlternatingGradientDescentState`](@ref) | ``f=(f_1,\ldots,f_n)``, ``\operatorname{grad} f_i`` |
[Augmented Lagrangian Method](augmented_Lagrangian_method.md) | [`augmented_Lagrangian_method`](@ref), [`AugmentedLagrangianMethodState`](@ref) | ``f``, ``\operatorname{grad} f``, ``g``, ``\operatorname{grad} g_i``, ``h``, ``\operatorname{grad} h_j`` |
[Chambolle-Pock](ChambollePock.md) | [`ChambollePock`](@ref), [`ChambollePockState`](@ref) (using [`TwoManifoldProblem`](@ref)) | ``f=F+G(Λ\cdot)``, ``\operatorname{prox}_{σ F}``, ``\operatorname{prox}_{τ G^*}``, ``Λ`` |
[Conjugate Gradient Descent](conjugate_gradient_descent.md) | [`conjugate_gradient_descent`](@ref), [`ConjugateGradientDescentState`](@ref) | ``f``, ``\operatorname{grad} f``
[Convex Bundle Method](covex_bundle_method.md) | [`convex_bundle_method`](@ref), [`ConvexBundleMethodState`](@ref) | ``f``, ``\partial f``
[Cyclic Proximal Point](cyclic_proximal_point.md) | [`cyclic_proximal_point`](@ref), [`CyclicProximalPointState`](@ref) | ``f=\sum f_i``, ``\operatorname{prox}_{\lambda f_i}`` |
[Difference of Convex Algorithm](@ref solver-difference-of-convex) | [`difference_of_convex_algorithm`](@ref), [`DifferenceOfConvexState`](@ref) | ``f=g-h``, ``∂h``, and for example ``g``, ``\operatorname{grad} g`` |
[Difference of Convex Proximal Point](@ref solver-difference-of-convex-proximal-point) | [`difference_of_convex_proximal_point`](@ref), [`DifferenceOfConvexProximalState`](@ref) | ``f=g-h``, ``∂h``, and for example ``g``, ``\operatorname{grad} g`` |
[Douglas—Rachford](DouglasRachford.md) | [`DouglasRachford`](@ref), [`DouglasRachfordState`](@ref) | ``f=\sum f_i``, ``\operatorname{prox}_{\lambda f_i}`` |
[Exact Penalty Method](exact_penalty_method.md) | [`exact_penalty_method`](@ref), [`ExactPenaltyMethodState`](@ref) | ``f``, ``\operatorname{grad} f``, ``g``, ``\operatorname{grad} g_i``, ``h``, ``\operatorname{grad} h_j`` |
[Frank-Wolfe algorithm](FrankWolfe.md) | [`Frank_Wolfe_method`](@ref), [`FrankWolfeState`](@ref) | sub-problem solver |
[Gradient Descent](gradient_descent.md) | [`gradient_descent`](@ref), [`GradientDescentState`](@ref) | ``f``, ``\operatorname{grad} f`` |
[Levenberg-Marquardt](LevenbergMarquardt.md) | [`LevenbergMarquardt`](@ref), [`LevenbergMarquardtState`](@ref) | ``f = \sum_i f_i`` ``\operatorname{grad} f_i`` (Jacobian)|
[Nelder-Mead](NelderMead.md) | [`NelderMead`](@ref), [`NelderMeadState`](@ref) | ``f``
[Particle Swarm](particle_swarm.md) | [`particle_swarm`](@ref), [`ParticleSwarmState`](@ref) | ``f`` |
[Primal-dual Riemannian semismooth Newton Algorithm](@ref solver-pdrssn) | [`primal_dual_semismooth_Newton`](@ref),  [`PrimalDualSemismoothNewtonState`](@ref) (using [`TwoManifoldProblem`](@ref)) | ``f=F+G(Λ\cdot)``, ``\operatorname{prox}_{σ F}`` & diff., ``\operatorname{prox}_{τ G^*}`` & diff., ``Λ``
[Proximal Bundle Method](covex_bundle_method.md) | [`prox_bundle_method`](@ref), [`ProxBundleMethodState`](@ref) | ``f``, ``\partial f``
[Quasi-Newton Method](quasi_Newton.md) | [`quasi_Newton`](@ref), [`QuasiNewtonState`](@ref) | ``f``, ``\operatorname{grad} f`` |
[Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) | [`truncated_conjugate_gradient_descent`](@ref), [`TruncatedConjugateGradientState`](@ref) | ``f``, ``\operatorname{grad} f``, ``\operatorname{Hess} f`` |
[Subgradient Method](subgradient.md) | [`subgradient_method`](@ref), [`SubGradientMethodState`](@ref) | ``f``, ``∂ f`` |
[Stochastic Gradient Descent](stochastic_gradient_descent.md) | [`stochastic_gradient_descent`](@ref), [`StochasticGradientDescentState`](@ref) | ``f = \sum_i f_i``, ``\operatorname{grad} f_i`` |
[The Riemannian Trust-Regions Solver](trust_regions.md) | [`trust_regions`](@ref), [`TrustRegionsState`](@ref) | ``f``, ``\operatorname{grad} f``, ``\operatorname{Hess} f`` |

Note that the solvers (their [`AbstractManoptSolverState`](@ref), to be precise) can also be decorated to enhance your algorithm by general additional properties, see [debug output](@ref sec-debug) and [recording values](@ref sec-record). This is done using the `debug=` and `record=` keywords in the function calls. Similarly, since Manopt.jl 0.4 a (simple) [caching of the objective function](@ref subsection-cache-objective) using the `cache=` keyword is available in any of the function calls..

## Technical details

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
available for a solver. Assume the solver is called `new_solver` and requires
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

This first interface would set up the objective and pass all keywords on the
objective based call.

### Objective based calls to solvers

```
new_solver(M, obj, p=rand(M); kwargs...)
new_solver!(M, obj, p; kwargs...)
```

Here the objective would be created beforehand for example to compare different solvers on the
same objective, and for the first variant the start point is optional.
Keyword arguments include decorators like `debug=` or `record=`
as well as algorithm specific ones.

This variant would generate the `problem` and the `state` and verify validity of all provided
keyword arguments that affect the state.
Then it would call the iterate process.

### Manual calls

If you generate the corresponding `problem` and `state` as the previous step does, you can
also use the third (lowest level) and just call

```
solve!(problem, state)
```