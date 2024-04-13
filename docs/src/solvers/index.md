
# Available solvers in Manopt.jl

```@meta
CurrentModule = Manopt
```

Optimisation problems can be classified with respect to several criteria.
In the following we provide a grouping of the algorithms with respect to the “information”
available about your optimisation problem

```math
\operatorname*{arg\,min}_{p∈\mathbb M} f(p)
```

Within the groups we provide short notes on advantages of the individual solvers,
for example whether ``f`` should be convex, or nonconvex – or whether the method works
locally or even globally.

## Derivative Free

For derivative free

## First Order

### Gradient

### Subgradient

## Second Order

## Splitting based

For splitting methods, the algorithms are based on splitting the cost into different parts, usually in a sum of two or more summands.

* The [Alternating Gradient Descent](@ref solver-alternating-gradient-descent) alternates gradient descent steps on the components of the product manifold. All these components should be smooth aso the gradient exists, and (locally) convex.
  [`alternating_gradient_descent`](@ref)`(M::ProductManifold, f, grad_f)`
* The [Chambolle-Pock](ChambollePock.md) algorithm uses a splitting ``f(p) = F(p) + G(Λ(p))``,
  where ``G`` is defined on a manifold ``\mathcal N`` and we need the proximal map of its Fenchel dual. Both these functions can be non-smooth.
  [`ChambollePock`](@ref)`(M, N, cost, x0, ξ0, m, n, prox_F, prox_G_dual, Λ)`

## Constrained

Constrained problems of the form

```math
\begin{align*}
\operatorname*{arg\,min}_{p∈\mathbb M}& f(p)\\
\text{such that} & g(p) \leq 0\\&h(p) = 0
\end{align*}
```

For these you can use

* The [Augmented Lagrangian Method](augmented_Lagrangian_method.md) | [`augmented_Lagrangian_method`](@ref)`(M, f, grad_f, p0)`, where both `g` and `grad_g` as well as `h` and `grad_h` are keyword arguments, and one of these pairs is mandatory.



# Temp – still to sort solvers

The following algorithms are currently available

* [Adaptive Regularisation with Cubics](adaptive-regularization-with-cubics.md), [`adaptive_regularization_with_cubics`](@ref)
* [Conjugate Gradient Descent](conjugate_gradient_descent.md) | [`conjugate_gradient_descent`](@ref)
* [Convex Bundle Method](convex_bundle_method.md) | [`convex_bundle_method`](@ref)
* [Cyclic Proximal Point](cyclic_proximal_point.md) | [`cyclic_proximal_point`](@ref)
* [Difference of Convex Algorithm](@ref solver-difference-of-convex) | [`difference_of_convex_algorithm`](@ref)
* [Difference of Convex Proximal Point](@ref solver-difference-of-convex-proximal-point) | [`difference_of_convex_proximal_point`](@ref)
* [Douglas—Rachford](DouglasRachford.md) | [`DouglasRachford`](@ref)
* [Exact Penalty Method](exact_penalty_method.md) | [`exact_penalty_method`](@ref)
* [Frank-Wolfe algorithm](FrankWolfe.md) | [`Frank_Wolfe_method`](@ref)
* [Gradient Descent](gradient_descent.md) | [`gradient_descent`](@ref)
* [Levenberg-Marquardt](LevenbergMarquardt.md) | [`LevenbergMarquardt`](@ref)
* [Nelder-Mead](NelderMead.md) | [`NelderMead`](@ref)
* [Particle Swarm](particle_swarm.md) | [`particle_swarm`](@ref)
* [Primal-dual Riemannian semismooth Newton Algorithm](@ref solver-pdrssn) | [`primal_dual_semismooth_Newton`](@ref)
* [Proximal Bundle Method](proximal_bundle_method.md) | [`proximal_bundle_method`](@ref)
* [Quasi-Newton Method](quasi_Newton.md) | [`quasi_Newton`](@ref)
* [Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) | [`truncated_conjugate_gradient_descent`](@ref)
* [Subgradient Method](subgradient.md) | [`subgradient_method`](@ref)
* [Stochastic Gradient Descent](stochastic_gradient_descent.md) | [`stochastic_gradient_descent`](@ref), [`StochasticGradientDescentState`](@ref)
* [The Riemannian Trust-Regions Solver](trust_regions.md) | [`trust_regions`](@ref)

# Alphabetical list List of algorithms

| Solver   | Function        | State   |
|:---------|:----------------|:---------|
| [Adaptive Regularisation with Cubics](adaptive-regularization-with-cubics.md) | [`adaptive_regularization_with_cubics`](@ref) | [`AdaptiveRegularizationState`](@ref) |
| [Augmented Lagrangian Method](augmented_Lagrangian_method.md) | [`augmented_Lagrangian_method`](@ref) | [`AugmentedLagrangianMethodState`](@ref) |
| [Chambolle-Pock](ChambollePock.md) | [`ChambollePock`](@ref) | [`ChambollePockState`](@ref) |
| [Conjugate Gradient Descent](conjugate_gradient_descent.md) | [`conjugate_gradient_descent`](@ref) | [`ConjugateGradientDescentState`](@ref) |
| [Convex Bundle Method](convex_bundle_method.md) | [`convex_bundle_method`](@ref) |  [`ConvexBundleMethodState`](@ref) |
| [Cyclic Proximal Point](cyclic_proximal_point.md) | [`cyclic_proximal_point`](@ref) |  [`CyclicProximalPointState`](@ref) |
| [Difference of Convex Algorithm](@ref solver-difference-of-convex) |
[`difference_of_convex_algorithm`](@ref) | [`DifferenceOfConvexState`](@ref) |
| [Difference of Convex Proximal Point](@ref solver-difference-of-convex-proximal-point) | [`difference_of_convex_proximal_point`](@ref) | [`DifferenceOfConvexProximalState`](@ref) |
| [Douglas—Rachford](DouglasRachford.md) | [`DouglasRachford`](@ref) | [`DouglasRachfordState`](@ref) |
| [Exact Penalty Method](exact_penalty_method.md) | [`exact_penalty_method`](@ref) |  [`ExactPenaltyMethodState`](@ref) |
| [Frank-Wolfe algorithm](FrankWolfe.md) | [`Frank_Wolfe_method`](@ref) | [`FrankWolfeState`](@ref) |
| [Gradient Descent](gradient_descent.md) | [`gradient_descent`](@ref) |  [`GradientDescentState`](@ref) |
| [Levenberg-Marquardt](LevenbergMarquardt.md) | [`LevenbergMarquardt`](@ref) | [`LevenbergMarquardtState`](@ref) | ``f = \sum_i f_i`` ``\operatorname{grad} f_i`` (Jacobian)|
| [Nelder-Mead](NelderMead.md) | [`NelderMead`](@ref) | [`NelderMeadState`](@ref) |
| [Particle Swarm](particle_swarm.md) | [`particle_swarm`](@ref) | [`ParticleSwarmState`](@ref) |
[Primal-dual Riemannian semismooth Newton Algorithm](@ref solver-pdrssn) | [`primal_dual_semismooth_Newton`](@ref) | [`PrimalDualSemismoothNewtonState`](@ref) |
| [Proximal Bundle Method](proximal_bundle_method.md) | [`proximal_bundle_method`](@ref) | [`ProximalBundleMethodState`](@ref) |
| [Quasi-Newton Method](quasi_Newton.md) | [`quasi_Newton`](@ref) | [`QuasiNewtonState`](@ref) |
| [Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) | [`truncated_conjugate_gradient_descent`](@ref) | [`TruncatedConjugateGradientState`](@ref) |
| [Subgradient Method](subgradient.md) | [`subgradient_method`](@ref) | [`SubGradientMethodState`](@ref) |
| [Stochastic Gradient Descent](stochastic_gradient_descent.md) | [`stochastic_gradient_descent`](@ref) | [`StochasticGradientDescentState`](@ref) |
| [Riemannian Trust-Regions](trust_regions.md) | [`trust_regions`](@ref) | [`TrustRegionsState`](@ref) |


Note that the solvers (their [`AbstractManoptSolverState`](@ref), to be precise) can also be decorated to enhance your algorithm by general additional properties, see [debug output](@ref sec-debug) and [recording values](@ref sec-record). This is done using the `debug=` and `record=` keywords in the function calls. Similarly, a `cache=` keyword is available in any of the function calls, that wraps the [`AbstractManoptProblem`](@ref) in a cache for certain parts of the objective.

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