
# Available solvers in Manopt.jl

```@meta
CurrentModule = Manopt
```

Optimisation problems can be classified with respect to several criteria.
The following list of the algorithms is a grouped with respect to the “information”
available about a optimisation problem

```math
\operatorname*{arg\,min}_{p∈\mathbb M} f(p)
```

Within each group short notes on advantages of the individual solvers, and required properties the cost ``f`` should have, are provided.
In that list a 🏅 is used to indicate state-of-the-art solvers, that usually perform best in their corresponding group and 🫏 for a maybe not so fast, maybe not so state-of-the-art method, that nevertheless gets the job done most reliably.

## Derivative free

For derivative free only function evaluations of ``f`` are used.

* [Nelder-Mead](NelderMead.md) a simplex based variant, that is using ``d+1`` points, where ``d`` is the dimension of the manifold.
* [Particle Swarm](particle_swarm.md) 🫏 use the evolution of a set of points, called swarm, to explore the domain of the cost and find a minimizer.
* [Mesh adaptive direct search](mesh_adaptive_direct_search.md) performs a mesh based exploration (poll) and search.
* [CMA-ES](cma_es.md) uses a stochastic evolutionary strategy to perform minimization robust to local minima of the objective.

## First order

### Gradient

* [Gradient Descent](gradient_descent.md) uses the gradient from ``f`` to determine a descent direction. Here, the direction can also be changed to be Averaged, Momentum-based, based on Nesterovs rule.
* [Conjugate Gradient Descent](conjugate_gradient_descent.md) uses information from the previous descent direction to improve the current (gradient-based) one including several such update rules.
* The [Quasi-Newton Method](quasi_Newton.md) 🏅 uses gradient evaluations to approximate the Hessian, which is then used in a Newton-like scheme, where both a limited memory and a full Hessian approximation are available with several different update rules.
* [Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) a solver for a constrained problem defined on a tangent space.

### Subgradient

The following methods require the Riemannian subgradient ``∂f`` to be available.
While the subgradient might be set-valued, the function should provide one of the subgradients.

* The [Subgradient Method](subgradient.md) takes the negative subgradient as a step direction and can be combined with a step size.
* The [Convex Bundle Method](convex_bundle_method.md) (CBM) uses a former collection of sub gradients at the previous iterates and iterate candidates to solve a local approximation to `f` in every iteration by solving a quadratic problem in the tangent space.
* The [Proximal Bundle Method](proximal_bundle_method.md) works similar to CBM, but solves a proximal map-based problem in every iteration.

## Second order

* [Adaptive Regularisation with Cubics](adaptive-regularization-with-cubics.md) 🏅 locally builds a cubic model to determine the next descent direction.
* The [Riemannian Trust-Regions Solver](trust_regions.md) builds a quadratic model within a trust region to determine the next descent direction.

## Splitting based

For splitting methods, the algorithms are based on splitting the cost into different parts, usually in a sum of two or more summands.
This is usually very well tailored for non-smooth objectives.

### Smooth

The following methods require that the splitting, for example into several summands, is smooth in the sense that for every summand of the cost, the gradient should still exist everywhere

* [Levenberg-Marquardt](LevenbergMarquardt.md) minimizes the square norm of ``f: \mathcal M→ℝ^d`` provided the gradients of the component functions, or in other words the Jacobian of ``f``.
* [Stochastic Gradient Descent](stochastic_gradient_descent.md) is based on a splitting of ``f`` into a sum of several components ``f_i`` whose gradients are provided. Steps are performed according to gradients of randomly selected components.
* The [Alternating Gradient Descent](@ref solver-alternating-gradient-descent) alternates gradient descent steps on the components of the product manifold. All these components should be smooth as it is required, that the gradient exists, and is (locally) convex.

### Nonsmooth

If the gradient does not exist everywhere, that is if the splitting yields summands that are nonsmooth, usually methods based on proximal maps are used.

* The [Chambolle-Pock](ChambollePock.md) algorithm uses a splitting ``f(p) = F(p) + G(Λ(p))``,
  where ``G`` is defined on a manifold ``\mathcal N`` and the proximal map of its Fenchel dual is required.
  Both these functions can be non-smooth.
* The [Cyclic Proximal Point](cyclic_proximal_point.md) 🫏 uses proximal maps of the functions from splitting ``f`` into summands ``f_i``
* [Difference of Convex Algorithm](@ref solver-difference-of-convex) (DCA) uses a splitting of the (non-convex) function ``f = g - h`` into a difference of two functions; for each of these it is required to have access to the gradient of ``g`` and the subgradient of ``h`` to state a sub problem in every iteration to be solved.
* [Difference of Convex Proximal Point](@ref solver-difference-of-convex-proximal-point) uses a splitting of the (non-convex) function ``f = g - h`` into a difference of two functions; provided the proximal map of ``g`` and the subgradient of ``h``, the next iterate is computed. Compared to DCA, the corresponding sub problem is here written in a form that yields the proximal map.
* [Douglas—Rachford](DouglasRachford.md) uses a splitting ``f(p) = F(x) + G(x)`` and their proximal maps to compute a minimizer of ``f``, which can be non-smooth.
* [Primal-dual Riemannian semismooth Newton Algorithm](@ref solver-pdrssn) extends Chambolle-Pock and requires the differentials of the proximal maps additionally.
* The [Proximal Point](proximal_point.md) uses the proximal map of ``f`` iteratively.

## Constrained

Constrained problems of the form

```math
\begin{align*}
\operatorname*{arg\,min}_{p∈\mathbb M}& f(p)\\
\text{such that } & g(p) \leq 0\\&h(p) = 0
\end{align*}
```

For these you can use

* The [Augmented Lagrangian Method](augmented_Lagrangian_method.md) (ALM), where both `g` and `grad_g` as well as `h` and `grad_h` are keyword arguments, and one of these pairs is mandatory.
* The [Exact Penalty Method](exact_penalty_method.md) (EPM) uses a penalty term instead of augmentation, but has the same interface as ALM.
* The [Interior Point Newton Method](interior_point_Newton.md) (IPM) rephrases the KKT system of a constrained problem into an Newton iteration being performed in every iteration.
* [Frank-Wolfe algorithm](FrankWolfe.md), where besides the gradient of ``f`` either a closed form solution or a (maybe even automatically generated) sub problem solver for ``\operatorname*{arg\,min}_{q ∈ C} ⟨\operatorname{grad} f(p_k), \log_{p_k}q⟩`` is required, where ``p_k`` is a fixed point on the manifold (changed in every iteration).
* [Gradient Projection Method](projected_gradient_method.md)
## On the tangent space

* [Conjugate Residual](conjugate_residual.md) a solver for a linear system ``\mathcal A[X] + b = 0`` on a tangent space.
* [Steihaug-Toint Truncated Conjugate-Gradient Method](truncated_conjugate_gradient_descent.md) a solver for a constrained problem defined on a tangent space.


## Alphabetical list of algorithms

| Solver   | Function        | State   |
|:---------|:----------------|:---------|
| [Adaptive Regularisation with Cubics](adaptive-regularization-with-cubics.md) | [`adaptive_regularization_with_cubics`](@ref) | [`AdaptiveRegularizationState`](@ref) |
| [Augmented Lagrangian Method](augmented_Lagrangian_method.md) | [`augmented_Lagrangian_method`](@ref) | [`AugmentedLagrangianMethodState`](@ref) |
| [Chambolle-Pock](ChambollePock.md) | [`ChambollePock`](@ref) | [`ChambollePockState`](@ref) |
| [Conjugate Gradient Descent](conjugate_gradient_descent.md) | [`conjugate_gradient_descent`](@ref) | [`ConjugateGradientDescentState`](@ref) |
| [Conjugate Residual](conjugate_residual.md) | [`conjugate_residual`](@ref) | [`ConjugateResidualState`](@ref) |
| [Convex Bundle Method](convex_bundle_method.md) | [`convex_bundle_method`](@ref) |  [`ConvexBundleMethodState`](@ref) |
| [Cyclic Proximal Point](cyclic_proximal_point.md) | [`cyclic_proximal_point`](@ref) |  [`CyclicProximalPointState`](@ref) |
| [Difference of Convex Algorithm](@ref solver-difference-of-convex) | [`difference_of_convex_algorithm`](@ref) | [`DifferenceOfConvexState`](@ref) |
| [Difference of Convex Proximal Point](@ref solver-difference-of-convex-proximal-point) | [`difference_of_convex_proximal_point`](@ref) | [`DifferenceOfConvexProximalState`](@ref) |
| [Douglas—Rachford](DouglasRachford.md) | [`DouglasRachford`](@ref) | [`DouglasRachfordState`](@ref) |
| [Exact Penalty Method](exact_penalty_method.md) | [`exact_penalty_method`](@ref) |  [`ExactPenaltyMethodState`](@ref) |
| [Frank-Wolfe algorithm](FrankWolfe.md) | [`Frank_Wolfe_method`](@ref) | [`FrankWolfeState`](@ref) |
| [Gradient Descent](gradient_descent.md) | [`gradient_descent`](@ref) |  [`GradientDescentState`](@ref) |
| [Interior Point Newton](interior_point_Newton.md) | [`interior_point_Newton`](@ref) | |
| [Levenberg-Marquardt](LevenbergMarquardt.md) | [`LevenbergMarquardt`](@ref) | [`LevenbergMarquardtState`](@ref) | ``f = \sum_i f_i`` ``\operatorname{grad} f_i`` (Jacobian)|
| [Nelder-Mead](NelderMead.md) | [`NelderMead`](@ref) | [`NelderMeadState`](@ref) |
| [Particle Swarm](particle_swarm.md) | [`particle_swarm`](@ref) | [`ParticleSwarmState`](@ref) |
[Primal-dual Riemannian semismooth Newton Algorithm](@ref solver-pdrssn) | [`primal_dual_semismooth_Newton`](@ref) | [`PrimalDualSemismoothNewtonState`](@ref) |
| [Proximal Bundle Method](proximal_bundle_method.md) | [`proximal_bundle_method`](@ref) | [`ProximalBundleMethodState`](@ref) |
| [Proximal Point](proximal_point.md) | [`proximal_point`](@ref) |  [`ProximalPointState`](@ref) |
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

### Closed-form sub solvers

If a subsolver solution is available in closed form, `ClosedFormSubSolverState` is used to indicate that.

```@docs
Manopt.ClosedFormSubSolverState
```
