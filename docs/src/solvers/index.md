# Available solvers in Manopt.jl

```@meta
CurrentModule = Manopt
```

Optimization problems can be classified with respect to several criteria.
The following list of the algorithms is grouped with respect to the “information”
available about an optimization problem

```math
\operatorname*{arg\,min}_{p∈\mathcal M} f(p)
```

Within each group, short notes on advantages of the individual solvers, and required properties the cost ``f`` should have, are provided.
In that list a 🏅 is used to indicate state-of-the-art solvers that usually perform best in their corresponding group, and 🫏 for a maybe not so fast, maybe not so state-of-the-art method that nevertheless gets the job done most reliably.

## Derivative free

For derivative free solvers only function evaluations of ``f`` are used.

* [Nelder-Mead](NelderMead.md) a simplex-based variant that uses ``d+1`` points, where ``d`` is the dimension of the manifold.
* [Particle Swarm](particle_swarm.md) 🫏 uses the evolution of a set of points, called swarm, to explore the domain of the cost and find a minimizer.
* [Mesh adaptive direct search](mesh_adaptive_direct_search.md) performs a mesh based exploration (poll) and search.
* [CMA-ES](cma_es.md) uses a stochastic evolutionary strategy to perform minimization robust to local minima of the objective.

## First order

### Gradient

* [Gradient Descent](gradient_descent.md) uses the gradient from ``f`` to determine a descent direction. Here, the direction can also be changed to be averaged, momentum-based, or based on Nesterov's rule.
* [Conjugate Gradient Descent](conjugate_gradient_descent.md) uses information from the previous descent direction to improve the current (gradient-based) one including several such update rules.
* The [Quasi-Newton Method](quasi_Newton.md) 🏅 uses gradient evaluations to approximate the Hessian, which is then used in a Newton-like scheme, where both a limited memory and a full Hessian approximation are available with several different update rules.

### Subgradient

The following methods require the Riemannian subgradient ``∂f`` to be available.
While the subdifferential might be set-valued, the function should provide one of the subgradients.

* The [Subgradient Method](subgradient.md) takes the negative subgradient as a step direction and can be combined with a step size.
* The [Convex Bundle Method](convex_bundle_method.md) (CBM) uses a former collection of subgradients at the previous iterates and iterate candidates to solve a local approximation to ``f`` in every iteration by solving a quadratic problem in the tangent space.
* The [Proximal Bundle Method](proximal_bundle_method.md) works similar to CBM, but solves a proximal map-based problem in every iteration.

## Second order

* [Adaptive Regularization with Cubics](adaptive_regularization_with_cubics.md) 🏅 locally builds a cubic model to determine the next descent direction.
* The [Riemannian Trust-Regions Solver](trust_regions.md) builds a quadratic model within a trust region to determine the next descent direction.

## Splitting based

For splitting methods, the algorithms are based on splitting the cost into different parts, usually in a sum of two or more summands.
This is usually very well tailored for non-smooth objectives.

### Smooth

The following methods require that the splitting, for example into several summands, is smooth in the sense that for every summand of the cost, the gradient should still exist everywhere.

* [Levenberg-Marquardt](LevenbergMarquardt.md) minimizes the squared norm of ``f: \mathcal M→ℝ^d``, that is it solves ``\operatorname*{arg\,min}_{p∈\mathcal M} \frac{1}{2}\lVert f(p) \rVert^2``, where ``\frac{1}{2}\lVert f(p) \rVert^2 = \frac{1}{2}\sum_{i=1}^d f_i(p)^2``, provided the gradients ``\operatorname{grad} f_i`` of the component functions, or in other words the Jacobian of ``f``.
  A robust variant is available as well, where each summand is additionally passed through a
  [robustifier](../commons/robustifiers.md) ``ρ_i``, that is ``\frac{1}{2}\sum_{i=1}^d ρ_i\bigl(f_i(p)^2\bigr)``,
  to reduce the influence of outliers. It is specified with the `robustifier=` keyword.
* [Stochastic Gradient Descent](stochastic_gradient_descent.md) is based on a splitting of ``f`` into a sum of several components ``f_i`` whose gradients are provided. Steps are performed according to gradients of randomly selected components.
* The [Alternating Gradient Descent](@ref solver-alternating-gradient-descent) alternates gradient descent steps on the components of the product manifold. All these components should be smooth, since it is required that their gradients exist, and each component should be (locally) convex.

### Nonsmooth

If the gradient does not exist everywhere, that is if the splitting yields summands that are nonsmooth, usually methods based on proximal maps are used.

* The [Chambolle-Pock](ChambollePock.md) algorithm uses a splitting ``f(p) = F(p) + G(Λ(p))``,
  where ``G`` is defined on a manifold ``\mathcal N`` and the proximal map of its Fenchel dual is required.
  Both these functions can be non-smooth.
* The [Cyclic Proximal Point](cyclic_proximal_point.md) 🫏 uses proximal maps of the functions from splitting ``f`` into summands ``f_i``.
* [Difference of Convex Algorithm](@ref solver-difference-of-convex) (DCA) uses a splitting of the (non-convex) function ``f = g - h`` into a difference of two functions; for each of these it is required to have access to the gradient of ``g`` and the subgradient of ``h`` to state a sub problem in every iteration to be solved.
* [Difference of Convex Proximal Point](@ref solver-difference-of-convex-proximal-point) uses a splitting of the (non-convex) function ``f = g - h`` into a difference of two functions; provided the proximal map of ``g`` and the subgradient of ``h``, the next iterate is computed. Compared to DCA, the corresponding sub problem is here written in a form that yields the proximal map.
* [Douglas-Rachford](DouglasRachford.md) uses a splitting ``f(p) = F(p) + G(p)`` and their proximal maps to compute a minimizer of ``f``, which can be non-smooth.
* The [Gradient Sampling Algorithm](gradient_sampling.md) samples the gradient at points in a ball around the current iterate to build a surrogate and solve that instead to find a next iterate.
* [Primal-dual Riemannian semismooth Newton Algorithm](@ref solver-pdrssn) extends Chambolle-Pock and requires the differentials of the proximal maps additionally.
* The [Proximal Gradient Method](proximal_gradient_method.md) uses a splitting ``f = g + h`` into a smooth ``g``, whose gradient is required, and a nonsmooth ``h``, whose proximal map is required.
* The [Proximal Point](proximal_point.md) uses the proximal map of ``f`` iteratively.

## Constrained

Constrained problems have the form

```math
\begin{align*}
\operatorname*{arg\,min}_{p∈\mathcal M}& f(p)\\
\text{such that } & g(p) \leq 0\\&h(p) = 0
\end{align*}
```

For these you can use

* The [Augmented Lagrangian Method](augmented_Lagrangian_method.md) (ALM), where both `g` and `grad_g` as well as `h` and `grad_h` are keyword arguments, and one of these pairs is mandatory.
* The [Exact Penalty Method](exact_penalty_method.md) (EPM) uses a penalty term instead of augmentation, but has the same interface as ALM.
* The [Interior Point Newton Method](interior_point_Newton.md) (IPM) rephrases the KKT system of a constrained problem as a Newton step that is performed in every iteration.
* [Frank-Wolfe algorithm](FrankWolfe.md), where besides the gradient of ``f`` either a closed form solution or a (maybe even automatically generated) sub problem solver for ``\operatorname*{arg\,min}_{q ∈ C} ⟨\operatorname{grad} f(p^{(k)}), \log_{p^{(k)}}q⟩`` is required, where ``p^{(k)}`` is a fixed point on the manifold (changed in every iteration).
* The [Projected Gradient Method](projected_gradient_method.md) projects the gradient step back onto the feasible set in every iteration.

## On the tangent space

* [Conjugate Residual](conjugate_residual.md) a solver for a linear system ``\mathcal A[X] + b = 0`` on a tangent space.
* [Steihaug-Toint Truncated Conjugate-Gradient Method](truncated_conjugate_gradient_descent.md) a solver for a constrained problem defined on a tangent space.

## Nonlinear equations

The following solver does not minimize a cost, but finds a zero of a mapping into a vector bundle.

* The [Vector Bundle Newton Method](vectorbundle_newton.md) performs Newton's method for a mapping ``F: \mathcal M → \mathcal E`` into a vector bundle ``\mathcal E`` over ``\mathcal M``. In every iteration a Newton equation is solved to determine a direction, and the next iterate is obtained with a retraction.

## Alphabetical list of algorithms

| Solver   | Function        | State   |
|:---------|:----------------|:---------|
| [Adaptive Regularization with Cubics](adaptive_regularization_with_cubics.md) | [`adaptive_regularization_with_cubics`](@ref) | [`AdaptiveRegularizationState`](@ref) |
| [Alternating Gradient Descent](@ref solver-alternating-gradient-descent) | [`alternating_gradient_descent`](@ref) | [`AlternatingGradientDescentState`](@ref) |
| [Augmented Lagrangian Method](augmented_Lagrangian_method.md) | [`augmented_Lagrangian_method`](@ref) | [`AugmentedLagrangianMethodState`](@ref) |
| [Chambolle-Pock](ChambollePock.md) | [`ChambollePock`](@ref) | [`ChambollePockState`](@ref) |
| [CMA-ES](cma_es.md) | [`cma_es`](@ref) | [`CMAESState`](@ref) |
| [Conjugate Gradient Descent](conjugate_gradient_descent.md) | [`conjugate_gradient_descent`](@ref) | [`ConjugateGradientDescentState`](@ref) |
| [Conjugate Residual](conjugate_residual.md) | [`conjugate_residual`](@ref) | [`ConjugateResidualState`](@ref) |
| [Convex Bundle Method](convex_bundle_method.md) | [`convex_bundle_method`](@ref) |  [`ConvexBundleMethodState`](@ref) |
| [Cyclic Proximal Point](cyclic_proximal_point.md) | [`cyclic_proximal_point`](@ref) |  [`CyclicProximalPointState`](@ref) |
| [Difference of Convex Algorithm](@ref solver-difference-of-convex) | [`difference_of_convex_algorithm`](@ref) | [`DifferenceOfConvexState`](@ref) |
| [Difference of Convex Proximal Point](@ref solver-difference-of-convex-proximal-point) | [`difference_of_convex_proximal_point`](@ref) | [`DifferenceOfConvexProximalState`](@ref) |
| [Douglas-Rachford](DouglasRachford.md) | [`DouglasRachford`](@ref) | [`DouglasRachfordState`](@ref) |
| [Exact Penalty Method](exact_penalty_method.md) | [`exact_penalty_method`](@ref) |  [`ExactPenaltyMethodState`](@ref) |
| [Frank-Wolfe algorithm](FrankWolfe.md) | [`Frank_Wolfe_method`](@ref) | [`FrankWolfeState`](@ref) |
| [Gradient Descent](gradient_descent.md) | [`gradient_descent`](@ref) |  [`GradientDescentState`](@ref) |
| [Gradient Sampling](gradient_sampling.md) | [`gradient_sampling`](@ref) |  [`GradientSamplingState`](@ref) |
| [Interior Point Newton](interior_point_Newton.md) | [`interior_point_Newton`](@ref) | [`InteriorPointNewtonState`](@ref) |
| [Levenberg-Marquardt](LevenbergMarquardt.md) | [`LevenbergMarquardt`](@ref) | [`LevenbergMarquardtState`](@ref) |
| [Mesh Adaptive Direct Search](mesh_adaptive_direct_search.md) | [`mesh_adaptive_direct_search`](@ref) | [`MeshAdaptiveDirectSearchState`](@ref) |
| [Nelder-Mead](NelderMead.md) | [`NelderMead`](@ref) | [`NelderMeadState`](@ref) |
| [Particle Swarm](particle_swarm.md) | [`particle_swarm`](@ref) | [`ParticleSwarmState`](@ref) |
| [Primal-dual Riemannian semismooth Newton Algorithm](@ref solver-pdrssn) | [`primal_dual_semismooth_Newton`](@ref) | [`PrimalDualSemismoothNewtonState`](@ref) |
| [Projected Gradient Method](projected_gradient_method.md) | [`projected_gradient_method`](@ref) | [`ProjectedGradientMethodState`](@ref) |
| [Proximal Bundle Method](proximal_bundle_method.md) | [`proximal_bundle_method`](@ref) | [`ProximalBundleMethodState`](@ref) |
| [Proximal Gradient Method](proximal_gradient_method.md) | [`proximal_gradient_method`](@ref) | [`ProximalGradientMethodState`](@ref) |
| [Proximal Point](proximal_point.md) | [`proximal_point`](@ref) |  [`ProximalPointState`](@ref) |
| [Quasi-Newton Method](quasi_Newton.md) | [`quasi_Newton`](@ref) | [`QuasiNewtonState`](@ref) |
| [Riemannian Trust-Regions](trust_regions.md) | [`trust_regions`](@ref) | [`TrustRegionsState`](@ref) |
| [Steihaug-Toint Truncated Conjugate-Gradient Method](@ref tCG) | [`truncated_conjugate_gradient_descent`](@ref) | [`TruncatedConjugateGradientState`](@ref) |
| [Stochastic Gradient Descent](stochastic_gradient_descent.md) | [`stochastic_gradient_descent`](@ref) | [`StochasticGradientDescentState`](@ref) |
| [Subgradient Method](subgradient.md) | [`subgradient_method`](@ref) | [`SubGradientMethodState`](@ref) |
| [Vector Bundle Newton](vectorbundle_newton.md) | [`vectorbundle_newton`](@ref) | [`VectorBundleNewtonState`](@ref) |
