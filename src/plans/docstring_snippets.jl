#
#
# This file collects a few docstrings to be reused in documentation to avoid retyping everything

_arg_alt_mgo = raw"""
Alternatively to `f` and `grad_f` you can provide
the [`AbstractManifoldGradientObjective`](@ref) `gradient_objective` directly.
"""

_arg_f = raw"* `f`, a cost function ``f: \mathcal M→ℝ``"
_arg_grad_f = raw"""
* `grad_f`, the gradient ``\operatorname{grad}f: \mathcal M → T\mathcal M`` of f
  as a function `(M, p) -> X` or a function `(M, X, p) -> X` computing `X` in-place
"""
_arg_p = raw"* `p`, an initial value `p` ``= p_0 ∈ \mathcal M``"

_arg_M = raw"* `M`, a manifold ``\mathcal M``"

_kw_evaluation_default = raw"`evaluation = ` [`AllocatingEvaluation`](@ref)`()`"
_kw_evaluation = raw"""
  specify whether the functions that return a value on a manifold or a tangent space
  work by allocating its result([`AllocatingEvaluation`](@ref) or whether it accepts the result
  as its (usual second) input argument (after the manifold), that is we have an [`InplaceEvaluation`](@ref).
  For example `grad_f(M,p)` allocates, but `grad_f!(M, X, p)` computes the result in-place of `X`.
"""

_kw_others = raw"""
All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or
[`decorate_objective!`](@ref) for objective, respectively.
"""

_kw_retraction_method_default = raw"`retraction_method = ` [`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`)`(M, typeof(p))`"
_kw_retraction_method = raw"""
  a retraction to use internally, see [the section on retractions](@extref ManifoldsBase :doc:`retractions`).
"""

_kw_stepsize = raw"""
  a functor inheriting from [`Stepsize`](@ref) to determine a step size
"""

_kw_stopping_criterion = raw"""
  a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop.
"""
