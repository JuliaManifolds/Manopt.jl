#
#
# This file collects a few strings to be reused in documentation to avoid retyping everything

# LateX symbols
_l_Manifold(M="M") = "\\mathcal $M"
_l_M = "$(_l_Manifold())"
_l_TpM(p="p") = "T_{$p}$_l_M"

_l_cO = raw"\mathcal O"
_L_argmin = raw"\operatorname{arg\,min}"
_l_grad = raw"\operatorname{grad}"
_l_grad_long = raw"\operatorname{grad} f: \mathcal M → T\mathcal M"
_l_Hess = raw"\operatorname{Hess}"
_l_Hess_long = "$_l_Hess f(p)[⋅]: $(_l_TpM()) → $(_l_TpM())"
_l_refl = raw"\operatorname{refl}_p(x) = \operatorname{retr}_p(-\operatorname{retr}^{-1}_p x)"

_l_retr = raw"\operatorname{retr}"
_l_retr_long = raw"\operatorname{retr}: T\mathcal M \to \mathcal M"
_l_cal(letter::String) = raw"\mathcal "*"$letter"
_l_vt = raw"\mathcal T_{\cdot\gets\cdot}"
_l_C_subset_M = raw"\mathcal C \subset \mathcal M"
_l_M = raw"\mathcal M"
# Math terms
_math_VT = raw"a vector transport ``T``"
_math_inv_retr = "an inverse retraction ``$_l_retr^{-1}``"
_math_retr = " a retraction $_l_retr"
_math_reflect = raw"""
```math
  \operatorname{refl}_p(x) = \operatorname{retr}_p(-\operatorname{retr}^{-1}_p x),
```
where ``\operatorname{retr}`` and ``\operatorname{retr}^{-1}`` denote a retraction and an inverse
retraction, respectively.
"""
function _math_sequence(name, index, i_start=1, i_end="n")
    return "\\{$(name)_{$index}\\}_{i=$(i_start)}^{$i_end}"
end

_problem_default = raw"""
```math
\operatorname*{arg\,min}_{p ∈ \mathcal M} f(p)
```
"""

# Arguments of functions
_arg_alt_mgo = raw"""
Alternatively to `f` and `grad_f` you can provide
the [`AbstractManifoldGradientObjective`](@ref) `gradient_objective` directly.
"""

# Arguments
_arg_f = raw"* `f`, a cost function ``f: \mathcal M→ℝ`` implemented as `(M, p) -> v`"
_arg_grad_f = raw"""
* `grad_f`, the gradient ``\operatorname{grad}f: \mathcal M → T\mathcal M`` of f
  as a function `(M, p) -> X` or a function `(M, X, p) -> X` computing `X` in-place
"""
_arg_Hess_f = """
* `Hess_f`, the Hessian ``$_l_Hess_long`` of f
  as a function `(M, p, X) -> Y` or a function `(M, Y, p, X) -> Y` computing `Y` in-place
"""
_arg_p = raw"* `p`, an initial value `p` ``= p^{(0)} ∈ \mathcal M``"
_arg_M = "* `M`, a manifold ``$_l_M``"
_arg_inline_M = "the manifold `M`"
_arg_X = "* `X` a tangent vector"
_arg_sub_problem = "* `sub_problem` a [`AbstractManoptProblem`](@ref) to specify a problem for a solver or a closed form solution function."
_arg_sub_state = "* `sub_state` a [`AbstractManoptSolverState`](@ref) for the `sub_problem` or a [`AbstractEvaluationType`](@ref) if a closed form solution is provided."

_doc_remark_tutorial_debug = "If you activate [`tutorial_mode`]"
_doc_sec_output = """
# Output

The obtained approximate minimizer ``p^*``.
To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details, especially the `return_state=` keyword.
"""
# Fields
_fild_at_iteration = "`at_iteration`: an integer indicating at which the stopping criterion last indicted to stop, which might also be before the solver started (`0`).\
  any negative value indicates that this was not yet the case; "
_field_iterate = "`p` : the current iterate ``p=p^{(k)} ∈ $_l_M``"
_field_gradient = "`X` : the current gradient ``$(_l_grad)f(p^{(k)}) ∈ T_p$_l_M``"
_field_inv_retr = "`inverse_retraction_method::`[`AbstractInverseRetractionMethod`](@extref `ManifoldsBase.AbstractInverseRetractionMethod`) : an inverse retraction ``$_l_retr_long^{-1}``"
_field_p = raw"`p`, an initial value `p` ``= p^{(0)} ∈ \mathcal M``"
_field_retr = "`retraction_method::`[`AbstractRetractionMethod`](@extref `ManifoldsBase.AbstractRetractionMethod`) : a retraction ``$_l_retr_long``"
_field_sub_problem = "`sub_problem::Union{`[`AbstractManoptProblem`](@ref)`, F}`: a manopt problem or a function for a closed form solution of the sub problem"
_field_sub_state = "`sub_state::Union{`[`AbstractManoptSolverState`](@ref)`,`[`AbstractEvaluationType`](@ref)`}`: for a sub problem state which solver to use, for the closed form solution function,\
indicate, whether the closed form solution function works with [`AllocatingEvaluation`](@ref)) `(M, p, X) -> q` or with an [`InplaceEvaluation`](@ref)) `(M, q, p, X) -> q`"
_field_stop = "`stop::`[`StoppingCriterion`](@ref) : a functor indicating when to stop and whether the algorithm has stopped"
_field_step = "`stepsize::`[`Stepsize`](@ref) : a stepsize."
_field_vector_transp = "`vector_transport_method::`[`AbstractVectorTransportMethod`](@extref `ManifoldsBase.AbstractVectorTransportMethod`) : a vector transport ``$_l_vt``"
_field_X = "`X` a tangent vector"

# Keywords
_kw_evaluation_default = "`evaluation=`[`AllocatingEvaluation`](@ref)`()`"
_kw_evaluation = "specify whether the functions that return an array, for example a point\
or a tangent vector, work by allocating its result ([`AllocatingEvaluation`](@ref)) or\
whether they modify their input argument to return the result therein ([`InplaceEvaluation`](@ref)).\
 Since usually the first argument is the manifold, the modified argument is the second."
_kw_evaluation_example = "For example `grad_f(M,p)` allocates, but `grad_f!(M, X, p)` computes the result in-place of `X`."

_kw_inverse_retraction_method_default = "`inverse_retraction_method=`[`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`)`(M, typeof(p))`"
_kw_inverse_retraction_method = "an inverse retraction ``$(_l_retr)^{-1}`` to use, see [the section on retractions and their inverses](@extref ManifoldsBase :doc:`retractions`)."

_kw_others = raw"""
All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or
[`decorate_objective!`](@ref) for objective, respectively.
"""

_kw_retraction_method_default = raw"`retraction_method=`[`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`)`(M, typeof(p))`"
_kw_retraction_method = raw"a retraction ``\operatorname{retr}`` to use, see [the section on retractions](@extref ManifoldsBase :doc:`retractions`)."

_kw_stepsize = raw"a functor inheriting from [`Stepsize`](@ref) to determine a step size"

_kw_stopping_criterion = raw"a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop."
_kw_stop_note = "is used to set the field `stop`."

_kw_sub_kwargs_default = "`sub_kwargs=(;)`"
_kw_sub_kwargs = "a named tuple of keyword arguments that are passed to [`decorate_objective!`](@ref) of the sub solvers objective,\
the [`decorate_state!`](@ref) of the subsovlers state, and the sub state constructor itself.
"

_kw_sub_objective = "a shortcut to modify the objective of the subproblem used within in the `sub_problem=` keyword"
function _kw_sub_objective_default_text(type::String)
    return "By default, this is initialized as a [`$type`](@ref),\
which can further be decorated by using the `sub_kwargs=` keyword"
end

_kw_vector_transport_method_default = raw"`vector_transport_method=`[`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`)`(M, typeof(p))`"
_kw_vector_transport_method = raw"a vector transport ``\mathcal T`` to use, see [the section on vector transports](@extref ManifoldsBase :doc:`vector_transports`)."

_kw_X_default = raw"`X=`[`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)`(M,p)`"
_kw_X = raw"specify a memory internally to store a tangent vector"

function _kw_used_in(s::String)
    return "This is used to define the `$s=` keyword and has hence no effect, if you set `$s` directly."
end
