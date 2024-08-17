#
#
# This file collects a few strings to be reused in documentation to avoid retyping everything

# In general every dictionary here can be either :Symbol-> String or :Symbol -> Dictionary enrties

_MANOPT_DOC_TYPE = Dict{Symbol,Union{String,Dict,Function}}()

_manopt_docs = _MANOPT_DOC_TYPE()
# ---
# LaTeX
_manopt_docs[:LaTeX] = _MANOPT_DOC_TYPE()
_l = _manopt_docs[:LaTeX]
_l[:Cal] = (letter) -> raw"\mathcal " * "$letter"
_l[:frac] = (a,b) -> raw"\frac" * "{$a}{$b}"

# ---
# Mathematics and semantic symbols
# :symbol the symbol,
# :descr the description
_manopt_docs[:Math] = _MANOPT_DOC_TYPE()

# ---
# Links
# Collect short forms for links, especially Interdocs ones.
_manopt_docs[:Link] = _MANOPT_DOC_TYPE()
_link = _manopt_docs[:Link]

# ---
# Variables
# in fields, keyword arguments, parameters
# for each variable as a symbol, we store
# The variable name should be the symbol
# :default – in positional or keyword arguments
# :description – a text description of the variable
# :type a type
#
_manopt_docs[:Var] = _MANOPT_DOC_TYPE()
_var[:p] = Dict(
  :description => "a point on a manifold ``$(_l[:Cal]("M"))``",
  :type => "P",
  default => "rand(M)", # TODO Fix when the Links dictionary exists
)

# ---
# Problems

# ---
# Notes

# ---
# Old strings

# LateX symbols
_l_ds = raw"\displaystyle"
_l_argmin = raw"\operatorname{arg\,min}"
_l_grad = raw"\operatorname{grad}"
_l_Hess = raw"\operatorname{Hess}"
_l_log = raw"\log"
_l_prox = raw"\operatorname{prox}"
_l_refl = raw"\operatorname{refl}_p(x) = \operatorname{retr}_p(-\operatorname{retr}^{-1}_p x)"
_l_subgrad = raw"∂"
_l_min = raw"\min"
_l_max = raw"\min"
_l_norm(v, i="") = raw"\lVert" * "$v" * raw"\rVert" * "_{$i}"
# Semantics
_l_Manifold(M="M") = _l[:Cal](M)
_l_M = "$(_l_Manifold())"
_l_TpM(p="p") = "T_{$p}$_l_M"
_l_DΛ = "DΛ: T_{m}$(_l_M) → T_{Λ(m)}$(_l_Manifold("N"))"
_l_grad_long = raw"\operatorname{grad} f: \mathcal M → T\mathcal M"
_l_Hess_long = "$_l_Hess f(p)[⋅]: $(_l_TpM()) → $(_l_TpM())"
_l_retr = raw"\operatorname{retr}"
_l_retr_long = raw"\operatorname{retr}: T\mathcal M \to \mathcal M"
_l_vt = raw"\mathcal T_{\cdot\gets\cdot}"
_l_C_subset_M = "$(_l[:Cal]("C")) ⊂ $(_l[:Cal]("M"))"
_l_txt(s) = "\\text{$s}"

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

#
#
# Links

function _link_zero_vector(M="M", p="p")
    arg = length(M) > 0 ? "`($M, $p)`" : ""
    return "[`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)$arg"
end
function _link_manifold_dimension(M="M")
    arg = length(M) > 0 ? "`($M)`" : ""
    return "[`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`)$arg"
end
function _link_rand(M="M")
    arg = length(M) > 0 ? "`($M)`" : ""
    return "[`rand`](@extref Base.rand-Tuple{AbstractManifold})$arg"
end

#
#
# Problems

_problem_default = raw"""
```math
\operatorname*{arg\,min}_{p ∈ \mathcal M} f(p)
```
"""

_problem_constrained = raw"""```math
\begin{aligned}
\min_{p ∈\mathcal{M}} &f(p)\\
\text{subject to } &g_i(p)\leq 0 \quad \text{ for } i= 1, …, m,\\
\quad &h_j(p)=0 \quad \text{ for } j=1,…,n,
\end{aligned}
```
"""

# Arguments of functions
_arg_alt_mgo = raw"""
Alternatively to `f` and `grad_f` you can provide
the [`AbstractManifoldGradientObjective`](@ref) `gradient_objective` directly.
"""

# Arguments
_arg_f = raw"* `f`: a cost function ``f: \mathcal M→ℝ`` implemented as `(M, p) -> v`"
_arg_grad_f = raw"""
* `grad_f`: the gradient ``\operatorname{grad}f: \mathcal M → T\mathcal M`` of f
  as a function `(M, p) -> X` or a function `(M, X, p) -> X` computing `X` in-place
"""
_arg_Hess_f = """
* `Hess_f`: the Hessian ``$_l_Hess_long`` of f
  as a function `(M, p, X) -> Y` or a function `(M, Y, p, X) -> Y` computing `Y` in-place
"""
_arg_p = raw"* `p` an initial value `p` ``= p^{(0)} ∈ \mathcal M``"
_arg_M = "* `M` a manifold ``$_l_M``"
_arg_inline_M = "the manifold `M`"
_arg_X = "* `X` a tangent vector"
_arg_sub_problem = "* `sub_problem` a [`AbstractManoptProblem`](@ref) to specify a problem for a solver or a closed form solution function."
_arg_sub_state = "* `sub_state` a [`AbstractManoptSolverState`](@ref) for the `sub_problem`."
_arg_subgrad_f = raw"""
* `∂f`: the subgradient ``∂f: \mathcal M → T\mathcal M`` of f
  as a function `(M, p) -> X` or a function `(M, X, p) -> X` computing `X` in-place.
  This function should always only return one element from the subgradient.
"""

_doc_remark_tutorial_debug = "If you activate tutorial mode (cf. [`is_tutorial_mode`](@ref)), this solver provides additional debug warnings."
_doc_sec_output = """
# Output

The obtained approximate minimizer ``p^*``.
To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details, especially the `return_state=` keyword.
"""

_sc_any = "[` | `](@ref StopWhenAny)"
_sc_all = "[` & `](@ref StopWhenAll)"

# Fields
_field_at_iteration = "`at_iteration`: an integer indicating at which the stopping criterion last indicted to stop, which might also be before the solver started (`0`). Any negative value indicates that this was not yet the case; "
_field_iterate = "`p`: the current iterate ``p=p^{(k)} ∈ $(_l_M)``"
_field_gradient = "`X`: the current gradient ``$(_l_grad)f(p^{(k)}) ∈ T_p$(_l_M)``"
_field_subgradient = "`X` : the current subgradient ``$(_l_subgrad)f(p^{(k)}) ∈ T_p$_l_M``"
_field_inv_retr = "`inverse_retraction_method::`[`AbstractInverseRetractionMethod`](@extref `ManifoldsBase.AbstractInverseRetractionMethod`) : an inverse retraction ``$(_l_retr)^{-1}``"
_field_p = raw"`p`, an initial value `p` ``= p^{(0)} ∈ \mathcal M``"
_field_retr = "`retraction_method::`[`AbstractRetractionMethod`](@extref `ManifoldsBase.AbstractRetractionMethod`) : a retraction ``$(_l_retr_long)``"
_field_sub_problem = "`sub_problem::Union{`[`AbstractManoptProblem`](@ref)`, F}`: a manopt problem or a function for a closed form solution of the sub problem"
_field_sub_state = "`sub_state::Union{`[`AbstractManoptSolverState`](@ref)`,`[`AbstractEvaluationType`](@ref)`}`: for a sub problem state which solver to use, for the closed form solution function, indicate, whether the closed form solution function works with [`AllocatingEvaluation`](@ref)) `(M, p, X) -> q` or with an [`InplaceEvaluation`](@ref)) `(M, q, p, X) -> q`"
_field_stop = "`stop::`[`StoppingCriterion`](@ref) : a functor indicating when to stop and whether the algorithm has stopped"
_field_step = "`stepsize::`[`Stepsize`](@ref) : a stepsize."
_field_vector_transp = "`vector_transport_method::`[`AbstractVectorTransportMethod`](@extref `ManifoldsBase.AbstractVectorTransportMethod`) : a vector transport ``$_l_vt``"
_field_X = "`X`: a tangent vector"

#
#
# Keywords
_kw_evaluation_default = "`evaluation=`[`AllocatingEvaluation`](@ref)`()`"
_kw_evaluation = "specify whether the functions that return an array, for example a point or a tangent vector, work by allocating its result ([`AllocatingEvaluation`](@ref)) or whether they modify their input argument to return the result therein ([`InplaceEvaluation`](@ref)). Since usually the first argument is the manifold, the modified argument is the second."
_kw_evaluation_example = "For example `grad_f(M,p)` allocates, but `grad_f!(M, X, p)` computes the result in-place of `X`."

_kw_inverse_retraction_method_default = "`inverse_retraction_method=`[`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`)`(M, typeof(p))`"
_kw_inverse_retraction_method = "an inverse retraction ``$(_l_retr)^{-1}`` to use, see [the section on retractions and their inverses](@extref ManifoldsBase :doc:`retractions`)."

_kw_others = raw"""
All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or
[`decorate_objective!`](@ref) for objective, respectively.
"""

_kw_p_default = "`p=`$(_link_rand())"
_kw_p = raw"specify an initial value for the point `p`."

_kw_retraction_method_default = raw"`retraction_method=`[`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`)`(M, typeof(p))`"
_kw_retraction_method = "a retraction ``$(_l_retr)`` to use, see [the section on retractions](@extref ManifoldsBase :doc:`retractions`)."

_kw_stepsize = raw"a functor inheriting from [`Stepsize`](@ref) to determine a step size"

_kw_stopping_criterion = raw"a functor inheriting from [`StoppingCriterion`](@ref) indicating when to stop."
_kw_stop_note = "is used to set the field `stop`."

_kw_sub_kwargs_default = "`sub_kwargs=(;)`"
_kw_sub_kwargs = "a named tuple of keyword arguments that are passed to [`decorate_objective!`](@ref) of the sub solvers objective, the [`decorate_state!`](@ref) of the subsovlers state, and the sub state constructor itself."

_kw_sub_objective = "a shortcut to modify the objective of the subproblem used within in the `sub_problem=` keyword"
function _kw_sub_objective_default_text(type::String)
    return "By default, this is initialized as a [`$type`](@ref), which can further be decorated by using the `sub_kwargs=` keyword"
end

_kw_vector_transport_method_default = "`vector_transport_method=`[`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`)`(M, typeof(p))`"
_kw_vector_transport_method = "a vector transport ``$_l_vt`` to use, see [the section on vector transports](@extref ManifoldsBase :doc:`vector_transports`)."

_kw_X_default = "`X=`$(_link_zero_vector())"
_kw_X = raw"specify a memory internally to store a tangent vector"
_kw_X_init = raw"specify an initial value for the tangent vector"

function _kw_used_in(s::String)
    return "This is used to define the `$s=` keyword and has hence no effect, if you set `$s` directly."
end
