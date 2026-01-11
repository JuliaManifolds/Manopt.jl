#
# Manopt Glossary
# ===
#
# This file collects
# - LaTeX snippets
# - math formulae
# - Variable names
# - links
# - notes
#
# to keep naming, notation, and formatting

# In general every dictionary here can be either :Symbol-> String or :Symbol -> Dictionary entries

# generate Glossary for Manopt.jl
Glossaries.@Glossary()
# Set a glossary for Manopt.jl
_glossary = Glossaries.Glossary()
# Set this as the current glossary for this module
current_glossary!(_glossary)

# LaTeX snippets – we use a separate sub glossary to still always use the
# math formatter
#
# These are meant to be syntactic helpers for pure LaTeX snippets,
# since otherwise they would be hard to type in (non-raw) strings.
_glossary[:LaTeX] = Glossaries.Glossary()
_glossary_tex_terms = _glossary[:LaTeX]
__tex_formatter = Glossaries.Math()
_tex(args...; kwargs...) = __tex_formatter(_glossary_tex_terms, args...; kwargs...)

function _tex_aligned(lines...)
    return raw"\begin{aligned}\n" *
        "$(join(["   $(line)" for line in lines], raw"\\\\ "))" *
        raw"\n\end{aligned}\n"
end
Glossaries.define!(_glossary_tex_terms, :aligned, :math, _tex_aligned)
_tex_abs(v) = raw"\lvert " * "$v" * raw" \rvert"
Glossaries.define!(_glossary_tex_terms, :abs, :math, _tex_abs)
Glossaries.define!(_glossary_tex_terms, :argmin, :math, raw"\operatorname*{arg\,min}")
Glossaries.define!(_glossary_tex_terms, :ast, :math, raw"\ast")
_tex_bar(letter) = raw"\bar" * "$(letter)"
Glossaries.define!(_glossary_tex_terms, :bar, :math, _tex_bar)
Glossaries.define!(_glossary_tex_terms, :big, :math, raw"\big")
Glossaries.define!(_glossary_tex_terms, :bigl, :math, raw"\bigl")
Glossaries.define!(_glossary_tex_terms, :bigr, :math, raw"\bigr")
Glossaries.define!(_glossary_tex_terms, :biggl, :math, raw"\biggl")
Glossaries.define!(_glossary_tex_terms, :biggr, :math, raw"\biggr")
Glossaries.define!(_glossary_tex_terms, :Big, :math, raw"\Big")
Glossaries.define!(_glossary_tex_terms, :Bigl, :math, raw"\Bigl")
Glossaries.define!(_glossary_tex_terms, :Bigr, :math, raw"\Bigr")
_tex_Cal(letter) = raw"\mathcal{" * "$letter" * "}"
Glossaries.define!(_glossary_tex_terms, :Cal, :math, _tex_Cal)
function _tex_cases(cases...)
    return raw"\begin{cases}" *
        "$(join(["   $(ci)" for ci in c], raw"\\\\ "))" *
        raw"\end{cases}"
end
Glossaries.define!(_glossary_tex_terms, :cases, :math, _tex_cases)
Glossaries.define!(_glossary_tex_terms, :cdots, :math, raw"\cdots")
Glossaries.define!(_glossary_tex_terms, :cot, :math, raw"\cot")
Glossaries.define!(_glossary_tex_terms, :ddots, :math, raw"\ddots")
_tex_deriv(t = "t") = raw"\frac{\mathrm{d}}{\mathrm{d}" * "$(t)" * "}"
Glossaries.define!(_glossary_tex_terms, :deriv, :math, _tex_deriv)
_tex_diff(t = "t") = raw"\mathrm{D}_{" * "$(t)" * "}"
Glossaries.define!(_glossary_tex_terms, :diff, :math, _tex_diff)
Glossaries.define!(_glossary_tex_terms, :displaystyle, :math, raw"\displaystyle")
Glossaries.define!(_glossary_tex_terms, :eR, :math, raw"\bar{\mathbb R}")
_tex_frac(a, b) = raw"\frac" * "{$a}{$b}"
Glossaries.define!(_glossary_tex_terms, :frac, :math, _tex_frac)
Glossaries.define!(_glossary_tex_terms, :grad, :math, raw"\operatorname{grad}")
_tex_hat(letter) = raw"\hat{" * "$letter" * "}"
Glossaries.define!(_glossary_tex_terms, :hat, :math, _tex_hat)
Glossaries.define!(_glossary_tex_terms, :Hess, :math, raw"\operatorname{Hess}")
Glossaries.define!(_glossary_tex_terms, :Id, :math, raw"\mathrm{Id}")
Glossaries.define!(_glossary_tex_terms, :invretr, :math, raw"\operatorname{retr}^{-1}")
_tex_inner(a, b; index = "") = "⟨$a,$b⟩_{$index}"
Glossaries.define!(_glossary_tex_terms, :inner, :math, _tex_inner)
Glossaries.define!(_glossary_tex_terms, :log, :math, raw"\log")
Glossaries.define!(_glossary_tex_terms, :max, :math, raw"\max")
Glossaries.define!(_glossary_tex_terms, :min, :math, raw"\min")
_tex_norm(v; index = "") = raw"\lVert " * "$v" * raw" \rVert" * "_{$index}"
Glossaries.define!(_glossary_tex_terms, :norm, :math, _tex_norm)
_tex_pmatrix(lines...) = raw"\begin{pmatrix} " * join(lines, raw"\\ ") * raw"\end{pmatrix}"
Glossaries.define!(_glossary_tex_terms, :pmatrix, :math, _tex_pmatrix)
_tex_operatorname(name) = raw"\operatorname{" * "$name" * "}"
Glossaries.define!(_glossary_tex_terms, :operatorname, :math, _tex_operatorname)
Glossaries.define!(_glossary_tex_terms, :proj, :math, raw"\operatorname{proj}")
Glossaries.define!(_glossary_tex_terms, :prox, :math, raw"\operatorname{prox}")
Glossaries.define!(_glossary_tex_terms, :quad, :math, raw"\quad")
Glossaries.define!(_glossary_tex_terms, :qquad, :math, raw"\qquad")
Glossaries.define!(_glossary_tex_terms, :reflect, :math, raw"\operatorname{refl}")
Glossaries.define!(_glossary_tex_terms, :retr, :math, raw"\operatorname{retr}")
_tex_rm(letter) = raw"\mathrm{" * "$letter" * "}"
Glossaries.define!(_glossary_tex_terms, :rm, :math, _tex_rm)
_tex_sqrt(s) = raw"\sqrt{" * "$s" * "}"
Glossaries.define!(_glossary_tex_terms, :sqrt, :math, _tex_sqrt)
Glossaries.define!(_glossary_tex_terms, :subgrad, :math, raw"∂")
_tex_set(s) = raw"\set{" * "$s" * "}"
Glossaries.define!(_glossary_tex_terms, :set, :math, _tex_set)
_tex_sum(b = "", t = "") = raw"\sum" * (length(b) > 0 ? "_{$b}" : "") * (length(t) > 0 ? "^{$t}" : "")
Glossaries.define!(_glossary_tex_terms, :sum, :math, _tex_sum)
_tex_text(s) = raw"\text{" * "$s" * "}"
Glossaries.define!(_glossary_tex_terms, :text, :math, _tex_text)
Glossaries.define!(_glossary_tex_terms, :tilde, :math, raw"\tilde")
Glossaries.define!(_glossary_tex_terms, :transp, :math, raw"\mathrm{T}")
Glossaries.define!(_glossary_tex_terms, :vdots, :math, raw"\vdots")
Glossaries.define!(_glossary_tex_terms, :vert, :math, raw"\vert")
_tex_widehat(letter) = raw"\widehat{" * "$letter" * "}"
Glossaries.define!(_glossary_tex_terms, :widehat, :math, _tex_widehat)
_tex_widetilde(letter) = raw"\widetilde{" * "$letter" * "}"
Glossaries.define!(_glossary_tex_terms, :widetilde, :math, _tex_widetilde)
#
# ---
# Mathematics and semantic symbols
#
# as opposed to the LaTeX snippets above, these are meant to represent semantic shortcuts
# for mathematical notation to allow consistent naming and formatting
_glossary[:Math] = Glossaries.Glossary()
_glossary_math_terms = _glossary[:Math]

_math_formatter = Glossaries.Math()
_math(args...; kwargs...) = _math_formatter(_glossary_math_terms, args...; kwargs...)

Glossaries.define!(_glossary_math_terms, :distance, :math, raw"\mathrm{d}")
_math_manifold(; M = "M") = _tex(:Cal, M)
Glossaries.define!(_glossary_math_terms, :Manifold, :math, _math_manifold)
Glossaries.define!(_glossary_math_terms, :Manifold, :description, "the Riemannian manifold")
_math_iterate(; p = "p", k = "k") = "$(p)^{($(k))}"
Glossaries.define!(_glossary_math_terms, :Iterate, :math, _math_iterate)
Glossaries.define!(_glossary_math_terms, :Iterate, :description, "the current iterate at iteration ``k``")
_math_sequence(var, ind, from, to) = raw"\{" * "$(var)_$(ind)" * raw"\}" * "_{$(ind)=$from}^{$to}"
Glossaries.define!(_glossary_math_terms, :Sequence, :math, _math_sequence)
_math_TangentBundle(; M = "M") = "T$(_tex(:Cal, M))"
Glossaries.define!(_glossary_math_terms, :TangentBundle, :math, _math_TangentBundle)
_math_TangentBundle_description(; M = "M") = "the tangent bundle of the manifold $(_math(:Manifold; M = M))"
Glossaries.define!(_glossary_math_terms, :TangentBundle, :description, _math_TangentBundle_description)
_math_TangentSpace(; M = "M", p = "p") = "T_{$p}$(_tex(:Cal, M))"
Glossaries.define!(_glossary_math_terms, :TangentSpace, :math, _math_TangentSpace)
_math_TangentSpace_description(; M = "M", p = "p") = "the tangent space at the point ``$p`` on the manifold ``$(_math(:Manifold; M = M))``"
Glossaries.define!(_glossary_math_terms, :TangentSpace, :description, _math_TangentSpace_description)

_math_vector_transport(p = "⋅", q = "⋅") = raw"\mathcal T_{" * "$q←$p" * "}"
Glossaries.define!(_glossary_math_terms, :VectorTransport, :math, _math_vector_transport)
function _math_vector_transport_description(; M = "M", p = "⋅", q = "⋅")
    points_given == (length(p) > 0 && p != "⋅" && length(q) > 0 && q != "⋅")
    return manifold_given = length(M) > 0 &&
        return "a vector transport $(manifold_given ? "on the manifold $(_math(:Manifold; M = M)) " : "")" *
        "$(points_given ? "from ``$p`` to ``$q``" : "between two points")"
end
Glossaries.define!(_glossary_math_terms, :VectorTransport, :description, _math_vector_transport_description)
Glossaries.define!(_glossary_math_terms, :VectorTransport, :name, "the vector transport")

#
# ---
# Links Glossary
# Collect short forms for links, especially Interdocs ones.
_glossary[:Link] = Glossaries.Glossary()
_glossary_links = _glossary[:Link]

__link_formatter = Glossaries.Plain(:link)
_link(args...; kwargs...) = __link_formatter(_glossary_links, args...; kwargs...)

Glossaries.define!(
    _glossary_links, :AbstractManifold, :link,
    "[`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)",
)
Glossaries.define!(
    _glossary_links, :AbstractPowerManifold, :link,
    "[`AbstractPowerManifold`](@extref `ManifoldsBase.AbstractPowerManifold`)",
)
Glossaries.define!(
    _glossary_links, :injectivity_radius, :link,
    "[`injectivity_radius`](@extref `ManifoldsBase.injectivity_radius-Tuple{AbstractManifold}`)",
)
function _link_manifold_dimension(; M = "M")
    return "[`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`)$(length(M) > 0 ? "`($M)`" : "")"
end
Glossaries.define!(_glossary_links, :manifold_dimension, :link, _link_manifold_dimension)

Glossaries.define!(_glossary_links, :Manopt, :link, "[`Manopt.jl`](https://manoptjl.org)")
Glossaries.define!(_glossary_links, :Manifolds, :link, "[`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/)")
Glossaries.define!(
    _glossary_links, :ManifoldsBase, :link,
    "[`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/)",
)
function _link_rand(; M = "M")
    return "[`rand`](@extref Base.rand-Tuple{AbstractManifold})$(length(M) > 0 ? "`($M)`" : "")"
end
Glossaries.define!(_glossary_links, :rand, :link, _link_rand)
function _link_zero_vector(; M = "M", p = "p")
    return "[`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)$(length(M) > 0 ? "`($M, $p)`" : "")"
end
Glossaries.define!(_glossary_links, :zero_vector, :link, _link_zero_vector)

#
#
# Notes / Remarks
_glossary[:Note] = Glossaries.Glossary()
_glossary_notes = _glossary[:Note]

__note_formatter = Glossaries.Plain(:note)
_note(args...; kwargs...) = __note_formatter(_glossary_notes, args...; kwargs...)
Glossaries.define!(
    _glossary_notes, :ManifoldDefaultFactory, :note,
    (type::String) -> """
    !!! info
        This function generates a [`ManifoldDefaultsFactory`](@ref) for [`$(type)`](@ref).
        For default values, that depend on the manifold, this factory postpones the construction
        until the manifold from for example a corresponding [`AbstractManoptSolverState`](@ref) is available.
    """,
)
Glossaries.define!(
    _glossary_notes, :GradientObjective, :note,
    (; objective = "gradient_objective", f = "f", grad_f = "grad_f") -> """
    Alternatively to `$f` and `$grad_f` you can provide
    the corresponding [`AbstractManifoldFirstOrderObjective`](@ref) `$objective` directly.
    """,
)
Glossaries.define!(
    _glossary_notes, :OtherKeywords, :note,
    "All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or [`decorate_objective!`](@ref) for objective decorators, respectively.",
)
Glossaries.define!(
    _glossary_notes, :OutputSection, :note,
    (; p_min = "p^*") -> """
    # Output

    The obtained approximate minimizer ``$(p_min)``.
    To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details, especially the `return_state=` keyword.
    """,
)
Glossaries.define!(
    _glossary_notes, :TutorialMode, :note,
    "If you activate tutorial mode (cf. [`is_tutorial_mode`](@ref)), this solver provides additional debug warnings.",
)
Glossaries.define!(
    _glossary_notes, :KeywordUsedIn, :note,
    function (kw::String)
        return "This is used to define the `$(kw)=` keyword and has hence no effect, if you set `$(kw)` directly."
    end,
)

#
#
# Problems
_glossary[:Problem] = Glossaries.Glossary()
_glossary_problems = _glossary[:Problem]
__problem_formatter = Glossaries.Plain(:problem)
_problem(args...; kwargs...) = __problem_formatter(_glossary_problems, args...; kwargs...)

Glossaries.define!(
    _glossary_problems, :Constrained, :problem,
    (; M = "M", p = "p") -> """
        ```math
    \\begin{aligned}
    $(_tex(:argmin))_{$p ∈ $(_math(:Manifold; M = M))} & f($p)\\\\
    $(_tex(:text, "subject to"))$(_tex(:quad))&g_i($p) ≤ 0 \\quad $(_tex(:text, " for ")) i= 1, …, m,\\\\
    \\quad & h_j($p)=0 \\quad $(_tex(:text, " for ")) j=1,…,n,
    \\end{aligned}
    ```
    """,
)
Glossaries.define!(
    _glossary_problems, :SetConstrained, :problem,
    (; M = "M", p = "p") -> """
        ```math
    \\begin{aligned}
    $(_tex(:argmin))_{$p ∈ $(_math(:Manifold; M = M))} & f($p)\\\\
    $(_tex(:text, "subject to"))$(_tex(:quad))& p ∈ $(_tex(:Cal, "C")) ⊂ $(_math(:Manifold; M = M))
    \\end{aligned}
    ```
    """,
)
Glossaries.define!(
    _glossary_problems, :Default, :problem,
    (; M = "M", p = "p") -> """
    ```math
    $(_tex(:argmin))_{$p ∈ $(_math(:Manifold; M = M))} f($p)
    ```
    """
)
Glossaries.define!(
    _glossary_problems, :NonLinearLeastSquares, :problem,
    (; M = "M", p = "p") -> """
    ```math
    $(_tex(:argmin))_{$p ∈ $(_math(:Manifold; M = M))} $(_tex(:frac, 1, 2)) $(_tex(:sum, "i=1", "m")) $(_tex(:abs, "f_i($p)"))^2
    ```

    where ``f: $(_math(:Manifold; M = M)) → ℝ^m`` is written with component functions ``f_i: $(_math(:Manifold; M = M)) → ℝ``, ``i=1,…,m``,
    and each component function is continuously differentiable.
    """,
)

#
#
# Stopping Criteria
_glossary[:StoppingCriterion] = Glossaries.Glossary()
_glossary_stopping_criteria = _glossary[:StoppingCriterion]
__stopping_criterion_formatter = Glossaries.Plain(:stopping_criterion)
_sc(args...; kwargs...) = __stopping_criterion_formatter(_glossary_stopping_criteria, args...; kwargs...)

Glossaries.define!(
    _glossary_stopping_criteria, :Any, :stopping_criterion,
    "[` | `](@ref StopWhenAny)",
)
Glossaries.define!(
    _glossary_stopping_criteria, :All, :stopping_criterion,
    "[` & `](@ref StopWhenAll)",
)

#
#
# ---
# Variables
# in fields, keyword arguments, parameters
# for each variable as a symbol, we store
# The variable name should be the symbol
# :default – in positional or keyword arguments
# :description – a text description of the variable
# :type a type
_glossary[:Variable] = Glossaries.Glossary()
_glossary_variables = _glossary[:Variable]
__arg_formatter = Glossaries.Argument()
_args(args...; kwargs...) = __arg_formatter(_glossary_variables, args...; kwargs...)
__kwargs_formatter = Glossaries.Keyword()
_kwargs(args...; kwargs...) = __kwargs_formatter(_glossary_variables, args...; kwargs...)
__field_formatter = Glossaries.Field()
_fields(args...; kwargs...) = __field_formatter(_glossary_variables, args...; kwargs...)

#
#
#
# variables / Names used in Arguments, Fields, and Keywords

Glossaries.define!(_glossary_variables, :at_iteration)
Glossaries.define!(
    _glossary_variables, :at_iteration, :description,
    "an integer indicating at which the stopping criterion last indicted to stop, which might also be before the solver started (`0`). Any negative value indicates that this was not yet the case;",
)
Glossaries.define!(_glossary_variables, :at_iteration, :type, "Int")

Glossaries.define!(_glossary_variables, :differential)
Glossaries.define!(
    _glossary_variables, :differential, :description,
    "specify a specific function to evaluate the differential. By default, ``Df(p)[X] = ⟨$(_tex(:grad))f(p),X⟩``. is used",
)
Glossaries.define!(_glossary_variables, :differential, :default, "nothing")

Glossaries.define!(_glossary_variables, :evaluation)
Glossaries.define!(
    _glossary_variables, :evaluation, :description,
    "specify whether the functions that return an array, for example a point or a tangent vector, work by allocating its result ([`AllocatingEvaluation`](@ref)) or whether they modify their input argument to return the result therein ([`InplaceEvaluation`](@ref)). Since usually the first argument is the manifold, the modified argument is the second.",
)
Glossaries.define!(_glossary_variables, :evaluation, :type, "`[`AbstractEvaluationType`](@ref)` ")
Glossaries.define!(_glossary_variables, :evaluation, :default, "`[`AllocatingEvaluation`](@ref)`()")
Glossaries.define!(
    _glossary_variables, :evaluation,
    :GradientExample,
    "For example `grad_f(M,p)` allocates, but `grad_f!(M, X, p)` computes the result in-place of `X`.",
)

Glossaries.define!(_glossary_variables, :f)
Glossaries.define!(
    _glossary_variables, :f, :description,
    function (; M = "M", p = "p")
        return "a cost function ``f: $(_tex(:Cal, M))→ ℝ`` implemented as `($M, $p) -> v`"
    end,
)

Glossaries.define!(_glossary_variables, :grad_f)
Glossaries.define!(
    _glossary_variables, :grad_f, :description,
    (; M = "M", p = "p", f = "f", kwargs...) ->
    "the (Riemannian) gradient ``$(_tex(:grad))$f: $(_math(:Manifold, M = M)) → $(_math(:TangentSpace; M = M, p = p))`` of $f as a function `(M, p) -> X` or a function `(M, X, p) -> X` computing `X` in-place",
)

Glossaries.define!(_glossary_variables, :Hess_f)
Glossaries.define!(
    _glossary_variables, :Hess_f, :description,
    (; M = "M", p = "p", f = "f") ->
    "the (Riemannian) Hessian ``$(_tex(:Hess))$f: $(_math(:TangentSpace, M = M, p = p)) → $(_math(:TangentSpace; M = M, p = p))`` of $f as a function `(M, p, X) -> Y` or a function `(M, Y, p, X) -> Y` computing `Y` in-place",
)

Glossaries.define!(_glossary_variables, :inverse_retraction_method)
Glossaries.define!(
    _glossary_variables, :inverse_retraction_method, :description,
    "an inverse retraction ``$(_tex(:invretr))`` to use, see [the section on retractions and their inverses](@extref ManifoldsBase :doc:`retractions`)",
)
Glossaries.define!(_glossary_variables, :inverse_retraction_method, :type, "`[`AbstractInverseRetractionMethod`](@extref `ManifoldsBase.AbstractInverseRetractionMethod`)` ")
Glossaries.define!(
    _glossary_variables, :inverse_retraction_method, :default,
    (; M = "M", p = "p") ->
    "`[`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`)`($M, typeof($p))",
)

Glossaries.define!(_glossary_variables, :last_change)
Glossaries.define!(
    _glossary_variables, :last_change, :description,
    "the last change recorded in this stopping criterion",
)
Glossaries.define!(_glossary_variables, :last_change, :type, "Real")


Glossaries.define!(_glossary_variables, :M)
Glossaries.define!(
    _glossary_variables, :M, :description,
    (; M = "M") -> "a Riemannian manifold ``$(_tex(:Cal, M))``"
)
Glossaries.define!(_glossary_variables, :M, :type, "`$(_link(:AbstractManifold))` ")

Glossaries.define!(_glossary_variables, :p)
Glossaries.define!(
    _glossary_variables, :p, :description,
    (; M = "M") -> "a point on the manifold ``$(_tex(:Cal, M))``"
)
Glossaries.define!(_glossary_variables, :p, :type, "P")
Glossaries.define!(_glossary_variables, :p, :default, (; M = "M") -> "`$(_link(:rand; M = M))` ")
Glossaries.define!(_glossary_variables, :p, :as_Iterate, " storing the current iterate")
Glossaries.define!(_glossary_variables, :p, :as_Initial, " to specify the initial value")

Glossaries.define!(_glossary_variables, :retraction_method)
Glossaries.define!(
    _glossary_variables, :retraction_method, :description,
    "a retraction ``$(_tex(:retr))`` to use, see [the section on retractions](@extref ManifoldsBase :doc:`retractions`)",
)
Glossaries.define!(_glossary_variables, :retraction_method, :type, "`[`AbstractRetractionMethod`](@extref `ManifoldsBase.AbstractRetractionMethod`)` ")
Glossaries.define!(
    _glossary_variables, :retraction_method, :default,
    (; M = "M", p = "p") -> "`[`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`)`($M, typeof($p))",
)

Glossaries.define!(_glossary_variables, :storage)
Glossaries.define!(
    _glossary_variables, :storage, :description,
    (; M = "M") -> "a storage to access the previous iterate",
)
Glossaries.define!(_glossary_variables, :storage, :type, "`[`StoreStateAction`](@ref)` ")

Glossaries.define!(_glossary_variables, :stepsize)
Glossaries.define!(_glossary_variables, :stepsize, :description, (; M = "M") -> "a functor inheriting from [`Stepsize`](@ref) to determine a step size")
Glossaries.define!(_glossary_variables, :stepsize, :type, "`[`Stepsize`](@ref)` ")

Glossaries.define!(_glossary_variables, :stopping_criterion)
Glossaries.define!(
    _glossary_variables, :stopping_criterion, :description,
    (; M = "M") -> "a functor indicating that the stopping criterion is fulfilled",
)
Glossaries.define!(_glossary_variables, :stopping_criterion, :type, "`[`StoppingCriterion`](@ref)` ")

Glossaries.define!(_glossary_variables, :sub_kwargs)
Glossaries.define!(_glossary_variables, :sub_kwargs, :description, "a named tuple of keyword arguments that are passed to [`decorate_objective!`](@ref) of the sub solvers objective, the [`decorate_state!`](@ref) of the subsovlers state, and the sub state constructor itself.")
Glossaries.define!(_glossary_variables, :sub_kwargs, :default, "(;)")

Glossaries.define!(_glossary_variables, :sub_problem)
Glossaries.define!(
    _glossary_variables, :sub_problem, :description,
    (; M = "M") -> " specify a problem for a solver or a closed form solution function, which can be allocating or in-place."
)
Glossaries.define!(_glossary_variables, :sub_problem, :type, "Union{`[`AbstractManoptProblem`](@ref)`, F}")

Glossaries.define!(_glossary_variables, :sub_state)
Glossaries.define!(
    _glossary_variables, :sub_state, :description,
    (; M = "M") -> " a state to specify the sub solver to use. For a closed form solution, this indicates the type of function.",
)
Glossaries.define!(_glossary_variables, :sub_state, :type, "Union{`[`AbstractManoptProblem`](@ref)`, F}")

Glossaries.define!(_glossary_variables, :subgrad_f, :name, "∂f")
Glossaries.define!(
    _glossary_variables, :subgrad_f, :description,
    (; M = "M", p = "p", f = "f", kwargs...) -> """
    the subgradient ``∂$f: $(_math(:Manifold; M = M)) → $(_math(:TangentBundle; M = M))`` of ``$f`` as a function `(M, p) -> X` or a function `(M, X, p) -> X` computing `X` in-place. This function should always only return one element from the subgradient.
    """,
)

Glossaries.define!(_glossary_variables, :vector_transport_method)
Glossaries.define!(
    _glossary_variables, :vector_transport_method, :description,
    (; M = "M", p = "p") ->
    "a vector transport ``$(_math(:VectorTransport))`` to use, see [the section on vector transports](@extref ManifoldsBase :doc:`vector_transports`)",
)
Glossaries.define!(_glossary_variables, :vector_transport_method, :type, "`[`AbstractVectorTransportMethod`](@extref `ManifoldsBase.AbstractVectorTransportMethod`)` ")
Glossaries.define!(
    _glossary_variables, :vector_transport_method, :default,
    (; M = "M", p = "p") ->
    "`[`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`)`($M, typeof($p))",
)

Glossaries.define!(_glossary_variables, :X)
Glossaries.define!(
    _glossary_variables, :X, :description,
    (; M = "M", p = "p") ->
    "a tangent vector at the point ``$p`` on the manifold ``$(_tex(:Cal, M))``",
)
Glossaries.define!(_glossary_variables, :X, :type, "T")
Glossaries.define!(_glossary_variables, :X, :default, (; M = "M", p = "p") -> "`$(_link(:zero_vector; M = M, p = p))` ")
Glossaries.define!(_glossary_variables, :X, :as_Gradient, "storing the gradient at the current iterate")
Glossaries.define!(_glossary_variables, :X, :as_Subgradient, "storing a subgradient at the current iterate")
Glossaries.define!(_glossary_variables, :X, :as_Memory, "to specify the representation of a tangent vector")
