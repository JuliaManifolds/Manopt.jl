#
# Manopt Glossary
# ===
#
# This file collects
# * LaTeX snippets
# * math formulae
# * Variable names
# * links
# * notes
#
# to keep naming, notation, and formatting

# In general every dictionary here can be either :Symbol-> String or :Symbol -> Dictionary entries

# generate Glossary for Manopt.jl
Glossaries.@Glossary

Glossaries.@define!(
    :LaTeX,
    :aligned,
    (lines...) ->
    raw"\begin{aligned}\n" *
        "$(join(["   $(line)" for line in lines], raw"\\\\ "))" *
        raw"\n\end{aligned}\n",
)
Glossaries.@define!(:LaTeX, :abs, (v) -> raw"\lvert " * "$v" * raw" \rvert")
Glossaries.@define!(:LaTeX, :argmin, raw"\operatorname*{arg\,min}")
Glossaries.@define!(:LaTeX, :ast, raw"\ast")
Glossaries.@define!(:LaTeX, :bar, (letter) -> raw"\bar" * "$(letter)")
Glossaries.@define!(:LaTeX, :big, raw"\big")
Glossaries.@define!(:LaTeX, :bigl, raw"\bigl")
Glossaries.@define!(:LaTeX, :bigr, raw"\bigr")
Glossaries.@define!(:LaTeX, :biggl, raw"\biggl")
Glossaries.@define!(:LaTeX, :biggr, raw"\biggr")
Glossaries.@define!(:LaTeX, :Big, raw"\Big")
Glossaries.@define!(:LaTeX, :Bigl, raw"\Bigl")
Glossaries.@define!(:LaTeX, :Bigr, raw"\Bigr")
Glossaries.@define!(:LaTeX, :Cal, (letter) -> raw"\mathcal " * "$letter")
Glossaries.@define!(
    :LaTeX,
    :cases,
    (c...) ->
    raw"\begin{cases}" *
        "$(join(["   $(ci)" for ci in c], raw"\\\\ "))" *
        raw"\end{cases}",
)
Glossaries.@define!(:LaTeX, :cdots, raw"\cdots")
Glossaries.@define!(:LaTeX, :cot, raw"\cot")
Glossaries.@define!(:LaTeX, :ddots, raw"\ddots")
Glossaries.@define!(:LaTeX, :deriv, (t = "t") -> raw"\frac{\mathrm{d}}{\mathrm{d}" * "$(t)" * "}")
Glossaries.@define!(:LaTeX, :diff, (t = "") -> raw"\mathrm{D}_{" * "$(t)" * "}")
Glossaries.@define!(:LaTeX, :displaystyle, raw"\displaystyle")
Glossaries.@define!(:LaTeX, :eR, raw"\bar{\mathbb R}")
Glossaries.@define!(:LaTeX, :frac, (a, b) -> raw"\frac" * "{$a}{$b}")
Glossaries.@define!(:LaTeX, :grad, raw"\operatorname{grad}")
Glossaries.@define!(:LaTeX, :hat, (letter) -> raw"\hat{" * "$letter" * "}")
Glossaries.@define!(:LaTeX, :Hess, raw"\operatorname{Hess}")
Glossaries.@define!(:LaTeX, :Id, raw"\mathrm{Id}")
Glossaries.@define!(:LaTeX, :invretr, raw"\operatorname{retr}^{-1}")
Glossaries.@define!(:LaTeX, :inner, (a, b; index = "") -> "⟨$a,$b⟩_{$index}")
Glossaries.@define!(:LaTeX, :log, raw"\log")
Glossaries.@define!(:LaTeX, :max, raw"\max")
Glossaries.@define!(:LaTeX, :min, raw"\min")
Glossaries.@define!(:LaTeX, :norm, (v; index = "") -> raw"\lVert " * "$v" * raw" \rVert" * "_{$index}")
Glossaries.@define!(
    :LaTeX, :pmatrix,
    (lines...) -> raw"\begin{pmatrix} " * join(lines, raw"\\ ") * raw"\end{pmatrix}",
)
Glossaries.@define!(:LaTeX, :operatorname, (name) -> raw"\operatorname{$name}")
Glossaries.@define!(:LaTeX, :proj, raw"\operatorname{proj}")
Glossaries.@define!(:LaTeX, :prox, raw"\operatorname{prox}")
Glossaries.@define!(:LaTeX, :quad, raw"\quad")
Glossaries.@define!(:LaTeX, :qquad, raw"\qquad")
Glossaries.@define!(:LaTeX, :reflect, raw"\operatorname{refl}")
Glossaries.@define!(:LaTeX, :retr, raw"\operatorname{retr}")
Glossaries.@define!(:LaTeX, :rm, (letter) -> raw"\mathrm{" * "$letter" * "}")
Glossaries.@define!(:LaTeX, :sqrt, (s) -> raw"\sqrt{" * "$s}")
Glossaries.@define!(:LaTeX, :subgrad, raw"∂")
Glossaries.@define!(:LaTeX, :set, (s) -> raw"\{" * "$s" * raw"\}")
Glossaries.@define!(:LaTeX, :sum, (b = "", t = "") -> raw"\sum" * (length(b) > 0 ? "_{$b}" : "") * (length(t) > 0 ? "^{$t}" : ""))
Glossaries.@define!(:LaTeX, :text, (letter) -> raw"\text{" * "$letter" * "}")
Glossaries.@define!(:LaTeX, :tilde, raw"\tilde")
Glossaries.@define!(:LaTeX, :transp, raw"\mathrm{T}")
Glossaries.@define!(:LaTeX, :vdots, raw"\vdots")
Glossaries.@define!(:LaTeX, :vert, raw"\vert")
Glossaries.@define!(:LaTeX, :widehat, (letter) -> raw"\widehat{" * "$letter" * "}")
Glossaries.@define!(:LaTeX, :widetilde, (letter) -> raw"\widetilde{" * "$letter" * "}")
_tex(args...; kwargs...) = glossary(:LaTeX, args...; kwargs...)
#
# ---
# Mathematics and semantic symbols
# :symbol the symbol,
# :description the description
Glossaries.@define!(:distance, :math, raw"\mathrm{d}")
Glossaries.@define!(:Manifold, :math, (; M = "M") -> _tex(:Cal, M))
Glossaries.@define!(:Manifold, :description, "the Riemannian manifold")
Glossaries.@define!(:Iterate, :math, (; p = "p", k = "k") -> "$(p)^{($(k))}")
Glossaries.@define!(:Sequence, :math,
    (var, ind, from, to) -> raw"\{" * "$(var)_$(ind)" * raw"\}" * "_{$(ind)=$from}^{$to}",
)
Glossaries.@define!(:TangentBundle, :math, (; M = "M") -> "T$(_tex(:Cal, M))")
Glossaries.@define!(:TangentBundle,
    :description,
    (; M = "M") -> "the tangent bundle of the manifold ``$(_math(:M; M = M))``",
)
Glossaries.@define!(:TangentSpace, :math, (; M = "M", p = "p") -> "T_{$p}$(_tex(:Cal, M))")
Glossaries.@define!(
    :TangentSpace, :description,
    (; M = "M", p = "p") -> "the tangent space at the point ``p`` on the manifold ``$(_math(:M; M = M))``",
)
Glossaries.@define!(
    :vector_transport, :math, (a = "⋅", b = "⋅") -> raw"\mathcal T_{" * "$a←$b" * "}"
)
Glossaries.@define!(:vector_transport, :name, "the vector transport")

# TODO: Replace with new formatter.
_math(args...; kwargs...) = glossary(:Math, args...; kwargs...)

#
# ---
# Links
# Collect short forms for links, especially Interdocs ones.
_link(args...; kwargs...) = glossary(:Link, args...; kwargs...)

Glossaries.@define!(
    :Link, :AbstractManifold,
    "[`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)",
)
Glossaries.@define!(
    :Link, :AbstractPowerManifold,
    "[`AbstractPowerManifold`](@extref `ManifoldsBase.AbstractPowerManifold`)",
)
Glossaries.@define!(
    :Link, :injectivity_radius,
    "[`injectivity_radius`](@extref `ManifoldsBase.injectivity_radius-Tuple{AbstractManifold}`)",
)
Glossaries.@define!(
    :Link, :manifold_dimension,
    (; M = "M") ->
    "[`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`)$(length(M) > 0 ? "`($M)`" : "")",
)
Glossaries.@define!(:Link, :Manopt, "[`Manopt.jl`](https://manoptjl.org)")
Glossaries.@define!(
    :Link, :Manifolds, "[`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/)"
)
Glossaries.@define!(
    :Link, :ManifoldsBase,
    "[`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/)",
)
Glossaries.@define!(
    :Link, :rand,
    (; M = "M") ->
    "[`rand`](@extref Base.rand-Tuple{AbstractManifold})$(length(M) > 0 ? "`($M)`" : "")",
)
Glossaries.@define!(
    :Link, :zero_vector,
    (; M = "M", p = "p") ->
    "[`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)$(length(M) > 0 ? "`($M, $p)`" : "")",
)

#
#
# Notes / Remarks
_note(args...; kwargs...) = glossary(:Note, args...; kwargs...)
Glossaries.@define!(
    :Note,
    :ManifoldDefaultFactory,
    (type::String) -> """
    !!! info
        This function generates a [`ManifoldDefaultsFactory`](@ref) for [`$(type)`](@ref).
        For default values, that depend on the manifold, this factory postpones the construction
        until the manifold from for example a corresponding [`AbstractManoptSolverState`](@ref) is available.
    """,
)
Glossaries.@define!(
    :Note,
    :GradientObjective,
    (; objective = "gradient_objective", f = "f", grad_f = "grad_f") -> """
    Alternatively to `$f` and `$grad_f` you can provide
    the corresponding [`AbstractManifoldFirstOrderObjective`](@ref) `$objective` directly.
    """,
)
Glossaries.@define!(
    :Note,
    :OtherKeywords,
    "All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or [`decorate_objective!`](@ref) for objective decorators, respectively.",
)
Glossaries.@define!(
    :Note,
    :OutputSection,
    (; p_min = "p^*") -> """
    # Output

    The obtained approximate minimizer ``$(p_min)``.
    To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details, especially the `return_state=` keyword.
    """,
)
Glossaries.@define!(
    :Note,
    :TutorialMode,
    "If you activate tutorial mode (cf. [`is_tutorial_mode`](@ref)), this solver provides additional debug warnings.",
)
Glossaries.@define!(
    :Note,
    :KeywordUsedIn,
    function (kw::String)
        return "This is used to define the `$(kw)=` keyword and has hence no effect, if you set `$(kw)` directly."
    end,
)

#
#
# Problems
_problem(args...; kwargs...) = glossary(:Problem, args...; kwargs...)

Glossaries.@define!(
    :Problem,
    :Constrained,
    (; M = "M", p = "p") -> """
        ```math
    \\begin{aligned}
    $(_tex(:argmin))_{$p ∈ $(_math(:M; M = M))} & f($p)\\\\
    $(_tex(:text, "subject to"))$(_tex(:quad))&g_i($p) ≤ 0 \\quad $(_tex(:text, " for ")) i= 1, …, m,\\\\
    \\quad & h_j($p)=0 \\quad $(_tex(:text, " for ")) j=1,…,n,
    \\end{aligned}
    ```
    """,
)
Glossaries.@define!(
    :Problem,
    :SetConstrained,
    (; M = "M", p = "p") -> """
        ```math
    \\begin{aligned}
    $(_tex(:argmin))_{$p ∈ $(_math(:M; M = M))} & f($p)\\\\
    $(_tex(:text, "subject to"))$(_tex(:quad))& p ∈ $(_tex(:Cal, "C")) ⊂ $(_math(:M; M = M))
    \\end{aligned}
    ```
    """,
)
Glossaries.@define!(
    :Problem, :Default, (; M = "M", p = "p") -> """
    ```math
    $(_tex(:argmin))_{$p ∈ $(_math(:M; M = M))} f($p)
    ```
    """
)
Glossaries.@define!(
    :Problem,
    :NonLinearLeastSquares,
    (; M = "M", p = "p") -> """
    ```math
    $(_tex(:argmin))_{$p ∈ $(_math(:M; M = M))} $(_tex(:frac, 1, 2)) $(_tex(:sum, "i=1", "m")) $(_tex(:abs, "f_i($p)"))^2
    ```

    where ``f: $(_math(:M; M = M)) → ℝ^m`` is written with component functions ``f_i: $(_math(:M; M = M)) → ℝ``, ``i=1,…,m``,
    and each component function is continuously differentiable.
    """,
)

#
#
# Stopping Criteria
Glossaries.@define!(:StoppingCriterion, :Any, "[` | `](@ref StopWhenAny)")
Glossaries.@define!(:StoppingCriterion, :All, "[` & `](@ref StopWhenAll)")
_sc(args...; kwargs...) = glossary(:StoppingCriterion, args...; kwargs...)

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
_var(args...; kwargs...) = glossary(:Variable, args...; kwargs...)

#Meta: How to format an argument, a field of a struct, and a keyword
Glossaries.@define!(
    :Variable,
    :Argument,
    # for the symbol s (possibly with special type t)
    # d is the symbol to display, for example for Argument p that in some signatures is called q
    # t is its type (if different from the default _var(s, :type))
    # type= determines whether to display the default or passed type
    # add=[] adds more information from _var(s, add[i]; kwargs...) sub fields
    function (s::Symbol, d = "$s", t = ""; type = false, add = "", kwargs...)
        # create type to display
        disp_type = type ? "::$(length(t) > 0 ? t : _var(s, :type))" : ""
        addv = !isa(add, Vector) ? [add] : add
        disp_add = join([a isa Symbol ? _var(s, a; kwargs...) : "$a" for a in addv])
        return "* `$(d)$(disp_type)`: $(_var(s, :description; kwargs...))$(disp_add)"
    end,
)
Glossaries.@define!(
    :Variable,
    :Field,
    function (s::Symbol, d = "$s", t = ""; type = true, add = "", kwargs...)
        disp_type = type ? "::$(length(t) > 0 ? "$(t)" : _var(s, :type))" : ""
        addv = !isa(add, Vector) ? [add] : add
        disp_add = join([a isa Symbol ? _var(s, a; kwargs...) : "$a" for a in addv])
        return "* `$(d)$(disp_type)`: $(_var(s, :description; kwargs...))$(disp_add)"
    end,
)
Glossaries.@define!(
    :Variable,
    :Keyword,
    function (
            s::Symbol,
            display = "$s",
            t = "";
            default = "",
            add = "",
            type = false,
            description::Bool = true,
            kwargs...,
        )
        addv = !isa(add, Vector) ? [add] : add
        disp_add = join([a isa Symbol ? _var(s, a; kwargs...) : "$a" for a in addv])
        return "* `$(display)$(type ? "::$(length(t) > 0 ? t : _var(s, :type))" : "")=`$(length(default) > 0 ? default : _var(s, :default; kwargs...))$(description ? ": $(_var(s, :description; kwargs...))" : "")$(disp_add)"
    end,
)

#
#
#
# variables / Names used in Arguments, Fields, and Keywords

Glossaries.@define!(
    :Variable,
    :at_iteration,
    :description,
    "an integer indicating at which the stopping criterion last indicted to stop, which might also be before the solver started (`0`). Any negative value indicates that this was not yet the case;",
)
Glossaries.@define!(:Variable, :at_iteration, :type, "Int")

Glossaries.@define!(
    :Variable,
    :differential,
    :description,
    "specify a specific function to evaluate the differential. By default, ``Df(p)[X] = ⟨$(_tex(:grad))f(p),X⟩``. is used",
)
Glossaries.@define!(:Variable, :differential, :default, "`nothing`")

Glossaries.@define!(
    :Variable,
    :evaluation,
    :description,
    "specify whether the functions that return an array, for example a point or a tangent vector, work by allocating its result ([`AllocatingEvaluation`](@ref)) or whether they modify their input argument to return the result therein ([`InplaceEvaluation`](@ref)). Since usually the first argument is the manifold, the modified argument is the second.",
)
Glossaries.@define!(:Variable, :evaluation, :type, "AbstractEvaluationType")
Glossaries.@define!(:Variable, :evaluation, :default, "[`AllocatingEvaluation`](@ref)`()`")
Glossaries.@define!(
    :Variable,
    :evaluation,
    :GradientExample,
    "For example `grad_f(M,p)` allocates, but `grad_f!(M, X, p)` computes the result in-place of `X`.",
)

Glossaries.@define!(
    :Variable,
    :f,
    :description,
    function (; M = "M", p = "p")
        return "a cost function ``f: $(_tex(:Cal, M))→ ℝ`` implemented as `($M, $p) -> v`"
    end,
)

Glossaries.@define!(
    :Variable,
    :grad_f,
    :description,
    (; M = "M", p = "p", f = "f", kwargs...) ->
    "the (Riemannian) gradient ``$(_tex(:grad))$f: $(_math(:M, M = M)) → $(_math(:TpM; M = M, p = p))`` of $f as a function `(M, p) -> X` or a function `(M, X, p) -> X` computing `X` in-place",
)

Glossaries.@define!(
    :Variable,
    :Hess_f,
    :description,
    (; M = "M", p = "p", f = "f") ->
    "the (Riemannian) Hessian ``$(_tex(:Hess))$f: $(_math(:TpM, M = M, p = p)) → $(_math(:TpM; M = M, p = p))`` of $f as a function `(M, p, X) -> Y` or a function `(M, Y, p, X) -> Y` computing `Y` in-place",
)

Glossaries.@define!(
    :Variable,
    :inverse_retraction_method,
    :description,
    "an inverse retraction ``$(_tex(:invretr))`` to use, see [the section on retractions and their inverses](@extref ManifoldsBase :doc:`retractions`)",
)
Glossaries.@define!(:Variable, :inverse_retraction_method, :type, "AbstractInverseRetractionMethod")
Glossaries.@define!(
    :Variable,
    :inverse_retraction_method,
    :default,
    (; M = "M", p = "p") ->
    "[`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`)`($M, typeof($p))`",
)
Glossaries.@define!(
    :Variable,
    :last_change,
    :description,
    "the last change recorded in this stopping criterion",
)
Glossaries.@define!(:Variable, :last_change, :type, "Real")

Glossaries.@define!(
    :Variable, :M, :description, (; M = "M") -> "a Riemannian manifold ``$(_tex(:Cal, M))``"
)
Glossaries.@define!(:Variable, :M, :type, "`$(_link(:AbstractManifold))` ")

Glossaries.@define!(
    :Variable, :p, :description, (; M = "M") -> "a point on the manifold ``$(_tex(:Cal, M))``"
)
Glossaries.@define!(:Variable, :p, :type, "P")
Glossaries.@define!(:Variable, :p, :default, (; M = "M") -> _link(:rand; M = M))
Glossaries.@define!(:Variable, :p, :as_Iterate, " storing the current iterate")
Glossaries.@define!(:Variable, :p, :as_Initial, " to specify the initial value")

Glossaries.@define!(
    :Variable,
    :retraction_method,
    :description,
    "a retraction ``$(_tex(:retr))`` to use, see [the section on retractions](@extref ManifoldsBase :doc:`retractions`)",
)
Glossaries.@define!(:Variable, :retraction_method, :type, "AbstractRetractionMethod")
Glossaries.@define!(
    :Variable,
    :retraction_method,
    :default,
    "[`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`)`(M, typeof(p))`",
)

Glossaries.@define!(
    :Variable,
    :storage,
    :description,
    (; M = "M") -> "a storage to access the previous iterate",
)
Glossaries.@define!(:Variable, :storage, :type, "StoreStateAction")
Glossaries.@define!(
    :Variable,
    :stepsize,
    :description,
    (; M = "M") -> "a functor inheriting from [`Stepsize`](@ref) to determine a step size",
)
Glossaries.@define!(:Variable, :stepsize, :type, "Stepsize")

Glossaries.@define!(
    :Variable,
    :stopping_criterion,
    :description,
    (; M = "M") -> "a functor indicating that the stopping criterion is fulfilled",
)
Glossaries.@define!(:Variable, :stopping_criterion, :type, "StoppingCriterion")

Glossaries.@define!(
    :Variable,
    :sub_kwargs,
    :description,
    "a named tuple of keyword arguments that are passed to [`decorate_objective!`](@ref) of the sub solvers objective, the [`decorate_state!`](@ref) of the subsovlers state, and the sub state constructor itself.",
)
Glossaries.@define!(:Variable, :sub_kwargs, :default, "`(;)`")

Glossaries.@define!(
    :Variable,
    :sub_problem,
    :description,
    (; M = "M") ->
    " specify a problem for a solver or a closed form solution function, which can be allocating or in-place.",
)
Glossaries.@define!(:Variable, :sub_problem, :type, "Union{AbstractManoptProblem, F}")

Glossaries.@define!(
    :Variable,
    :sub_state,
    :description,
    (; M = "M") ->
    " a state to specify the sub solver to use. For a closed form solution, this indicates the type of function.",
)
Glossaries.@define!(:Variable, :sub_state, :type, "Union{AbstractManoptProblem, F}")

Glossaries.@define!(:Variable, :subgrad_f, :symbol, "∂f")
Glossaries.@define!(
    :Variable,
    :subgrad_f,
    :description,
    (; M = "M", p = "p", f = "f", kwargs...) -> """
    the subgradient ``∂$f: $(_math(:M; M = M)) → $(_math(:TM; M = M))`` of ``$f`` as a function `(M, p) -> X` or a function `(M, X, p) -> X` computing `X` in-place. This function should always only return one element from the subgradient.
    """,
)

Glossaries.@define!(
    :Variable,
    :vector_transport_method,
    :description,
    (; M = "M", p = "p") ->
    "a vector transport ``$(_math(:vector_transport, :symbol))`` to use, see [the section on vector transports](@extref ManifoldsBase :doc:`vector_transports`)",
)
Glossaries.@define!(:Variable, :vector_transport_method, :type, "AbstractVectorTransportMethodP")
Glossaries.@define!(
    :Variable,
    :vector_transport_method,
    :default,
    (; M = "M", p = "p") ->
    "[`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`)`($M, typeof($p))`",
)

Glossaries.@define!(
    :Variable,
    :X,
    :description,
    (; M = "M", p = "p") ->
    "a tangent vector at the point ``$p`` on the manifold ``$(_tex(:Cal, M))``",
)
Glossaries.@define!(:Variable, :X, :type, "T")
Glossaries.@define!(:Variable, :X, :default, (; M = "M", p = "p") -> _link(:zero_vector; M = M, p = p))
Glossaries.@define!(:Variable, :X, :as_Gradient, "storing the gradient at the current iterate")
Glossaries.@define!(:Variable, :X, :as_Subgradient, "storing a subgradient at the current iterate")
Glossaries.@define!(:Variable, :X, :as_Memory, "to specify the representation of a tangent vector")
