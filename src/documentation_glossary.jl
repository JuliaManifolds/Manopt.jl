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

# In general every dictionary here can be either :Symbol-> String or :Symbol -> Dictionary enrties

_MANOPT_DOC_TYPE = Dict{Symbol,Union{String,Dict,Function}}

_manopt_glossary = _MANOPT_DOC_TYPE()

# easier access functions
"""
    glossary(s::Symbol, args...; kwargs...)
    glossary(g::Dict, s::Symbol, args...; kwargs...)

Access an entry in the glossary at `Symbol` s
if that entry is
* a string, this is returned
* a function, it is called with `args...` and `kwargs...` passed
* a dictionary, then the arguments and keyword arguments are passed to this dictionary, assuming `args[1]` is a symbol
"""
#do not document for now, until we have an internals section
glossary(s::Symbol, args...; kwargs...) = glossary(_manopt_glossary, s, args...; kwargs...)
function glossary(g::_MANOPT_DOC_TYPE, s::Symbol, args...; kwargs...)
    return glossary(g[s], args...; kwargs...)
end
glossary(s::String, args...; kwargs...) = s
glossary(f::Function, args...; kwargs...) = f(args...; kwargs...)

define!(s::Symbol, args...) = define!(_manopt_glossary, s, args...)
function define!(g::_MANOPT_DOC_TYPE, s::Symbol, e::Union{String,Function})
    g[s] = e
    return g
end
function define!(g::_MANOPT_DOC_TYPE, s1::Symbol, s2::Symbol, args...)
    !(haskey(g, s1)) && (g[s1] = _MANOPT_DOC_TYPE())
    define!(g[s1], s2, args...)
    return g
end

# ---
# LaTeX

define!(
    :LaTeX,
    :aligned,
    (lines...) ->
        raw"\begin{aligned}\n" *
        "$(join(["   $(line)" for line in lines], raw"\\\\ "))" *
        raw"\n\end{aligned}\n",
)
define!(:LaTeX, :abs, (v) -> raw"\lvert " * "$v" * raw" \rvert")
define!(:LaTeX, :argmin, raw"\operatorname*{arg\,min}")
define!(:LaTeX, :ast, raw"\ast")
define!(:LaTeX, :bar, (letter) -> raw"\bar" * "$(letter)")
define!(:LaTeX, :big, raw"\big")
define!(:LaTeX, :bigl, raw"\bigl")
define!(:LaTeX, :bigr, raw"\bigr")
define!(:LaTeX, :Big, raw"\Big")
define!(:LaTeX, :Bigl, raw"\Bigl")
define!(:LaTeX, :Bigr, raw"\Bigr")
define!(:LaTeX, :Cal, (letter) -> raw"\mathcal " * "$letter")
define!(
    :LaTeX,
    :cases,
    (c...) ->
        raw"\begin{cases}" *
        "$(join(["   $(ci)" for ci in c], raw"\\\\ "))" *
        raw"\end{cases}",
)
define!(:LaTeX, :cdots, raw"\cdots")
define!(:LaTeX, :cot, raw"\cot")
define!(:LaTeX, :ddots, raw"\ddots")
define!(:LaTeX, :deriv, (t = "t") -> raw"\frac{\mathrm{d}}{\mathrm{d}" * "$(t)" * "}")
define!(:LaTeX, :diff, (t = "") -> raw"\mathrm{D}_{" * "$(t)" * "}")
define!(:LaTeX, :displaystyle, raw"\displaystyle")
define!(:LaTeX, :eR, raw"\bar{\mathbb R}")
define!(:LaTeX, :frac, (a, b) -> raw"\frac" * "{$a}{$b}")
define!(:LaTeX, :grad, raw"\operatorname{grad}")
define!(:LaTeX, :hat, (letter) -> raw"\hat{" * "$letter" * "}")
define!(:LaTeX, :Hess, raw"\operatorname{Hess}")
define!(:LaTeX, :Id, raw"\mathrm{Id}")
define!(:LaTeX, :invretr, raw"\operatorname{retr}^{-1}")
define!(:LaTeX, :inner, (a, b; index="") -> "⟨$a,$b⟩_{$index}")
define!(:LaTeX, :log, raw"\log")
define!(:LaTeX, :max, raw"\max")
define!(:LaTeX, :min, raw"\min")
define!(:LaTeX, :norm, (v; index = "") -> raw"\lVert " * "$v" * raw" \rVert" * "_{$index}")
define!(
    :LaTeX,
    :pmatrix,
    (lines...) -> raw"\begin{pmatrix} " * join(lines, raw"\\ ") * raw"\end{pmatrix}",
)
define!(:LaTeX, :operatorname, (name) -> raw"\operatorname{$name}")
define!(:LaTeX, :proj, raw"\operatorname{proj}")
define!(:LaTeX, :prox, raw"\operatorname{prox}")
define!(:LaTeX, :quad, raw"\quad")
define!(:LaTeX, :qquad, raw"\qquad")
define!(:LaTeX, :reflect, raw"\operatorname{refl}")
define!(:LaTeX, :retr, raw"\operatorname{retr}")
define!(:LaTeX, :rm, (letter) -> raw"\mathrm{" * "$letter" * "}")
define!(:LaTeX, :sqrt, (s) -> raw"\sqrt{" * "$s}")
define!(:LaTeX, :subgrad, raw"∂")
define!(:LaTeX, :set, (s) -> raw"\{" * "$s" * raw"\}")
define!(:LaTeX, :sum, (b="", t="") -> raw"\sum" * "_{$b}^{$t}")
define!(:LaTeX, :text, (letter) -> raw"\text{" * "$letter" * "}")
define!(:LaTeX, :tilde, raw"\tilde")
define!(:LaTeX, :transp, raw"\mathrm{T}")
define!(:LaTeX, :vdots, raw"\vdots")
define!(:LaTeX, :vert, raw"\vert")
define!(:LaTeX, :widehat, (letter) -> raw"\widehat{" * "$letter" * "}")
define!(:LaTeX, :widetilde, raw"\widetilde")
_tex(args...; kwargs...) = glossary(:LaTeX, args...; kwargs...)
#
# ---
# Mathematics and semantic symbols
# :symbol the symbol,
# :description the description
define!(:Math, :distance, raw"\mathrm{d}")
define!(:Math, :M, (; M="M") -> _math(:Manifold, :symbol; M=M))
define!(:Math, :Manifold, :symbol, (; M="M") -> _tex(:Cal, M))
define!(:Math, :Manifold, :descrption, "the Riemannian manifold")
define!(:Math, :M, (; M="M") -> _math(:Manifold, :symbol; M=M))
define!(:Math, :Iterate, (; p="p", k="k") -> "$(p)^{($(k))}")
define!(
    :Math,
    :Sequence,
    (var, ind, from, to) -> raw"\{" * "$(var)_$(ind)" * raw"\}" * "_{$(ind)=$from}^{$to}",
)
define!(:Math, :TM, (; M="M") -> _math(:TangentBundle, :symbol; M=M))
define!(:Math, :TangentBundle, :symbol, (; M="M") -> "T$(_tex(:Cal, M))")
define!(
    :Math,
    :TangentBundle,
    :description,
    (; M="M") -> "the tangent bundle of the manifold ``$(_math(:M; M=M))``",
)
define!(:Math, :TpM, (; M="M", p="p") -> _math(:TangentSpace, :symbol; M=M, p=p))
define!(:Math, :TangentSpace, :symbol, (; M="M", p="p") -> "T_{$p}$(_tex(:Cal, M))")
define!(
    :Math,
    :TangentSpace,
    :description,
    (; M="M", p="p") ->
        "the tangent space at the point ``p`` on the manifold ``$(_math(:M; M=M))``",
)
define!(
    :Math, :vector_transport, :symbol, (a="⋅", b="⋅") -> raw"\mathcal T_{" * "$a←$b" * "}"
)
define!(:Math, :vector_transport, :name, "the vector transport")
_math(args...; kwargs...) = glossary(:Math, args...; kwargs...)

#
# ---
# Links
# Collect short forms for links, especially Interdocs ones.
_link(args...; kwargs...) = glossary(:Link, args...; kwargs...)

define!(
    :Link,
    :AbstractManifold,
    "[`AbstractManifold`](@extref `ManifoldsBase.AbstractManifold`)",
)
define!(
    :Link,
    :AbstractPowerManifold,
    "[`AbstractPowerManifold`](@extref `ManifoldsBase.AbstractPowerManifold`)",
)
define!(
    :Link,
    :injectivity_radius,
    "[`injectivity_radius`](@extref `ManifoldsBase.injectivity_radius-Tuple{AbstractManifold}`)",
)
define!(
    :Link,
    :manifold_dimension,
    (; M="M") ->
        "[`manifold_dimension`](@extref `ManifoldsBase.manifold_dimension-Tuple{AbstractManifold}`)$(length(M) > 0 ? "`($M)`" : "")",
)
define!(:Link, :Manopt, "[`Manopt.jl`](https://manoptjl.org)")
define!(
    :Link, :Manifolds, "[`Manifolds.jl`](https://juliamanifolds.github.io/Manifolds.jl/)"
)
define!(
    :Link,
    :ManifoldsBase,
    "[`ManifoldsBase.jl`](https://juliamanifolds.github.io/ManifoldsBase.jl/)",
)
define!(
    :Link,
    :rand,
    (; M="M") ->
        "[`rand`](@extref Base.rand-Tuple{AbstractManifold})$(length(M) > 0 ? "`($M)`" : "")",
)
define!(
    :Link,
    :zero_vector,
    (; M="M", p="p") ->
        "[`zero_vector`](@extref `ManifoldsBase.zero_vector-Tuple{AbstractManifold, Any}`)$(length(M) > 0 ? "`($M, $p)`" : "")",
)

#
#
# Notes / Remarks
_note(args...; kwargs...) = glossary(:Note, args...; kwargs...)
define!(
    :Note,
    :ManifoldDefaultFactory,
    (type::String) -> """
    !!! info
        This function generates a [`ManifoldDefaultsFactory`](@ref) for [`$(type)`](@ref).
        For default values, that depend on the manifold, this factory postpones the construction
        until the manifold from for example a corresponding [`AbstractManoptSolverState`](@ref) is available.
    """,
)
define!(
    :Note,
    :GradientObjective,
    (; objective="gradient_objective", f="f", grad_f="grad_f") -> """
Alternatively to `$f` and `$grad_f` you can provide
the corresponding [`AbstractManifoldFirstOrderObjective`](@ref) `$objective` directly.
""",
)
define!(
    :Note,
    :OtherKeywords,
    "All other keyword arguments are passed to [`decorate_state!`](@ref) for state decorators or [`decorate_objective!`](@ref) for objective decorators, respectively.",
)
define!(
    :Note,
    :OutputSection,
    (; p_min="p^*") -> """
                                                                     # Output

                                                                     The obtained approximate minimizer ``$(p_min)``.
                                                                     To obtain the whole final state of the solver, see [`get_solver_return`](@ref) for details, especially the `return_state=` keyword.
                                                                     """,
)
define!(
    :Note,
    :TutorialMode,
    "If you activate tutorial mode (cf. [`is_tutorial_mode`](@ref)), this solver provides additional debug warnings.",
)
define!(
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

define!(
    :Problem,
    :Constrained,
    (; M="M", p="p") -> """
        ```math
    \\begin{aligned}
    $(_tex(:argmin))_{$p ∈ $(_math(:M; M=M))} & f($p)\\\\
    $(_tex(:text, "subject to"))$(_tex(:quad))&g_i($p) ≤ 0 \\quad $(_tex(:text, " for ")) i= 1, …, m,\\\\
    \\quad & h_j($p)=0 \\quad $(_tex(:text, " for ")) j=1,…,n,
    \\end{aligned}
    ```
    """,
)
define!(
    :Problem,
    :SetConstrained,
    (; M="M", p="p") -> """
    ```math
\\begin{aligned}
$(_tex(:argmin))_{$p ∈ $(_math(:M; M=M))} & f($p)\\\\
$(_tex(:text, "subject to"))$(_tex(:quad))& p ∈ $(_tex(:Cal, "C")) ⊂ $(_math(:M; M=M))
\\end{aligned}
```
""",
)
define!(:Problem, :Default, (; M="M", p="p") -> """
                        ```math
                        $(_tex(:argmin))_{$p ∈ $(_math(:M; M=M))} f($p)
                        ```
                        """)
define!(
    :Problem,
    :NonLinearLeastSquares,
    (; M="M", p="p") -> """
                                                                        ```math
                                                                        $(_tex(:argmin))_{$p ∈ $(_math(:M; M=M))} $(_tex(:frac,1,2)) $(_tex(:sum))_{i=1}^m $(_tex(:abs, "f_i($p)"))^2
                                                                        ```

                                                                        where ``f: $(_math(:M; M=M)) → ℝ^m`` is written with component functions ``f_i: $(_math(:M; M=M)) → ℝ``, ``i=1,…,m``,
                                                                        and each component function is continuously differentiable.
                                                                        """,
)

#
#
# Stopping Criteria
define!(:StoppingCriterion, :Any, "[` | `](@ref StopWhenAny)")
define!(:StoppingCriterion, :All, "[` & `](@ref StopWhenAll)")
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
define!(
    :Variable,
    :Argument,
    # for the symbol s (possibly with special type t)
    # d is the symbol to display, for example for Argument p that in some signatures is called q
    # t is its type (if different from the default _var(s, :type))
    # type= determines whether to display the default or passed type
    # add=[] adds more information from _var(s, add[i]; kwargs...) sub fields
    function (s::Symbol, d="$s", t=""; type=false, add="", kwargs...)
        # create type to display
        disp_type = type ? "::$(length(t) > 0 ? t : _var(s, :type))" : ""
        addv = !isa(add, Vector) ? [add] : add
        disp_add = join([a isa Symbol ? _var(s, a; kwargs...) : "$a" for a in addv])
        return "* `$(d)$(disp_type)`: $(_var(s, :description;kwargs...))$(disp_add)"
    end,
)
define!(
    :Variable,
    :Field,
    function (s::Symbol, d="$s", t=""; type=true, add="", kwargs...)
        disp_type = type ? "::$(length(t) > 0 ? "$(t)" : _var(s, :type))" : ""
        addv = !isa(add, Vector) ? [add] : add
        disp_add = join([a isa Symbol ? _var(s, a; kwargs...) : "$a" for a in addv])
        return "* `$(d)$(disp_type)`: $(_var(s, :description; kwargs...))$(disp_add)"
    end,
)
define!(
    :Variable,
    :Keyword,
    function (
        s::Symbol,
        display="$s",
        t="";
        default="",
        add="",
        type=false,
        description::Bool=true,
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

define!(
    :Variable,
    :at_iteration,
    :description,
    "an integer indicating at which the stopping criterion last indicted to stop, which might also be before the solver started (`0`). Any negative value indicates that this was not yet the case;",
)
define!(:Variable, :at_iteration, :type, "Int")

define!(
    :Variable,
    :differential,
    :description,
    "specify a specific function to evaluate the differential. By default, ``Df(p)[X] = ⟨$(_tex(:grad))f(p),X⟩``. is used",
)
define!(:Variable, :differential, :default, "`nothing`")

define!(
    :Variable,
    :evaluation,
    :description,
    "specify whether the functions that return an array, for example a point or a tangent vector, work by allocating its result ([`AllocatingEvaluation`](@ref)) or whether they modify their input argument to return the result therein ([`InplaceEvaluation`](@ref)). Since usually the first argument is the manifold, the modified argument is the second.",
)
define!(:Variable, :evaluation, :type, "AbstractEvaluationType")
define!(:Variable, :evaluation, :default, "[`AllocatingEvaluation`](@ref)`()`")
define!(
    :Variable,
    :evaluation,
    :GradientExample,
    "For example `grad_f(M,p)` allocates, but `grad_f!(M, X, p)` computes the result in-place of `X`.",
)

define!(
    :Variable,
    :f,
    :description,
    function (; M="M", p="p")
        return "a cost function ``f: $(_tex(:Cal, M))→ ℝ`` implemented as `($M, $p) -> v`"
    end,
)

define!(
    :Variable,
    :grad_f,
    :description,
    (; M="M", p="p") ->
        "the (Riemannian) gradient ``$(_tex(:grad))f: $(_math(:M, M=M)) → $(_math(:TpM; M=M, p=p))`` of f as a function `(M, p) -> X` or a function `(M, X, p) -> X` computing `X` in-place",
)

define!(
    :Variable,
    :Hess_f,
    :description,
    (; M="M", p="p") ->
        "the (Riemannian) Hessian ``$(_tex(:Hess))f: $(_math(:TpM, M=M, p=p)) → $(_math(:TpM; M=M, p=p))`` of f as a function `(M, p, X) -> Y` or a function `(M, Y, p, X) -> Y` computing `Y` in-place",
)

define!(
    :Variable,
    :inverse_retraction_method,
    :description,
    "an inverse retraction ``$(_tex(:invretr))`` to use, see [the section on retractions and their inverses](@extref ManifoldsBase :doc:`retractions`)",
)
define!(:Variable, :inverse_retraction_method, :type, "AbstractInverseRetractionMethod")
define!(
    :Variable,
    :inverse_retraction_method,
    :default,
    (; M="M", p="p") ->
        "[`default_inverse_retraction_method`](@extref `ManifoldsBase.default_inverse_retraction_method-Tuple{AbstractManifold}`)`($M, typeof($p))`",
)
define!(
    :Variable,
    :last_change,
    :description,
    "the last change recorded in this stopping criterion",
)
define!(:Variable, :last_change, :type, "Real")

define!(
    :Variable, :M, :description, (; M="M") -> "a Riemannian manifold ``$(_tex(:Cal, M))``"
)
define!(:Variable, :M, :type, "`$(_link(:AbstractManifold))` ")

define!(
    :Variable, :p, :description, (; M="M") -> "a point on the manifold ``$(_tex(:Cal, M))``"
)
define!(:Variable, :p, :type, "P")
define!(:Variable, :p, :default, (; M="M") -> _link(:rand; M=M))
define!(:Variable, :p, :as_Iterate, " storing the current iterate")
define!(:Variable, :p, :as_Initial, " to specify the initial value")

define!(
    :Variable,
    :retraction_method,
    :description,
    "a retraction ``$(_tex(:retr))`` to use, see [the section on retractions](@extref ManifoldsBase :doc:`retractions`)",
)
define!(:Variable, :retraction_method, :type, "AbstractRetractionMethod")
define!(
    :Variable,
    :retraction_method,
    :default,
    "[`default_retraction_method`](@extref `ManifoldsBase.default_retraction_method-Tuple{AbstractManifold}`)`(M, typeof(p))`",
)

define!(
    :Variable,
    :storage,
    :description,
    (; M="M") -> "a storage to access the previous iterate",
)
define!(:Variable, :storage, :type, "StoreStateAction")
define!(
    :Variable,
    :stepsize,
    :description,
    (; M="M") -> "a functor inheriting from [`Stepsize`](@ref) to determine a step size",
)
define!(:Variable, :stepsize, :type, "Stepsize")

define!(
    :Variable,
    :stopping_criterion,
    :description,
    (; M="M") -> "a functor indicating that the stopping criterion is fulfilled",
)
define!(:Variable, :stopping_criterion, :type, "StoppingCriterion")

define!(
    :Variable,
    :sub_kwargs,
    :description,
    "a named tuple of keyword arguments that are passed to [`decorate_objective!`](@ref) of the sub solvers objective, the [`decorate_state!`](@ref) of the subsovlers state, and the sub state constructor itself.",
)
define!(:Variable, :sub_kwargs, :default, "`(;)`")

define!(
    :Variable,
    :sub_problem,
    :description,
    (; M="M") ->
        " specify a problem for a solver or a closed form solution function, which can be allocating or in-place.",
)
define!(:Variable, :sub_problem, :type, "Union{AbstractManoptProblem, F}")

define!(
    :Variable,
    :sub_state,
    :description,
    (; M="M") ->
        " a state to specify the sub solver to use. For a closed form solution, this indicates the type of function.",
)
define!(:Variable, :sub_state, :type, "Union{AbstractManoptProblem, F}")

define!(:Variable, :subgrad_f, :symbol, "∂f")
define!(
    :Variable,
    :subgrad_f,
    :description,
    (; M="M", p="p") -> """
                                                                        the subgradient ``∂f: $(_math(:M; M=M)) → $(_math(:TM; M=M))`` of f as a function `(M, p) -> X`
                                                                        or a function `(M, X, p) -> X` computing `X` in-place.
                                                                        This function should always only return one element from the subgradient.
                                                                        """,
)
define!(
    :Variable,
    :subgrad_f,
    :description,
    (; M="M") ->
        " a state to specify the sub solver to use. For a closed form solution, this indicates the type of function.",
)

define!(
    :Variable,
    :vector_transport_method,
    :description,
    (; M="M", p="p") ->
        "a vector transport ``$(_math(:vector_transport, :symbol))`` to use, see [the section on vector transports](@extref ManifoldsBase :doc:`vector_transports`)",
)
define!(:Variable, :vector_transport_method, :type, "AbstractVectorTransportMethodP")
define!(
    :Variable,
    :vector_transport_method,
    :default,
    (; M="M", p="p") ->
        "[`default_vector_transport_method`](@extref `ManifoldsBase.default_vector_transport_method-Tuple{AbstractManifold}`)`($M, typeof($p))`",
)

define!(
    :Variable,
    :X,
    :description,
    (; M="M", p="p") ->
        "a tangent vector at the point ``$p`` on the manifold ``$(_tex(:Cal, M))``",
)
define!(:Variable, :X, :type, "T")
define!(:Variable, :X, :default, (; M="M", p="p") -> _link(:zero_vector; M=M, p=p))
define!(:Variable, :X, :as_Gradient, "storing the gradient at the current iterate")
define!(:Variable, :X, :as_Subgradient, "storing a subgradient at the current iterate")
define!(:Variable, :X, :as_Memory, "to specify the representation of a tangent vector")
