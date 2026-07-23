function status_summary end

_doc_status_summary = """
    status_summary(io, e; context::Symbol = :default)
    status_summary(e; context::Symbol = :default)

Returns a string reporting about the current status of an element `e` defined in `Manopt.jl`,
which can also directly be printed to an `IO` stream `io`.
This method should generate a human readable summary of `e`.

By default, the variant with an `IO` stream dispatches to the one without to generate
a string and prints it to the `IO` stream.
If you implement the variant with the stream `io` remember to also provide the one without
Similarly, the

The summary is meant to be used in different contexts
* `:default` should be the default and refers to a (multiline) context in REPL where a
  human should read a comprehensive summary of `e`
  This should also be the default
* `:inline` should be a shorter variant that can be used inline of other summaries, e.g. in lists
* `:short` should be a form even shorter or equal to `inline`, for example when in a list,
  a certain element, like a [`DebugAction`](@ref) can be represented by a symbol.
  The short variant should by default fall back to `:inline`
"""

@doc "$(_doc_status_summary)"
status_summary(e; context::Symbol = :default)

@doc "$(_doc_status_summary)"
function status_summary(io::IO, e; context::Symbol = :default)
    return print(io, status_summary(e; context = context))
end
#
#
# status_summary string format helper
# ---
# check whether a context is inline or less
_is_inline(c) = (c == :inline || c == :short)
# _in_str - indent a string for use within another one
# * `indent = false` raise indentation by `indent_str` (`_MANOPT_INDENT` by default)
# * `headers = true` increase headers also on Headers that are indented with `indent_str`
# * `indent_str = _MANOPT_INDENT` string to use for indent
# * `indent_end = ""` a string to end the indentation, for example a `"| "` for visual distinction
function _in_str(s::String; indent = 0, headers = 1, indent_str = _MANOPT_INDENT, indent_end = "")
    t = s
    #add start
    t = replace("$(indent_end)$t", "\n" => "\n$(indent_end)")
    #add indent iteratively
    for _ in 1:indent
        t = replace("$(indent_str)$t", "\n" => "\n$(indent_str)")
    end
    # increase headers iteratively
    for _ in 1:headers
        t = replace(t, Regex("(?m)^($(indent_str)*)(#+)") => s"\1#\2")
    end
    return t
end

"""
    is_tutorial_mode()

A small internal helper to indicate whether tutorial mode is active.

You can set the mode by calling `set_parameter!(:Mode, "Tutorial")` or deactivate it
by `set_parameter!(:Mode, "")`.
"""
is_tutorial_mode() = (get_parameter(:Mode) == "Tutorial")

# include this first because all following elements might define keyword helpers.

include("solver_state.jl")

include("stopping_criterion.jl")

include("stepsize/initial_guess.jl")
include("stepsize/stepsize_message.jl")
include("stepsize/linesearch.jl")
include("stepsize/stepsize.jl")

# Generic plans I: based on objective structure
include("bundle_plan.jl")
include("hessian_plan.jl")
include("subgradient_plan.jl")
include("vectorial_plan.jl")

# Linear systems
include("conjugate_residual_plan.jl")
# Robutsifiers
include("robustifiers.jl")

# Generic plans II: based on subsolvers
include("subsolver_plan.jl")
include("constrained_plan.jl")
include("constrained_set_plan.jl")
include("trust_regions_plan.jl")

# Specific solver plans
include("adaptive_regularization_with_cubics_plan.jl")
include("alternating_gradient_plan.jl")
include("augmented_lagrangian_plan.jl")
include("conjugate_gradient_plan.jl")
include("exact_penalty_method_plan.jl")
include("interior_point_Newton_plan.jl")
include("quasi_newton_plan.jl")
include("nonlinear_least_squares/linear_surrogate_plan.jl")
include("nonlinear_least_squares/nls_objective.jl")
include("nonlinear_least_squares/nls_general_plan.jl")
include("nonlinear_least_squares/nls_in_coordinates_plan.jl")
include("nonlinear_least_squares/box_nls_plan.jl")
include("difference_of_convex_plan.jl")

include("primal_dual_plan.jl")
include("higher_order_primal_dual_plan.jl")

include("stochastic_gradient_plan.jl")

include("box_plan.jl")

include("embedded_objective.jl")
include("scaled_objective.jl")

include("cache.jl")
include("count.jl")
