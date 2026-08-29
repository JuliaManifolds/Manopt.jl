_doc_status_summary = """
    status_summary(io, e; context::Symbol = :default)
    status_summary(e; context::Symbol = :default)

Return a string reporting about the current status of an element `e` defined in `Manopt.jl`,
which can also directly be printed to an `IO` stream `io`.
This method should generate a human readable summary of `e`.

By default, the variant with an `IO` stream dispatches to the one without to generate
a string and prints it to the `IO` stream.
If you implement the variant with the stream `io`, remember to also provide the one without.

The summary is meant to be used in different contexts:

* `:default` refers to a (multiline) context in the REPL where a
  human should read a comprehensive summary of `e`. This is also the default.
* `:inline` should be a shorter variant that can be used inline of other summaries, e.g. in lists
* `:short` should be a form even shorter or equal to `:inline`, for example when in a list,
  a certain element, like a [`DebugAction`](@ref) can be represented by a symbol.
  The short variant should by default fall back to `:inline`.
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
# _ordinal_suffix - the English ordinal suffix of a nonnegative integer,
# for example 1 -> "st", 12 -> "th", 23 -> "rd". Julia provides no such helper.
# Used for the “every nth iteration” phrasing of the `…Every` status summaries.
function _ordinal_suffix(n::Integer)
    (n % 100) ∈ 11:13 && return "th" # the teens are always “th”
    r = n % 10
    (r == 1) && return "st"
    (r == 2) && return "nd"
    (r == 3) && return "rd"
    return "th"
end
# _in_str - indent a string for use within another one
# * `indent = 0` how often to raise indentation by `indent_str` (`_MANOPT_INDENT` by default)
# * `headers = 1` how often to increase headers, also on headers that are already indented
# * `indent_str = _MANOPT_INDENT` string to use for indent
# * `indent_end = ""` a string to end the indentation, for example a `"| "` for visual distinction
#
# The headers are raised before indenting, so that this works for any `indent` and `indent_end`,
# and empty lines are not indented, since that would only introduce trailing whitespace.
function _in_str(s::String; indent = 0, headers = 1, indent_str = _MANOPT_INDENT, indent_end = "")
    prefix = indent_str^indent * indent_end
    return join(
        map(split(s, "\n")) do line
            if headers > 0
                # a header might already be indented, keep that indent in front of the header
                m = match(r"^(\s*)(#+.*)$", line)
                isnothing(m) || (line = "$(m[1])$("#"^headers)$(m[2])")
            end
            return isempty(line) ? rstrip(prefix) : "$(prefix)$(line)"
        end,
        "\n",
    )
end

# in general, ignore printing the objective by default in tuples on REPL
function show(io::IO, t::Tuple{<:AbstractManifoldObjective, <:AbstractManoptSolverState})
    return show(io, t[2])
end
# on repl
function Base.show(io::IO, ::MIME"text/plain", t::Tuple{<:AbstractManifoldObjective, <:AbstractManoptSolverState})
    multiline = get(io, :multiline, true)
    return multiline ? status_summary(io, t[2]) : show(io, t[2])
end

# for decorated ones, default: pass down
function show(
        io::IO, t::Tuple{<:AbstractDecoratedManifoldObjective, <:AbstractManoptSolverState}
    )
    return show(io, t[2])
end
# for decorated ones, default: as both on status summary but first state then objective to print e.g. cache last
function show(
        io::IO, ::MIME"text/plain", t::Tuple{<:AbstractDecoratedManifoldObjective, <:AbstractManoptSolverState}
    )
    multiline = get(io, :multiline, true)
    multiline ? status_summary(io, t[2]) : show(io, t[2])
    return multiline ? status_summary(io, t[1]) : show(io, t[1])
end
