"""
    Keywords

A small internal struct to represent a set of keywords.

The type parameter `I` stores the function the keywords belong to, or `nothing`
if they are not associated with a certain function.

# Fields

* `accepted::Set{Symbol}`: symbols of keywords a certain function accepts
* `deprecated::Set{Symbol}`: symbols of keywords a certain function has deprecated
* `origins::Dict{Symbol,Vector{Any}}`: a dictionary that specifies for every keyword the function it is passed to.
  This usually should point to the function it is _directly_ passed to.

# Constructor

    Keywords(
        accepted=Set{Symbol}(), deprecated=Set{Symbol}();
        from=nothing, origins=nothing)

Generate a [`Keywords`](@ref) wrapper, where both sets default to being empty.

## Keyword arguments

* `from=nothing`: the function the keywords (directly or indirectly) come from or are
  accepted in; it becomes the type parameter `I`. Use `nothing` to indicate that these
  are not associated with a certain function.
* `origins=nothing`: an existing origins dictionary to extend; only used when `from` is given.
"""
struct Keywords{I}
    accepted::Set{Symbol}
    deprecated::Set{Symbol}
    origins::Dict{Symbol, Vector{Any}}
end
function Keywords(
        accepted::Set{Symbol} = Set{Symbol}(),
        deprecated::Set{Symbol} = Set{Symbol}();
        from = nothing, origins::Union{Dict, Nothing} = nothing,
    )
    if !isnothing(from)
        _origins = isnothing(origins) ? Dict{Symbol, Vector{Any}}() : origins
        for kw in accepted
            _origins[kw] = [from]
        end
        for kw in deprecated
            _origins[kw] = [from]
        end
    else
        _origins = Dict{Symbol, Vector{Any}}()
    end
    return Keywords{from}(accepted, deprecated, _origins)
end
function copy(kw::Keywords{I}) where {I}
    return Keywords{I}(copy(kw.accepted), copy(kw.deprecated), copy(kw.origins))
end

"""
    ManoptKeywordError <: Exception

An error to indicate that a certain function received keywords it does not accept.

# Fields

* `f`: the function that received the keywords
* `kw::Keywords`: the keywords that were not accepted

# Constructor

    ManoptKeywordError(f, kw::Keywords)
"""
struct ManoptKeywordError{F, K <: Keywords} <: Exception
    f::F
    kw::K
end
function Base.showerror(io::IO, e::ManoptKeywordError)
    return print(io, keyword_error_string(e.f, e.kw))
end
function keyword_error_string(f, kw::Keywords; hint = true)
    io = IOBuffer()
    if length(kw.accepted) > 0
        print(io, "$(f) does not accept the keyword(s)\n\n  * ")
        print(io, join(sort!(collect(kw.accepted)), "\n  * "), "\n")
    end
    if length(kw.deprecated) > 0
        print(io, "\n$(f) accepts, but deprecates the keyword(s):\n  ")
        print(io, join(sort!(collect(kw.deprecated)), ", "), "\n  ")
        print(io, "\n")
    end
    if hint
        akw = accepted_keywords(f).accepted
        if length(akw) == 0
            print(io, "\n$f does not accept any keywords.")
        else
            print(io, "\nHint: $f does accept the following keywords:\n\n  ")
            print(io, join(sort!(collect(akw)), ", "))
            print(io, "\n")
        end
    end
    return String(take!(io))
end

"""
    add!(kw::Keywords, kw2::Keywords)

Append the [`Keywords`](@ref) `kw2` to `kw`, that is, union the accepted and deprecated keywords
as well as their origins, but keep the type parameter of `kw`.
Origins already present in `kw` take precedence.
"""
function add!(kw::Keywords{I}, kw2::Keywords) where {I}
    union!(kw.accepted, kw2.accepted)
    union!(kw.deprecated, kw2.deprecated)
    for (k, v) in kw2.origins
        if !haskey(kw.origins, k)
            kw.origins[k] = [I, v...]
        end
    end
    return kw
end

function Base.show(io::IO, kw::Keywords{I}) where {I}
    if I === nothing && length(kw.accepted) == 0 && length(kw.deprecated) == 0
        return print(io, "Keywords()")
    end
    as = if (length(kw.accepted) == 0)
        "none"
    else
        ast = ""
        for kwn in sort!(collect(kw.accepted); by = s -> lowercase(String(s)))
            if !startswith(string(kwn), "_")
                astn = "\n  * $(kwn)"
                if haskey(kw.origins, kwn) && kw.origins[kwn] isa Vector
                    pass_on = last(kw.origins[kwn])
                    if !((pass_on === I) || ("$(I)!" == "$(pass_on)"))
                        astn *= " (passed on to $pass_on)"
                    end
                end
                ast *= astn
            end
        end
        ast
    end
    ds = if (length(kw.deprecated) == 0)
        ""
    else
        "\ndeprecated $(join(kw.deprecated, ", "))"
    end
    dt = isnothing(I) ? "A set of Keywords:" : "Keywords for $I:"
    return print(
        io,
        """
        $dt

        accepted: $as$ds
        """,
    )
end

"""
    accepted_keywords(problem)
    accepted_keywords(objective)
    accepted_keywords(solver)
    accepted_keywords(stepsize)

Return the set of keywords, see [`Keywords`](@ref), that a certain element of `Manopt.jl`
accepts when constructed.

This function uses [`direct_keywords`](@ref) to find keywords a function directly accepts,
and [`calls_with_kwargs`](@ref) to find functions it passes keywords to, where they also
might be accepted.
In order for a nonmutating function `f` to accept the same keywords as its mutating variant `f!`,
one should set [`calls_with_kwargs`](@ref)`(f) = (f!,)`.

This also includes keywords that are passed on to internal structures, which are specified using
[`calls_with_kwargs`](@ref) as well.
"""
function accepted_keywords(f)
    kw = direct_keywords(f)
    for g in calls_with_kwargs(f)
        kw2 = accepted_keywords(g)
        add!(kw, kw2)
    end
    return kw
end

"""
    calls_with_kwargs(f)

Return a tuple of functions `f` calls and passes its `kwargs...` to.
"""
calls_with_kwargs(f) = ()

"""
    direct_keywords(problem)
    direct_keywords(objective)
    direct_keywords(solver)
    direct_keywords(stepsize)

Return a set of keywords a function would directly work with.
"""
function direct_keywords(f)
    methods_f = methods(f)
    s = Set{Symbol}()
    for m in methods_f
        for fkw in filter(
                x -> !(
                    x == Symbol("...") || x == Symbol("kwargs...") || startswith("$(x)", "_")
                ),
                Base.kwarg_decl(m),
            )
            push!(s, fkw)
        end
    end
    d = deprecated_keywords(s)
    setdiff!(s, d)
    return Keywords(s, d; from = f)
end

deprecated_keywords(s) = Set{Symbol}()

"""
    keywords_accepted(f, mode=Symbol(get_parameter(:KeywordsErrorMode)), kw::Keywords=accepted_keywords(f); kwargs...)

Given a function `f` and the [`Keywords`](@ref) `kw` it accepts, check whether `kwargs...` are
accepted by those keywords and warn if deprecated keywords are passed.

For keywords that are not accepted, the `mode` argument specifies how to report the result:
`:warn` issues a warning, `:error` throws a [`ManoptKeywordError`](@ref), and any other value
(for example `:none`) neither warns nor errors.
The default is taken from the `:KeywordsErrorMode` setting, see [`get_parameter`](@ref).
"""
function keywords_accepted(
        f, mode::Symbol = Symbol(get_parameter(:KeywordsErrorMode)), kw::Keywords = accepted_keywords(f);
        kwargs...
    )
    d = Set{Symbol}()
    a = Set{Symbol}()
    for (k, v) in kwargs
        # Deprecated?
        if k in kw.deprecated
            push!(d, k)
        end
        # Not accepted? Note that an empty set of accepted keywords means that none are.
        # A deprecated keyword is not in `accepted` either, but is reported as deprecated above.
        if (k ∉ kw.accepted) && (k ∉ kw.deprecated)
            push!(a, k)
        end
    end
    # Warn about deprecated keywords always – but only once:
    # on `:warn` with unaccepted keywords present, the block below already reports them.
    if length(d) > 0 && !((mode === :warn) && (length(a) > 0))
        @warn keyword_error_string(f, Keywords(Set{Symbol}(), d); hint = false)
    end
    if length(a) > 0
        error_kws = Keywords(a, d; from = f)
        (mode === :warn) && (@warn keyword_error_string(f, error_kws; hint = true))
        (mode === :error) && throw(ManoptKeywordError(f, error_kws))
        # else handle as :none and do not warn or error
    end
    return (length(a) == 0) && (length(d) == 0)
end
