"""
    Keywords

A small internal struct to represent a set of keywords,

# Fields

* `accepted` a `Set` of symbols of keywords a certain function accepts
* `deprecated` a `Set` of symbols of keywords a certain function has deprecated
* `origins` a dictionary that specifies for every keyword the function it is passed to.
  this usually should point to the function it is _directly_ passed to.

# Constructor

    Keywords(accepted=Set{Symbol}(), deprecated=Set{Symbol}(); type::Type=Nothing)

Generate a Keywords wrapper, where both default to being the empty set.
For pretty printing you can provide a type they belong to.
"""
struct Keywords{I}
    accepted::Set{Symbol}
    deprecated::Set{Symbol}
    origins::Dict{Symbol,Union{T,Vector{T}} where {T<:Union{Function,Type}}}
end
function Keywords(
    accepted::Set{Symbol}=Set{Symbol}(),
    deprecated::Set{Symbol}=Set{Symbol}();
    in=nothing,
    origins=nothing,
)
    if !isnothing(in)
        _origins = Dict{Symbol,Any}()
        for kw in accepted
            _origins[kw] = in
        end
        for kw in deprecated
            _origins[kw] = in
        end
    else
        _origins = Dict{Symbol,Union{<:Function,<:Type}}()
    end
    return Keywords{in}(accepted, deprecated, _origins)
end
function copy(kw::Keywords{I}) where {I}
    return Keywords{I}(copy(kw.accepted), copy(kw.deprecated), copy(kw.origins))
end

"""
    union!(kw::Keywords, kw2::Keywords)

Append the [`Keywords`](@ref) `kw2` to `kw`, i.e. union the accepted and deprecated keywords,
as well as their origins, but keep first parameter of `kw`.
Also their origin takes precedence.
"""
function Base.union!(kw::Keywords{I}, kw2::Keywords) where {I}
    union!(kw.accepted, kw2.accepted)
    union!(kw.deprecated, kw2.deprecated)
    for (k, v) in kw2.origins
        if !haskey(kw.origins, k)
            kw.origins[k] = v isa Vector ? [I, v...] : [I, v]
        end
    end
    return kw
end

function Base.show(io::IO, ::Keywords{Nothing})
    return print(io, "Keywords()")
end
function Base.show(io::IO, kw::Keywords{I}) where {I}
    if I === Nothing && length(kw.accepted) == 0 && length(kw.deprecated) == 0
        return print(io, "Keywords()")
    end
    as = if (length(kw.accepted) == 0)
        "none"
    else
        ast = ""
        for kwn in sort!(collect(kw.accepted); by=s -> lowercase(String(s)))
            if !startswith(string(kwn), "_")
                astn = "\n  * $(kwn)"
                if haskey(kw.origins, kwn) && kw.origins[kwn] isa Vector
                    pass_on = last(kw.origins[kwn])
                    if !("$(I)!" == "$(pass_on)")
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
    dt = isnothing(I) ? "A set of Keywords" : "Keywords for $I:"
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

Return a set of keywords, see [`Keywords`](@ref), a certain element of `Manopt.jl`
accepts when constructed.

this also includes keywords that are passed on to internal structures.
"""
function accepted_keywords(f)
    kw = direct_keywords(f)
    for g in calls_with_kwargs(f)
        kw2 = accepted_keywords(g)
        union!(kw, kw2)
    end
    return kw
end

calls_with_kwargs(f) = ()

"""
    direct_keywords(problem)
    direct_keywords(objective)
    direct_keywords(solver)
    direct_keywords(stepsize)

Return a set of keywords a function directly would work with,
i.e. that are directly defined in one of its dispatch variants, including
a mutating variant.

For example `direct_keywords(gradient_descent)` and `direct_keywords(gradient_descent!)`
should return the same set [`Keywords!`](@ref).
"""
function direct_keywords(f)
    methods_f = methods(f)
    s = Set{Symbol}()
    for m in methods_f
        for fkw in filter(
            x -> !(x == Symbol("kwargs...") || startswith("$(x)", "_")), Base.kwarg_decl(m)
        )
            push!(s, fkw)
        end
    end
    d = deprecated_keywords(s)
    setdiff!(s, d)
    return Keywords(s, d; in=f)
end

deprecated_keywords(s) = Set{Symbol}()

function pretty_string_keywords(s::Set{Symbol})
    (length(s) == 0) && return ""
    return """
    * $(join(s, "\n* "))
    """
end
