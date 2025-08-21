"""
    Keywords

A small internal struct to represent a set of keywords,

# Fields

* `accepted`
* `deprecated`

# Constructor

    Keywords(accepted=Set{Symbol}(), deprecated=Set{Symbol}(); type::Type=Nothing)

Generate a Keywords wrapper, where both default to being the empty set.
For pretty printing you can provide a type they belong to.

    Keywords(nothing; type=Nothing)

A constructor to indicate that a certain function or element of type `T` is
not yet set up to work in this scenario.
"""
struct Keywords{I,A<:Union{Nothing,Set{Symbol}},D<:Union{Nothing,Set{Symbol}}}
    accepted::A
    deprecated::D
end
function Keywords(
    accepted::A, deprecated::D=Set{Symbol}(); in=nothing
) where {A<:Union{Nothing,Set{Symbol}},D<:Union{Nothing,Set{Symbol}}}
    return Keywords{in,A,D}(accepted, deprecated)
end
Keywords(::Nothing=nothing; in=Nothing) = Keywords{in,Nothing,Nothing}(nothing, nothing)

function Base.show(io::IO, ::Keywords{Nothing})
    return print(io, "Keywords()")
end

function Base.show(io::IO, kw::Keywords{I,A,B}) where {I,A,B}
    as = if (isnothing(kw.accepted) || length(kw.accepted) == 0)
        "none"
    else
        join(kw.accepted, ", ")
    end
    ds = if (isnothing(kw.deprecated) || length(kw.deprecated) == 0)
        "none"
    else
        join(kw.deprecated, ", ")
    end
    dt = isnothing(I) ? "A set of Keywords" : "Keywords for $I:"
    return print(
        io,
        """
$dt

accepted: $as
deprecated: $ds
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
function accepted_keywords(
    ::Union{Type{T},F}
) where {
    T<:Union{
        AbstractManoptProblem,
        AbstractManifoldObjective,
        AbstractManoptSolverState,
        Stepsize,
        StoppingCriterion,
    },
    F<:Function,
}
    return direct_keywords(T)
end

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
function direct_keywords(
    s::Union{Type{T},F}
) where {
    T<:Union{
        AbstractManoptProblem,
        AbstractManifoldObjective,
        AbstractManoptSolverState,
        Stepsize,
        StoppingCriterion,
    },
    F<:Function,
}
    return Keywords(; in=s)
end

function pretty_string_keywords(s::Set{Symbol})
    (length(s) == 0) && return ""
    return """
    * $(join(s, "\n* "))
    """
end
