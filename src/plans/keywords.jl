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
        join(kw.accepted, ", ")
    end
    ds = if (length(kw.deprecated) == 0)
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

#
#
# A Macro and a default (with docs) for accepted_keywords, that a function directly accepts.

"""
A macro that creates a unified keywords function from multiple functions/structs.
Takes 2 or more functions/structs and creates a function `keywords(f)` that returns
the union of all allowed_keywords tuples, keeping the name of the first argument.

Usage:
@unified_keywords func1 func2 func3 ...
@combine_keywords struct1 struct2 func1 ...
"""
macro combine_keywords(funcs...)
    if length(funcs) < 2
        error("@combine_keywords requires at least 2 functions/structs")
    end
    first_func = funcs[1]
    # Create the function body that unions all keyword information
    function_body = quote
        funcs_list = [$(esc.(funcs)...)]
        # Start with the first function's keywords - using only the direct ones
        combined_keywords = copy(Manopt.direct_keywords(funcs_list[1]))
        # Add all keywords from all functions and append these
        for func in funcs_list[2:end]
            # use the recursive one for the others and append that to keywords
            union!(combined_keywords, Manopt.accepted_keywords(func))
        end
        return combined_keywords
    end

    return quote
        # For both the function and a struct, we want to dispatch on its type:
        if isa($(esc(first_func)), Type) # For struct constructors, create dispatch for the type
            function Manopt.accepted_keywords(::Type{$(esc(first_func))})
                return $(function_body)
            end
        else
            # For regular functions, create dispatch for the function
            function Manopt.accepted_keywords(::typeof($(esc(first_func))))
                return $(function_body)
            end
        end
    end
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
accepted_keywords(T) = direct_keywords(T)

"""
A macro that extracts keyword arguments from a function definition and creates
a new dispatch that returns the keyword names and deprecated keywords as a NamedTuple.

Usage:
@extract_keywords function_definition
@extract_keywords function_definition deprecated_set
"""
macro extract_keywords(func_def, deprecated_set=:(Set{Symbol}()))
    # TODO allow inline functions as well?
    if func_def.head != :function
        error("@extract_keywords can only be applied to function definitions")
    end
    # Extract function signature and name
    func_signature = func_def.args[1]
    func_name = func_signature.args[1]
    # Find keyword arguments - they're in any :parameters expression
    keyword_symbols = Symbol[]
    for arg in func_signature.args
        if isa(arg, Expr) && arg.head == :parameters
            for kwarg in arg.args
                # Extract the keyword name (handle both "name" and "name=default")
                #TODO handle the case name::T=default
                if isa(kwarg, Symbol)
                    push!(keyword_symbols, kwarg)
                elseif isa(kwarg, Expr) && kwarg.head == :kw # both kw cases
                    if isa(kwarg.args[1], Symbol) # form b = 1
                        push!(keyword_symbols, kwarg.args[1])
                    else # We also have a type c::T = 1
                        push!(keyword_symbols, kwarg.args[1].args[1])
                    end
                    # ignore others, e.g. kwargs...
                end
            end
        end
    end
    # Create the keyword set
    keyword_set = Set(keyword_symbols)

    # Generate the original function and the dispatch
    result = quote
        # Original function
        $(esc(func_def))
        # For both the function and a struct, we want to dispatch on its type:
        if isa($(esc(func_name)), Type) # For struct constructors, create dispatch for the type
            function Manopt.direct_keywords(::Type{$(esc(func_name))})
                return Keywords($(keyword_set), $(deprecated_set); in=$(esc(func_name)))
            end
        else # For regular functions, create dispatch for the function
            function Manopt.direct_keywords(::typeof($(esc(func_name))))
                return Keywords($(keyword_set), $(deprecated_set); in=$(esc(func_name)))
            end
        end
    end
    return result
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
function direct_keywords(s)
    return Keywords(; in=s)
end

function pretty_string_keywords(s::Set{Symbol})
    (length(s) == 0) && return ""
    return """
    * $(join(s, "\n* "))
    """
end
