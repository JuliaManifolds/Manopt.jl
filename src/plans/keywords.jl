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
        for kwn in sort!(collect(kw.accepted); by = s->lowercase(String(s)))
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

#
#
# A Macro and a default (with docs) for accepted_keywords, that a function directly accepts.

"""
@combine_keywords(f, g, ...)

A macro that defines a new method for [`accepted_keywords`](@ref) for `f`.
It specifies that `f` accepts all keywords it defines itself and all that it passes
on to the other specified functions or constructors.

The resulting [`Keywords`](@ref) will also include the origin of a keyword,
or in other words where `f` passes a certain accepted keyword to.

# Examples

```julia
@combine_keywords f, g
@combine_keywords f, g, h
````
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

# TODO We need a slightly improved variant that can “continue” an existing one,
# since e.g. for `gradient_descent`there are two definitions (nested) that defined keywords
#
# Idea: Take the full signature of the outer function not just its name,
# but define the name to collect all of the full signatures.
# since the signatures are complicated, maybe their string?
# for multiple definitions make sure not to define the function one twice then.
#
# Maybe we can dispatch on at least a few types: whether the second is a
# (a) Nothing
# (b) anything, e.g. a function
# (c) an AbstractManifoldObjective
# For ease of use, extend the macro here to use that in defining the function
# this will become the second keyword of the direct calls; the accepted ones union also all direct ones (but without modifying origin then)
#
# Make internal ones with prefix _ (line embedded_X and embedded_p in decorate_objective) and do not add them to the keywords listed
#
# Format idea
# @extract_keywords Type(=Nothing) function deprecated
# and direct_keywords (as well as accepted) takes that as a symbol to map to types,
"""
@extract_keywords(function_definition[, deprecated_set::Set{Symbol}=Set{Symbol}()])

A macro that extracts keyword arguments from a function definition.
These are used to define a new [`direct_keywords`](@ref) definition that returns
these keywords that are _directly_ provided in the definition of a function.

# Examples

```julia
@extract_keywords function f(args...; x=1, y=2) return x+y end
@extract_keywords f(;x=1,y=2) Set([:y,]) # marks keyword `y` as deprecated
```
"""
macro extract_keywords(func_def, deprecated_set=Set{Symbol}())
    if (func_def.head != :function) && (func_def.head != :(=))
        error(
            "@extract_keywords can only be applied to function definitions or inline functions",
        )
    end
    # Extract function signature and name
    func_signature = func_def.args[1]
    #@info dump(func_signature)
    # If we have a _where_ in the definition, cut that away and just keep the sig
    (func_signature.head == :where) && (func_signature = func_signature.args[1])
    func_name = func_signature.args[1]
    if isa(func_signature.args[2], Expr) && func_signature.args[2].head == :parameters
        # we have keyword arguments
        callargs = func_signature.args[3:end]
        kwargs_list = func_signature.args[2].args
    else
        callargs = func_signature.args[2:end]
        kwargs_list = []
    end
    # Find keyword arguments - they're in any :parameters expression
    # extend existing ones.
    keyword_symbols = Manopt.direct_keywords(func_name).accepted
    for kwarg in kwargs_list
        # Extract the keyword name (handle both "name" and "name=default")
        #TODO handle the case name::T=default
        if isa(kwarg, Symbol)
            push!(keyword_symbols, kwarg)
        elseif isa(kwarg, Expr) && kwarg.head == :kw # both kw cases
            if isa(kwarg.args[1], Symbol) # form b = 1
                push!(keyword_symbols, kwarg.args[1])
            else # We also have a type c::Int = 1
                push!(keyword_symbols, kwarg.args[1].args[1])
            end
            # ignore others, e.g. kwargs...
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
