"""
    set_parameter!(f, element::Symbol, args...)

For any `f` and a `Symbol` `e`, dispatch on its value so by default, to
set some `args...` in `f` or one of uts sub elements.
"""
function set_parameter!(f, e::Symbol, args...)
    return set_parameter!(f, Val(e), args...)
end
function set_parameter!(f, args...)
    return f
end

"""
    get_parameter(f, element::Symbol, args...)

Access arbitrary parameters from `f` addressed by a symbol `element`.

For any `f` and a `Symbol` `e` dispatch on its value by default, to
get some element from `f` potentially further qualified by `args...`.

This functions returns `nothing` if `f` does not have the property `element`
"""
function get_parameter(f, e::Symbol, args...)
    return get_parameter(f, Val(e), args...)
end
get_parameter(f, args...) = nothing

"""
    get_parameter(element::Symbol; default=nothing)

Access global [`Manopt`](@ref) parameters addressed by a symbol `element`.
This first dispatches on the value of `element`.

If the value is not set, `default` is returned.

The parameters are queried from the global settings using [`Preferences.jl`](https://github.com/JuliaPackaging/Preferences.jl),
so they are persistent within your activated Environment, see also [`set_parameter!`](@ref).

## Currently used settings

`:Mode`
the mode can be set to `"Tutorial"` to get several hints especially in scenarios, where
the optimisation on manifolds is different from the usual “experience” in
(classical, Euclidean) optimization.
Any other value has the same effect as not setting it.

`:KeywordsErrorMode`
specify how to handle the case when unknown keywords are passed to a solver.
Since solvers often pass their keywords on to internal structures, to e.g.
decorate the objective or the state, checking keywords has its own method in `Manopt.jl`.
This parameter specifies how to handle the case where unknown keywords are handled.
* `"none"` does not report and the keyword gets just ignored
* `"warn"` issues a warning (default)
* `"error"` throw a [`ManoptKeywordError`](@ref)
all other symbol values are treated the same as `:none`.
"""
function get_parameter(e::Symbol, args...; default = get_parameter(Val(e), Val(:default)))
    return @load_preference("$(e)", default)
end
function get_parameter(
        e::Symbol, ::Symbol, args...; default = get_parameter(Val(e), Val(:default))
    )
    return @load_preference("$(e)", default)
end
# Handle empty defaults
get_parameter(::Symbol, ::Val{:default}) = nothing
get_parameter(::Val{:Mode}, ::Val{:default}) = nothing
get_parameter(::Val{:KeywordsErrorMode}, ::Val{:default}) = "warn"
"""
    set_parameter!(element::Symbol, value::Union{String,Bool,<:Number})

Set global [`Manopt`](@ref) parameters addressed by a symbol `element`.
W
This first dispatches on the value of `element`.

The parameters are stored to the global settings using [`Preferences.jl`](https://github.com/JuliaPackaging/Preferences.jl).

Passing a `value` of `""` deletes the corresponding entry from the preferences.
Whenever the `LocalPreferences.toml` is modified, this is also issued as an `@info`.
"""
function set_parameter!(e::Symbol, value::Union{String, Bool, <:Number})
    return if length(value) == 0
        @delete_preferences!(string(e))
        v = get_parameter(e, Val(:default))
        default = isnothing(v) ? "" : ((v isa String) ? " \"$v\"" : " ($v)")
        @info("Resetting the `Manopt.jl` parameter :$(e) to default$(default).")
    else
        @set_preferences!("$(e)" => value)
        @info("Setting the `Manopt.jl` parameter :$(e) to $value.")
    end
end
