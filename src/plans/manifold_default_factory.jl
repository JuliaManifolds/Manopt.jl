"""
    ManifoldDefaultsFactory{M,T,A,K}

A generic factory to postpone the instantiation of certain types from within $(_link(:Manopt)),
in order to be able to adapt it to defaults from different manifolds and/or postpone the
decision on which manifold to use to a later point

For now this is established for

* [`DirectionUpdateRule`](@ref)s
* [`Stepsize`](@ref)
* [`StoppingCriterion`](@ref)

This factory stores necessary and optional parameters as well as
keyword arguments provided by the user to later produce the type this factory is for.

Besides a manifold as a fallback, the factory can also be used for the (maybe simpler)
types from the list of types that do not require the manifold.

# Fields

* `M::Union{Nothing,AbstractManifold}`:  provide a manifold for defaults
* `args::A`:                             arguments (`args...`) that are passed to the type constructor
* `kwargs::K`:                           keyword arguments (`kwargs...`) that are passed to the type constructor
* `constructor_requires_manifold::Bool`: indicate whether the type constructor requires the manifold or not


# Constructor

    ManifoldDefaultsFactory(T, args...; kwargs...)
    ManifoldDefaultsFactory(T, M, args...; kwargs...)


# Input

* `T` a subtype of types listed above that this factory is to produce
* `M` (optional) a manifold used for the defaults in case no manifold is provided.
* `args...` arguments to pass to the constructor of `T`
* `kwargs...` keyword arguments to pass (overwrite) when constructing `T`.

# Keyword arguments

* `requires_manifold=true`: indicate whether the type constructor this factory wraps
  requires the manifold as first argument or not.

All other keyword arguments are internally stored to be used in the type constructor

as well as arguments and keyword arguments for the update rule.

# see also

[`_produce_type`](@ref)
"""
struct ManifoldDefaultsFactory{T, TM <: Union{<:AbstractManifold, Nothing}, A, K}
    M::TM
    args::A
    kwargs::K
    constructor_requires_manifold::Bool
end
function ManifoldDefaultsFactory(
        T::Type, M::TM, args...; requires_manifold = true, kwargs...
    ) where {TM <: AbstractManifold}
    return ManifoldDefaultsFactory{T, TM, typeof(args), typeof(kwargs)}(
        M, args, kwargs, requires_manifold
    )
end
function ManifoldDefaultsFactory(T::Type, args...; requires_manifold = true, kwargs...)
    return ManifoldDefaultsFactory{T, Nothing, typeof(args), typeof(kwargs)}(
        nothing, args, kwargs, requires_manifold
    )
end
function (mdf::ManifoldDefaultsFactory{T})(M::AbstractManifold) where {T}
    if mdf.constructor_requires_manifold
        return T(M, mdf.args...; mdf.kwargs...)
    else
        return T(mdf.args...; mdf.kwargs...)
    end
end
function (mdf::ManifoldDefaultsFactory{T, <:AbstractManifold})() where {T}
    if mdf.constructor_requires_manifold
        return T(mdf.M, mdf.args...; mdf.kwargs...)
    else
        return T(mdf.args...; mdf.kwargs...)
    end
end
function (mdf::ManifoldDefaultsFactory{T, Nothing})() where {T}
    (!mdf.constructor_requires_manifold) && (return T(mdf.args...; mdf.kwargs...))
    throw(MethodError(T, mdf.args))
end
"""
    _produce_type(t::T, M::AbstractManifold)
    _produce_type(t::ManifoldDefaultsFactory{T}, M::AbstractManifold)

Use the [`ManifoldDefaultsFactory`](@ref)`{T}` to produce an instance of type `T`.
This acts transparent in the way that if you provide an instance `t::T` already, this will
just be returned.
"""
_produce_type(t, M::AbstractManifold) = t
_produce_type(t::ManifoldDefaultsFactory, M::AbstractManifold) = t(M)

function show(io::IO, mdf::ManifoldDefaultsFactory{T, M}) where {T, M}
    rm = mdf.constructor_requires_manifold
    if M === Nothing
        mline = "without a default manifold"
    else
        mline = "Default manifold: $(mdf.M)"
        (!rm) && (mline = "$mline and the constructor does also not require a manifold.")
    end
    ar_s = length(mdf.args) == 0 ? " none" : "\n$(join(["  * $s" for s in mdf.args], "\n"))"
    kw_s = if length(mdf.kwargs) == 0
        " none"
    else
        "\n$(join(["  * $(s.first)=$(repr(s.second))" for s in mdf.kwargs], "\n"))"
    end
    s = """
    ManifoldDefaultsFactory($T)
    * $mline

    * Arguments:$(ar_s)

    * Keyword arguments:$(kw_s)
    """
    return print(io, s)
end
