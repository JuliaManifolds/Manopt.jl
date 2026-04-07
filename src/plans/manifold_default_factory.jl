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
    constructor_requires_point::Bool
end
function ManifoldDefaultsFactory(
        T::Type, M::TM, args...; requires_manifold = true, requires_point = false, kwargs...
    ) where {TM <: AbstractManifold}
    return ManifoldDefaultsFactory{T, TM, typeof(args), typeof(kwargs)}(
        M, args, kwargs, requires_manifold, requires_point
    )
end
function ManifoldDefaultsFactory(T::Type, args...; requires_manifold = true, requires_point = false, kwargs...)
    return ManifoldDefaultsFactory{T, Nothing, typeof(args), typeof(kwargs)}(
        nothing, args, kwargs, requires_manifold, requires_point
    )
end
function (mdf::ManifoldDefaultsFactory{T})(M::AbstractManifold, p) where {T}
    return if mdf.constructor_requires_manifold
        if mdf.constructor_requires_point
            return T(M, p, mdf.args...; mdf.kwargs...)
        else
            return T(M, mdf.args...; mdf.kwargs...)
        end
    else
        if mdf.constructor_requires_point
            return T(p, mdf.args...; mdf.kwargs...)
        else
            return T(mdf.args...; mdf.kwargs...)
        end
    end
end
function (mdf::ManifoldDefaultsFactory{T})(M::AbstractManifold) where {T}
    return if mdf.constructor_requires_manifold
        if mdf.constructor_requires_point
            return T(M, rand(M), mdf.args...; mdf.kwargs...)
        else
            return T(M, mdf.args...; mdf.kwargs...)
        end
    else
        if mdf.constructor_requires_point
            return T(rand(mdf.M), mdf.args...; mdf.kwargs...)
        else
            return T(mdf.args...; mdf.kwargs...)
        end
    end
end
function (mdf::ManifoldDefaultsFactory{T, <:AbstractManifold})() where {T}
    return if mdf.constructor_requires_manifold
        if mdf.constructor_requires_point
            return T(mdf.M, rand(mdf.M), mdf.args...; mdf.kwargs...)
        else
            return T(mdf.M, mdf.args...; mdf.kwargs...)
        end
    else
        if mdf.constructor_requires_point
            return T(rand(mdf.M), mdf.args...; mdf.kwargs...)
        else
            return T(mdf.args...; mdf.kwargs...)
        end
    end
end
function (mdf::ManifoldDefaultsFactory{T, Nothing})() where {T}
    (!mdf.constructor_requires_manifold) && (return T(mdf.args...; mdf.kwargs...))
    throw(MethodError(T, mdf.args))
end
"""
    _produce_type(t::T, M::AbstractManifold)
    _produce_type(t::ManifoldDefaultsFactory{T}, M::AbstractManifold)
    _produce_type(t::ManifoldDefaultsFactory{T}, M::AbstractManifold, p)

Use the [`ManifoldDefaultsFactory`](@ref)`{T}` to produce an instance of type `T`.
This acts transparent in the way that if you provide an instance `t::T` already, this will
just be returned.

If a point `p` on manifold `M` is provided, it is passed to the constructor `t` as a
template for allocating points. It is no supposed to be modified by the constructor or
stored in the produced object.
"""
_produce_type(t, M::AbstractManifold) = t
_produce_type(t, M::AbstractManifold, p) = t
_produce_type(t::ManifoldDefaultsFactory, M::AbstractManifold) = t(M)
_produce_type(t::ManifoldDefaultsFactory, M::AbstractManifold, p) = t(M, p)

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
