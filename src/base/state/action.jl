@doc """
    AbstractStateAction

a common `Type` for `AbstractStateActions` that might be triggered in decorators,
for example within the [`DebugSolverState`](@ref) or within the [`RecordSolverState`](@ref).
"""
abstract type AbstractStateAction end

status_summary(asa::AbstractStateAction; context::Symbol = :default) = repr(asa)
function Base.show(io::IO, ::MIME"text/plain", asa::AbstractStateAction)
    multiline = get(io, :multiline, true)
    return multiline ? status_summary(io, asa) : show(io, asa)
end

"""
    struct PointStorageKey{key} end

Refer to point storage of [`StoreStateAction`](@ref) in `get_storage` and `has_storage`
functions
"""
struct PointStorageKey{key} end
PointStorageKey(key::Symbol) = PointStorageKey{key}()

"""
    struct VectorStorageKey{key} end

Refer to tangent storage of [`StoreStateAction`](@ref) in `get_storage` and `has_storage`
functions
"""
struct VectorStorageKey{key} end
VectorStorageKey(key::Symbol) = VectorStorageKey{key}()

#
# Common Actions for decorated AbstractManoptSolverState
#

mutable struct StorageRef{T}
    x::T
end

function Base.copyto!(sr::StorageRef, new_x)
    sr.x = copy(new_x)
    return sr
end

"""
    _storage_copy_point(M::AbstractManifold, p)

Make a copy of point `p` from manifold `M` for storage in [`StoreStateAction`](@ref).
"""
_storage_copy_point(M::AbstractManifold, p) = copy(M, p)
_storage_copy_point(::AbstractManifold, p::Number) = StorageRef(p)

"""
    _storage_copy_vector(M::AbstractManifold, X)

Make a copy of tangent vector `X` from manifold `M` for storage in [`StoreStateAction`](@ref).
"""
_storage_copy_vector(M::AbstractManifold, X) = copy(M, X)
_storage_copy_vector(::AbstractManifold, X::Number) = StorageRef(X)

@doc """
    StoreStateAction <: AbstractStateAction

internal storage for [`AbstractStateAction`](@ref)s to store a tuple of fields from an
[`AbstractManoptSolverState`](@ref)s

This functor possesses the usual interface of functions called during an
iteration and acts on `(p, s, k)`, where `p` is a [`AbstractManoptProblem`](@ref),
`s` is an [`AbstractManoptSolverState`](@ref) and `k` is the current iteration.

# Fields

* `values`:        a dictionary to store interim values based on certain `Symbols`
* `keys`:          a `Vector` of `Symbols` to refer to fields of `AbstractManoptSolverState`
* `point_values`:  a `NamedTuple` of mutable values of points on a manifold to be stored in
  `StoreStateAction`. Manifold is later determined by `AbstractManoptProblem` passed
  to `update_storage!`.
* `point_init`:    a `NamedTuple` of boolean values indicating whether a point in
  `point_values` with matching key has been already initialized to a value. When it is
  false, it corresponds to a general value not being stored for the key present in the
  vector `keys`.
* `vector_values`: a `NamedTuple` of mutable values of tangent vectors on a manifold to be
  stored in `StoreStateAction`. Manifold is later determined by `AbstractManoptProblem`
  passed to `update_storage!`. It is not specified at which point the vectors are tangent
  but for storage it should not matter.
* `vector_init`:   a `NamedTuple` of boolean values indicating whether a tangent vector in
  `vector_values`: with matching key has been already initialized to a value. When it is
  false, it corresponds to a general value not being stored for the key present in the
  vector `keys`.
* `once`:          whether to update the internal values only once per iteration
* `lastStored`:    last iterate, where this `AbstractStateAction` was called (to determine `once`)

To handle the general storage, use `get_storage` and `has_storage` with keys as `Symbol`s.
For the point storage use `PointStorageKey`. For tangent vector storage use
`VectorStorageKey`. Point and tangent storage have been optimized to be more efficient.

# Constructors

    StoreStateAction(s::Vector{Symbol})

This is equivalent as providing `s` to the keyword `store_fields`, just that here, no manifold
is necessity for the construction.

    StoreStateAction(M)

## Keyword arguments

* `store_fields` (`Symbol[]`)
* `store_points` (`Symbol[]`)
* `store_vectors` (`Symbol[]`)

as vectors of symbols each referring to fields of the state (lower case symbols)
or semantic ones (upper case).

* `p_init` (`rand(M)`) but making sure this is not a number but a (mutable) array
* `X_init` (`zero_vector(M, p_init)`)

are used to initialize the point and vector storage, change these if you use other
types (than the default) for your points/vectors on `M`.

* `once` (`true`) whether to update internal storage only once per iteration or on every update call
"""
mutable struct StoreStateAction{
        TPS_asserts, TXS_assert, TPS <: NamedTuple, TXS <: NamedTuple, TPI <: NamedTuple, TTI <: NamedTuple,
    } <: AbstractStateAction
    values::Dict{Symbol, Any}
    keys::Vector{Symbol} # for values
    point_values::TPS
    vector_values::TXS
    point_init::TPI
    vector_init::TTI
    once::Bool
    last_stored::Int
    function StoreStateAction(
            general_keys::Vector{Symbol} = Symbol[],
            point_values::NamedTuple = NamedTuple(),
            vector_values::NamedTuple = NamedTuple(),
            once::Bool = true;
            M::AbstractManifold = DefaultManifold(),
        )
        point_init = NamedTuple{keys(point_values)}(map(u -> false, keys(point_values)))
        vector_init = NamedTuple{keys(vector_values)}(map(u -> false, keys(vector_values)))
        point_values_copy = NamedTuple{keys(point_values)}(
            map(u -> _storage_copy_point(M, point_values[u]), keys(point_values))
        )
        vector_values_copy = NamedTuple{keys(vector_values)}(
            map(u -> _storage_copy_vector(M, vector_values[u]), keys(vector_values))
        )
        return new{
            typeof(point_values),
            typeof(vector_values),
            typeof(point_values_copy),
            typeof(vector_values_copy),
            typeof(point_init),
            typeof(vector_init),
        }(
            Dict{Symbol, Any}(),
            general_keys,
            point_values_copy,
            vector_values_copy,
            point_init,
            vector_init,
            once,
            -1,
        )
    end
end
@noinline function StoreStateAction(
        M::AbstractManifold;
        store_fields::Vector{Symbol} = Symbol[],
        store_points::Union{Type{<:Tuple}, Vector{Symbol}} = Tuple{},
        store_vectors::Union{Type{<:Tuple}, Vector{Symbol}} = Tuple{},
        p_init = maybe_wrap_variable(rand(M)),
        X_init = zero_vector(M, p_init),
        once = true,
    )
    TPS_tuple = _store_to_tuple(store_points)
    TTS_tuple = _store_to_tuple(store_vectors)
    point_values = NamedTuple{TPS_tuple}(map(_ -> p_init, TPS_tuple))
    vector_values = NamedTuple{TTS_tuple}(map(_ -> X_init, TTS_tuple))
    return StoreStateAction(store_fields, point_values, vector_values, once; M = M)
end

_store_to_tuple(store::Type{<:Tuple}) = Tuple(store.parameters)
_store_to_tuple(store::Vector{Symbol}) = tuple(store...)

@generated function extract_type_from_namedtuple(::Type{nt}, ::Val{key}) where {nt, key}
    for i in 1:length(nt.parameters[1])
        if nt.parameters[1][i] === key
            return nt.parameters[2].parameters[i]
        end
    end
    return Any
end

function _store_point_assert_type(
        ::StoreStateAction{TPS_asserts, TXS_assert}, key::Val
    ) where {TPS_asserts, TXS_assert}
    return extract_type_from_namedtuple(TPS_asserts, key)
end

function _store_vector_assert_type(
        ::StoreStateAction{TPS_asserts, TXS_assert}, key::Val
    ) where {TPS_asserts, TXS_assert}
    return extract_type_from_namedtuple(TXS_assert, key)
end

function (a::StoreStateAction)(
        amp::AbstractManoptProblem, s::AbstractManoptSolverState, k::Int
    )
    (!a.once || a.last_stored != k) && (update_storage!(a, amp, s))
    a.last_stored = k
    return a
end

"""
    get_storage(a::AbstractStateAction, key::Symbol)

Return the internal value of the [`AbstractStateAction`](@ref) `a` at the
`Symbol` `key`. Returns `nothing` if the key does not exist
"""
function get_storage(a::AbstractStateAction, key::Symbol)
    return get(a.values, key, nothing)
end
"""
    get_storage(a::AbstractStateAction, ::PointStorageKey{key}) where {key}

Return the internal value of the [`AbstractStateAction`](@ref) `a` at the
`Symbol` `key` that represents a point.
"""
@inline function get_storage(a::AbstractStateAction, ::PointStorageKey{key}) where {key}
    if haskey(a.point_values, key)
        val = a.point_values[key]
        if val isa StorageRef
            return val.x
        else
            return val
        end
    else
        return get_storage(a, key)
    end
end

"""
    get_storage(a::AbstractStateAction, ::VectorStorageKey{key}) where {key}

Return the internal value of the [`AbstractStateAction`](@ref) `a` at the
`Symbol` `key` that represents a vector.
"""
@inline function get_storage(a::AbstractStateAction, ::VectorStorageKey{key}) where {key}
    if haskey(a.vector_values, key)
        val = a.vector_values[key]
        if val isa StorageRef
            return val.x
        else
            return val
        end
    else
        return get_storage(a, key)
    end
end

"""
    has_storage(a::AbstractStateAction, key::Symbol)

Return whether the [`AbstractStateAction`](@ref) `a` has a value stored at the
`Symbol` `key`.
"""
has_storage(a::AbstractStateAction, key::Symbol) = haskey(a.values, key)

"""
    has_storage(a::AbstractStateAction, ::PointStorageKey{key}) where {key}

Return whether the [`AbstractStateAction`](@ref) `a` has a point value stored at the
`Symbol` `key`.
"""
function has_storage(a::AbstractStateAction, ::PointStorageKey{key}) where {key}
    if haskey(a.point_init, key)
        return a.point_init[key]
    else
        return has_storage(a, key)
    end
end

"""
    has_storage(a::AbstractStateAction, ::VectorStorageKey{key}) where {key}

Return whether the [`AbstractStateAction`](@ref) `a` has a point value stored at the
`Symbol` `key`.
"""
function has_storage(a::AbstractStateAction, ::VectorStorageKey{key}) where {key}
    if haskey(a.vector_init, key)
        return a.vector_init[key]
    else
        return has_storage(a, key)
    end
end

"""
    update_storage!(a::AbstractStateAction, amp::AbstractManoptProblem, s::AbstractManoptSolverState)

Update the [`AbstractStateAction`](@ref) `a` internal values to the ones given on
the [`AbstractManoptSolverState`](@ref) `s`.
Optimized using the information from `amp`
"""
function update_storage!(
        a::AbstractStateAction, amp::AbstractManoptProblem, s::AbstractManoptSolverState
    )
    for key in a.keys
        if key === :Iterate
            a.values[key] = deepcopy(get_iterate(s))
        elseif key === :Gradient
            a.values[key] = deepcopy(get_gradient(s))
        elseif key === :Population
            swarm = get_parameter(s, key)
            if !isnothing(swarm)
                a.values[key] = deepcopy.(swarm)
            end
        else
            a.values[key] = deepcopy(getproperty(s, key))
        end
    end

    M = get_manifold(amp)
    @inline function update_points(key)
        return if key === :Iterate
            copyto!(M, a.point_values[key], get_iterate(s))
        else
            copyto!(
                M,
                a.point_values[key],
                getproperty(s, key)::_store_point_assert_type(a, Val(key)),
            )
        end
    end
    map(update_points, keys(a.point_values))
    a.point_init = NamedTuple{keys(a.point_values)}(map(u -> true, keys(a.point_values)))

    @inline function update_vector(key)
        return if key === :Gradient
            copyto!(M, a.vector_values[key], get_gradient(s))
        else
            copyto!(
                M,
                a.vector_values[key],
                getproperty(s, key)::_store_vector_assert_type(a, Val(key)),
            )
        end
    end
    map(update_vector, keys(a.vector_values))

    a.vector_init = NamedTuple{keys(a.vector_values)}(map(u -> true, keys(a.vector_values)))

    return a.keys
end

"""
    update_storage!(a::AbstractStateAction, d::Dict{Symbol,<:Any})

Update the [`AbstractStateAction`](@ref) `a` internal values to the ones given in
the dictionary `d`. The values are merged, where the values from `d` are preferred.
"""
function update_storage!(a::AbstractStateAction, d::Dict{Symbol, <:Any})
    merge!(a.values, d)
    # update keys
    return a.keys = collect(keys(a.values))
end
