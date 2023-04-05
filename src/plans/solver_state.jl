
@inline _extract_val(::Val{T}) where {T} = T

@doc raw"""
    AbstractManoptSolverState

A general super type for all solver states.

# Fields

The following fields are assumed to be default. If you use different ones,
provide the access functions accordingly

* `p` a point on a manifold with the current iterate
* `stop` a [`StoppingCriterion`](@ref).
"""
abstract type AbstractManoptSolverState end

@doc raw"""
    AbstractGradientSolverState <: AbstractManoptSolverState

A generic [`AbstractManoptSolverState`](@ref) type for gradient based options data.

It assumes that

* the iterate is stored in the field `p`
* the gradient at `p` is stored in `X`.

# see also
[`GradientDescentState`](@ref), [`StochasticGradientDescentState`](@ref), [`SubGradientMethodState`](@ref), [`QuasiNewtonState`](@ref).
"""
abstract type AbstractGradientSolverState <: AbstractManoptSolverState end

"""
    dispatch_state_decorator(s::AbstractManoptSolverState)

Indicate internally, whether an [`AbstractManoptSolverState`](@ref) `s` to be of decorating type, i.e.
it stores (encapsulates) state in itself, by default in the field `s.state`.

Decorators indicate this by returning `Val{true}` for further dispatch.

The default is `Val{false}`, i.e. by default an state is not decorated.
"""
dispatch_state_decorator(::AbstractManoptSolverState) = Val(false)

@doc raw"""
    get_message(du::AbstractManoptSolverState)

get a message (String) from e.g. performing a step computation.
This should return any message a sub-step might have issued
"""
function get_message(::AbstractManoptSolverState)
    return _get_message(s, dispatch_state_decorator(s))
end
_get_message(s::AbstractManoptSolverState, ::Val{true}) = _get_message(s.state)
#INtroduce a default that there is no message
_get_message(s::AbstractManoptSolverState, ::Val{false}) = ""

"""
    is_state_decorator(s::AbstractManoptSolverState)

Indicate, whether [`AbstractManoptSolverState`](@ref) `s` are of decorator type.
"""
function is_state_decorator(s::AbstractManoptSolverState)
    return _extract_val(dispatch_state_decorator(s))
end

@doc raw"""
    ReturnSolverState{O<:AbstractManoptSolverState} <: AbstractManoptSolverState

This internal type is used to indicate that the contained [`AbstractManoptSolverState`](@ref) `state`
should be returned at the end of a solver instead of the usual minimizer.

# See also

[`get_solver_result`](@ref)
"""
struct ReturnSolverState{S<:AbstractManoptSolverState} <: AbstractManoptSolverState
    state::S
end
status_summary(rst::ReturnSolverState) = status_summary(rst.state)
show(io::IO, rst::ReturnSolverState) = print(io, "ReturnSolverState($(rst.state))")
dispatch_state_decorator(::ReturnSolverState) = Val(true)

"""
    get_solver_return(O::AbstractManoptSolverState)

determine the result value of a call to a solver. By default this returns the same as [`get_solver_result`](@ref),
i.e. the last iterate or (approximate) minimizer.

    get_solver_return(O::ReturnSolverState)

return the internally stored state of the [`ReturnSolverState`](@ref) instead of the minimizer.
This means that when the state are decorated like this, the user still has to call [`get_solver_result`](@ref)
on the internal state separately.
"""
function get_solver_return(s::AbstractManoptSolverState)
    return _get_solver_return(s, dispatch_state_decorator(s))
end
_get_solver_return(s::AbstractManoptSolverState, ::Val{false}) = get_solver_result(s)
_get_solver_return(s::AbstractManoptSolverState, ::Val{true}) = get_solver_return(s.state)
get_solver_return(s::ReturnSolverState) = s.state

@doc raw"""
    get_state(s::AbstractManoptSolverState)

return the undecorated [`AbstractManoptSolverState`](@ref) of the (possibly) decorated `s`.
As long as your decorated state store the state within `s.state` and
the [`dispatch_state_decorator`](@ref) is set to `Val{true}`,
the internal state are extracted.
"""
get_state(s::AbstractManoptSolverState) = _get_state(s, dispatch_state_decorator(s))
_get_state(s::AbstractManoptSolverState, ::Val{false}) = s
_get_state(s::AbstractManoptSolverState, ::Val{true}) = get_state(s.state)

"""
    get_gradient(s::AbstractManoptSolverState)

return the (last stored) gradient within [`AbstractManoptSolverState`](@ref)` `s`.
By default also undecorates the state beforehand
"""
get_gradient(s::AbstractManoptSolverState) = _get_gradient(s, dispatch_state_decorator(s))
function _get_gradient(s::AbstractManoptSolverState, ::Val{false})
    return error("It seems that $s do not provide access to a gradient")
end
_get_gradient(s::AbstractManoptSolverState, ::Val{true}) = get_gradient(s.state)

"""
    set_gradient!(s::AbstractManoptSolverState, M::AbstractManifold, p, X)

set the gradient within an (possibly decorated) [`AbstractManoptSolverState`](@ref)
to some (start) value `X` in the tangent space at `p`.
"""
function set_gradient!(s::AbstractManoptSolverState, M, p, X)
    return _set_gradient!(s, M, p, X, dispatch_state_decorator(s))
end
function _set_gradient!(s::AbstractManoptSolverState, ::Any, ::Any, ::Any, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s do not provide (write) access to a gradient",
    )
end
function _set_gradient!(s::AbstractManoptSolverState, M, p, X, ::Val{true})
    return set_gradient!(s.state, M, p, X)
end

"""
    get_iterate(O::AbstractManoptSolverState)

return the (last stored) iterate within [`AbstractManoptSolverState`](@ref)` `s`.
By default also undecorates the state beforehand.
"""
get_iterate(s::AbstractManoptSolverState) = _get_iterate(s, dispatch_state_decorator(s))
function _get_iterate(s::AbstractManoptSolverState, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s do not provide access to an iterate"
    )
end
_get_iterate(s::AbstractManoptSolverState, ::Val{true}) = get_iterate(s.state)

"""
    set_iterate!(s::AbstractManoptSolverState, M::AbstractManifold, p)

set the iterate within an [`AbstractManoptSolverState`](@ref) to some (start) value `p`.
"""
function set_iterate!(s::AbstractManoptSolverState, M, p)
    return _set_iterate!(s, M, p, dispatch_state_decorator(s))
end
function _set_iterate!(s::AbstractManoptSolverState, ::Any, ::Any, ::Val{false})
    return error(
        "It seems the AbstractManoptSolverState $s do not provide (write) access to an iterate",
    )
end
_set_iterate!(s::AbstractManoptSolverState, M, p, ::Val{true}) = set_iterate!(s.state, M, p)

"""
    get_solver_result(ams::AbstractManoptSolverState)

Return the final result after all iterations that is stored within
the [`AbstractManoptSolverState`](@ref) `ams`, which was modified during the iterations.
"""
function get_solver_result(s::AbstractManoptSolverState)
    return get_solver_result(s, dispatch_state_decorator(s))
end
get_solver_result(s::AbstractManoptSolverState, ::Val{false}) = get_iterate(s)
get_solver_result(s::AbstractManoptSolverState, ::Val{true}) = get_solver_result(s.state)

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
@doc raw"""
    AbstractStateAction

a common `Type` for `AbstractStateActions` that might be triggered in decoraters,
for example within the [`DebugSolverState`](@ref) or within the [`RecordSolverState`](@ref).
"""
abstract type AbstractStateAction end

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
_storage_copy_vector(M::AbstractManifold, X) = copy(M, SA_F64[], X)
_storage_copy_vector(::AbstractManifold, X::Number) = StorageRef(X)

@doc raw"""
    StoreStateAction <: AbstractStateAction

internal storage for [`AbstractStateAction`](@ref)s to store a tuple of fields from an
[`AbstractManoptSolverState`](@ref)s

This functor posesses the usual interface of functions called during an
iteration, i.e. acts on `(p,o,i)`, where `p` is a [`AbstractManoptProblem`](@ref),
`o` is an [`AbstractManoptSolverState`](@ref) and `i` is the current iteration.

# Fields
* `values` – a dictionary to store interims values based on certain `Symbols`
* `keys` – a `Vector` of `Symbols` to refer to fields of `AbstractManoptSolverState`
* `point_values` – a `NamedTuple` of mutable values of points on a manifold to be stored in
  `StoreStateAction`. Manifold is later determined by `AbstractManoptProblem` passed
  to `update_storage!`.
* `point_init` – a `NamedTuple` of boolean values indicating whether a point in
  `point_values` with matching key has been already initialized to a value. When it is
  false, it corresponds to a general value not being stored for the key present in the
  vector `keys`.
* `vector_values` – a `NamedTuple` of mutable values of tangent vectors on a manifold to be
  stored in `StoreStateAction`. Manifold is later determined by `AbstractManoptProblem`
  passed to `update_storage!`. It is not specified at which point the vectors are tangent
  but for storage it should not matter.
* `vector_init` – a `NamedTuple` of boolean values indicating whether a tangent vector in
  `vector_values` with matching key has been already initialized to a value. When it is
  false, it corresponds to a general value not being stored for the key present in the
  vector `keys`.
* `once` – whether to update the internal values only once per iteration
* `lastStored` – last iterate, where this `AbstractStateAction` was called (to determine `once`)

To handle the general storage, use `get_storage` and `has_storage` with keys as `Symbol`s.
For the point storage use `PointStorageKey`. For tangent vector storage use
`VectorStorageKey`. Point and tangent storage have been optimized to be more efficient.

# Constructiors

   StoreStateAction(s::Vector{Symbol})

This is equivalent as providing `s` to the keyword `store_fields`, just that here, no manifold
is necessay for the construciton.

    StoreStateAction(M)

## Keyword arguments

* `store_fields` (`Symbol[]`)
* `store_points` (`Symbol[]`)
* `store_vectors` (`Symbol[]`)

as vectors of symbols each referring to fields of the state (lower case symbols)
or semantic ones (upper case).

* `p_init` (`rand(M)`)
* `X_init` (`zero_vector(M, p_init)`)

are used to initialize the point and vector storages, change these if you use other
types (than the default) for your points/vectors on `M`.

* `once` (`true`) whether to update internal storage only once per iteration or on every update call
"""
mutable struct StoreStateAction{
    TPS_asserts,TXS_assert,TPS<:NamedTuple,TXS<:NamedTuple,TPI<:NamedTuple,TTI<:NamedTuple
} <: AbstractStateAction
    values::Dict{Symbol,Any}
    keys::Vector{Symbol} # for values
    point_values::TPS
    vector_values::TXS
    point_init::TPI
    vector_init::TTI
    once::Bool
    last_stored::Int
    function StoreStateAction(
        general_keys::Vector{Symbol}=Symbol[],
        point_values::NamedTuple=NamedTuple(),
        vector_values::NamedTuple=NamedTuple(),
        once::Bool=true;
        M::AbstractManifold=DefaultManifold(),
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
            Dict{Symbol,Any}(),
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
    store_fields::Vector{Symbol}=Symbol[],
    store_points::Union{Type{TPS},Vector{Symbol}}=Tuple{},
    store_vectors::Union{Type{TTS},Vector{Symbol}}=Tuple{},
    p_init=rand(M),
    X_init=zero_vector(M, p_init),
    once=true,
) where {TPS<:Tuple,TTS<:Tuple}
    if store_points isa Vector{Symbol}
        TPS_tuple = tuple(store_points...)
    else
        TPS_tuple = Tuple(TPS.parameters)
    end
    if store_vectors isa Vector{Symbol}
        TTS_tuple = tuple(store_vectors...)
    else
        TTS_tuple = Tuple(TTS.parameters)
    end
    point_values = NamedTuple{TPS_tuple}(map(_ -> p_init, TPS_tuple))
    vector_values = NamedTuple{TTS_tuple}(map(_ -> X_init, TTS_tuple))
    return StoreStateAction(store_fields, point_values, vector_values, once; M=M)
end

@generated function extract_type_from_namedtuple(::Type{nt}, ::Val{key}) where {nt,key}
    for i in 1:length(nt.parameters[1])
        if nt.parameters[1][i] === key
            return nt.parameters[2].parameters[i]
        end
    end
    return Any
end

function _store_point_assert_type(
    ::StoreStateAction{TPS_asserts,TXS_assert}, key::Val
) where {TPS_asserts,TXS_assert}
    return extract_type_from_namedtuple(TPS_asserts, key)
end

function _store_vector_assert_type(
    ::StoreStateAction{TPS_asserts,TXS_assert}, key::Val
) where {TPS_asserts,TXS_assert}
    return extract_type_from_namedtuple(TXS_assert, key)
end

function (a::StoreStateAction)(
    amp::AbstractManoptProblem, s::AbstractManoptSolverState, i::Int
)
    #update values (maybe only once)
    if !a.once || a.last_stored != i
        update_storage!(a, amp, s)
    end
    return a.last_stored = i
end

"""
    get_storage(a::AbstractStateAction, key::Symbol)

Return the internal value of the [`AbstractStateAction`](@ref) `a` at the
`Symbol` `key`.
"""
get_storage(a::AbstractStateAction, key::Symbol) = a.values[key]

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
`Symbol` `key` that represents a vector vector.
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
        else
            a.values[key] = deepcopy(getproperty(s, key))
        end
    end

    M = get_manifold(amp)
    @inline function update_points(key)
        if key === :Iterate
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
        if key === :Gradient
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
function update_storage!(a::AbstractStateAction, d::Dict{Symbol,<:Any})
    merge!(a.values, d)
    # update keys
    return a.keys = collect(keys(a.values))
end

"""
    get_count(ams::AbstractManoptSolverState, ::Symbol)

Obtain the count for a certain countable size, e.g. the `:Iterations`.
This function returns 0 if there was nothing to count

Available symbols from within the solver state

* `:Iterations` is passed on to the `stop` field to obtain the
  iterataion at which the solver stopped.
"""
function get_count(ams::AbstractManoptSolverState, s::Symbol)
    return get_count(ams, Val(s))
end

function get_count(ams::AbstractManoptSolverState, v::Val{:Iterations})
    return get_count(ams.stop, v)
end
