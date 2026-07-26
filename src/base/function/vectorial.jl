@doc """
    AbstractVectorialType

An abstract type for different representations of a vectorial function
``f: $(_math(:Manifold)) → ℝ^m`` and its (component-wise) gradient/Jacobian
"""
abstract type AbstractVectorialType end

@doc """
    CoefficientVectorialType{B<:AbstractBasis} <: AbstractVectorialType

A type to indicate that the component, e.g. the Jacobian of a vectorial function ``F: $(_math(:Manifold)) → ℝ^m``
is implemented in coordinates, i.e. with respect to a certain basis
``$(_tex(:Cal, "B"))`` of ``$(_math(:TangentSpace))``, at ``p∈$(_math(:Manifold))``.
For example the Jacobian ``J_F(p) = (c_1^{$(_tex(:rm, "T"))},…,c_m^{$(_tex(:rm, "T"))})^{$(_tex(:rm, "T"))} ∈ ℝ^{m,d}``
is then an actual metric, where each row ``c_i`` is the coordinate representation of the
gradient ``$(_tex(:grad)) f_i`` of the component functions of ``F``,
cf. [`get_coordinates`](@extref `ManifoldsBase.get_coordinates`).

# Fields

* `basis` an [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) to indicate the basis
  with respect to which this representation is done.

# Constructor

    CoefficientVectorialType(basis = DefaultOrthonormalBasis())
"""
struct CoefficientVectorialType{B <: AbstractBasis} <: AbstractVectorialType
    basis::B
end
CoefficientVectorialType() = CoefficientVectorialType(DefaultOrthonormalBasis())

Base.show(io::IO, cvt::CoefficientVectorialType) = print(io, "CoefficientVectorialType($(cvt.basis))")

"""
    get_basis(::AbstractVectorialType)

Return a basis that fits a vector function representation.

For the case, where some vectorial data is stored with respect to a basis,
this function returns the corresponding basis, most prominently for the [`CoefficientVectorialType`](@ref).

If a type is not with respect to a certain basis, the [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`)
is returned.
"""
get_basis(::AbstractVectorialType) = DefaultOrthonormalBasis()
get_basis(cvt::CoefficientVectorialType) = cvt.basis

"""
    _to_iterable_indices(A::AbstractVector, i)

Convert index `i` (integer, colon, vector of indices, etc.) for array `A` into an iterable
structure of indices.
"""
function _to_iterable_indices(A::AbstractVector, i)
    idx = to_indices(A, (i,))[1]
    if idx isa Base.Slice
        return idx.indices
    else
        return idx
    end
end

@doc """
    ComponentVectorialType <: AbstractVectorialType

A type to indicate that a vectorial function ``F: $(_math(:Manifold)) → ℝ^m``
or one of its ingredients is implemented in single components ``f_i,$(_tex(:quad)) i=1,…,m``.

This can also be used to indicate that the Jacobian ``J_F`` of ``F`` is provided as single
gradient functions ``$(_tex(:grad)) f_i: $(_math(:Manifold)) → $(_math(:TangentBundle)),$(_tex(:quad)) p ↦ $(_tex(:grad)) f_i(p) ∈ $(_math(:TangentSpace))``.
"""
struct ComponentVectorialType <: AbstractVectorialType end

@doc """
    FunctionVectorialType{P<:AbstractPowerRepresentation} <: AbstractVectorialType

A type to indicate that a vectorial function ``F: $(_math(:Manifold)) → ℝ^m``
is implemented as a single function.

Similarly, its Jacobian is implemented as a single function
``J_F(p) ∈ ($(_math(:TangentSpace)))^m``,
where an [`AbstractPowerRepresentation`](@extref `ManifoldsBase.AbstractPowerRepresentation`)
is used to indicate how this ``m``-fold power of the tangent space is represented.

# Fields
* `range::P` the range this function maps into.
"""
struct FunctionVectorialType{P <: AbstractPowerRepresentation} <: AbstractVectorialType
    range::P
end
Base.show(io::IO, fvt::FunctionVectorialType) = print(io, "FunctionVectorialType($(fvt.range))")

"""
    get_range(::AbstractVectorialType)

Return an abstract power manifold representation that fits a vector function's range.
Most prominently a [`FunctionVectorialType`](@ref) returns its internal range.

Otherwise the default [`NestedPowerRepresentation`](@extref `ManifoldsBase.NestedPowerRepresentation`)`()` is used to work on
a vector of data.
"""
get_range(vt::FunctionVectorialType) = vt.range
get_range(::AbstractVectorialType) = NestedPowerRepresentation()

FunctionVectorialType() = FunctionVectorialType(NestedPowerRepresentation())

@doc """
    AbstractVectorFunction{FT} <: Function

Represent an abstract vectorial function ``f:$(_math(:Manifold)) → ℝ^n`` with an [`AbstractVectorialType`](@ref)
to specify the format ``f`` is implemented as.

# Representations of ``f``

There are three different representations of ``f``, which might be beneficial in one or
the other situation:

* the [`FunctionVectorialType`](@ref) storing a single function ``f`` that returns a vector,
* the [`ComponentVectorialType`](@ref) storing a vector of functions ``f_i`` that return a single value each,
* the [`CoefficientVectorialType`](@ref) storing functions with respect to a specific basis of the tangent space for gradients and Hessians.
  Gradients of this type are usually referred to as Jacobians.

For the [`ComponentVectorialType`](@ref) imagine that ``f`` could also be written
using its component functions,

```math
f(p) = $(_tex(:bigl))( f_1(p),f_2(p),…,.f_n(p) $(_tex(:bigr)))^{$(_tex(:rm, "T"))}
```

In this representation `f` is given as a vector `[f1(M,p), f2(M,p), ..., fn(M,p)]`
of its component functions.
An advantage is that the single components can be evaluated and from this representation
one even can directly read of the number `n`. A disadvantage might be, that one has to
implement a lot of individual (component) functions.

For the  [`FunctionVectorialType`](@ref) ``f`` is implemented as a single function
`f(M, p)`, that returns an `AbstractArray`.
And advantage here is, that this is a single function. A disadvantage might be,
that if this is expensive even to compute a single component, all of `f` has to be evaluated
"""
abstract type AbstractVectorFunction{FT <: AbstractVectorialType} <: Function end

@doc """
    length(vgf::AbstractVectorFunction)

Return the length of the vector the function ``f: $(_math(:Manifold)) → ℝ^n`` maps into,
that is the number `n`.
"""
Base.length(vgf::AbstractVectorFunction) = vgf.range_dimension

@doc """
    get_value(M::AbstractManifold, vgf::AbstractVectorFunction, p[, i=:])
    get_value!(M::AbstractManifold, V, vgf::AbstractVectorFunction, p[, i=:])

Evaluate the vector function [`VectorGradientFunction`](@ref) `vgf` at `p`.
The `range` can be used to specify a potential range, but is currently only present for consistency.

The `i` can be a linear index, you can provide

* a single integer
* a `UnitRange` to specify a range to be returned like `1:3`
* a `BitVector` specifying a selection
* a `AbstractVector{<:Integer}` to specify indices
* `:` to return the vector of all gradients, which is also the default

This function can perform the evaluation inplace of `V`.
"""
get_value(M::AbstractManifold, vgf::AbstractVectorFunction, p, i)
function get_value(
        M::AbstractManifold, vgf::AbstractVectorFunction{<:ComponentVectorialType}, p, i::Integer,
    )
    return vgf.value!![i](M, p)
end
function get_value(
        M::AbstractManifold, vgf::AbstractVectorFunction{<:ComponentVectorialType}, p, i = :
    )
    return i === Colon() ? [f(M, p) for f in vgf.value!!] : [f(M, p) for f in vgf.value!![i]]
end
function get_value(
        M::AbstractManifold, vgf::AbstractVectorFunction{<:FunctionVectorialType},
        p, i = :; value_cache = zeros(vgf.range_dimension),
    )
    vgf.value!!(M, value_cache, p)
    return value_cache[i]
end
function get_value!(
        M::AbstractManifold, V, vgf::AbstractVectorFunction{<:FunctionVectorialType}, p, i = :;
        value_cache = zeros(vgf.range_dimension),
    )
    vgf.value!!(M, value_cache, p)
    V .= value_cache[i]
    return V
end
function get_value!(
        M::AbstractManifold, V, vgf::AbstractVectorFunction{<:ComponentVectorialType}, p, i = :
    )
    if i === Colon()
        for i in eachindex(vgf.value!!, V)
            V[i] = vgf.value!![i](M, p)
        end
    else
        V .= i isa Number ? [vgf.value!![i](M, p)] : [f(M, p) for f in vgf.value!![i]]
    end
    return V
end

function Base.show(io::IO, ::MIME"text/plain", avf::AbstractVectorFunction)
    multiline = get(io, :multiline, true)
    return multiline ? status_summary(io, avf) : show(io, avf)
end

@doc """
    AbstractFirstOrderVectorFunction{FT, JT, F, J, I} <: AbstractManifoldObjective{E}

Represent an abstract vectorial function ``f:$(_math(:Manifold)) → ℝ^n`` that provides
some first order differential information.

The the [`AbstractVectorialType`](@ref)s `FT` and `JT` indicate the formats in which
the function and the first order information, e.g.

* a gradient – see [`AbstractVectorGradientFunction`](@ref)
* a differential (or Jacobian) – see [`VectorDifferentialFunction`](@ref)

are provided, respectively.
"""
abstract type AbstractFirstOrderVectorFunction{FT <: AbstractVectorialType, JT <: AbstractVectorialType} <: AbstractVectorFunction{FT} end

function allocate_jacobian(M::AbstractManifold, vgf::AbstractFirstOrderVectorFunction; T::Type = Float64)
    n = vgf.range_dimension
    d = number_of_coordinates(M, get_basis(vgf.jacobian_type))
    return Matrix{T}(undef, n, d)
end
function allocate_jacobian(M::AbstractManifold, vgf::AbstractFirstOrderVectorFunction, B::AbstractBasis; T::Type = Float64)
    n = vgf.range_dimension
    d = number_of_coordinates(M, B)
    return Matrix{T}(undef, n, d)
end

"""
    _change_basis!(M::AbstractManifold, JF, p, from_basis::B1, to_basis::B; X=zero_vector(M,p))

Given a jacobian matrix `JF` on a manifold `M` at `p` with respect to the `from_basis`
in the tangent space of `p` on `M`. Change the basis of the Jacobian to `to_basis` in place of `JF`.

# Keyword Arguments
* `X` a temporary vector to store a generated vector, before decomposing it again with respect to the new basis
"""
function _change_basis!(
        M, JF, p, from_basis::B1, to_basis::B2; X = zero_vector(M, p)
    ) where {B1 <: AbstractBasis, B2 <: AbstractBasis}
    # change every row to new basis
    for i in 1:size(JF, 1) # every row
        get_vector!(M, X, p, view(JF, i, :), from_basis)
        get_coordinates!(M, view(JF, i, :), p, X, to_basis)
    end
    return JF
end
# case we have the same basis: nothing to do, just return JF
function _change_basis!(
        M, JF, p, from_basis::B, to_basis_new::B; kwargs...
    ) where {B <: AbstractBasis}
    return JF
end

_doc_add_adjoint_jacobian_function_vector = """
    add_adjoint_jacobian!(M::AbstractManifold, X, vgf::AbstractFirstOrderVectorFunction, p, a; kwargs...)

Compute the adjoint Jacobian ``J_F^*(p)[a]`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^n``
and add it to `X`.
For more details see [`get_adjoint_jacobian`](@ref).
"""

@doc "$(_doc_add_adjoint_jacobian_function_vector)"
add_adjoint_jacobian!(M::AbstractManifold, X, vgf::AbstractFirstOrderVectorFunction, p, a; kwargs...)

_doc_add_adjoint_jacobian_function_coeff = """
    add_adjoint_jacobian!(M::AbstractManifold, c, vgf::AbstractVectorGradientFunction, p, a, B::AbstractBasis; kwargs...)

Compute the adjoint Jacobian ``J_F^*(p)[a]`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^n`` as a matrix in a tangent space.
For more details see [`get_adjoint_jacobian`](@ref).
"""

@doc "$(_doc_add_adjoint_jacobian_function_coeff)"
add_adjoint_jacobian!(M::AbstractManifold, c, vgf::AbstractFirstOrderVectorFunction, p, a::AbstractVector, B::AbstractBasis; kwargs...)


_doc_get_adjoint_jacobian_function_coeff = """
    get_adjoint_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, a::AbstractVector, B::AbstractBasis; kwargs...)
    get_adjoint_jacobian!(M::AbstractManifold, c, vgf::AbstractVectorGradientFunction, p, a::AbstractVector, B::AbstractBasis; kwargs...)

Compute the adjoint Jacobian ``J_F^*(p)[a]`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^n``
how it acts on a vector `a` at `p`, i.e., it is given by the relation

````math
$(_tex(:inner, "J_F^*(p)[a]", "X"; index = "p")) = $(_tex(:inner, "a", "J_F(p)[X]")),
````

where the basis indicates that the result should be given in coordinates `c` with respect to that basis.
This can be done in-place of `c`.

Note that if `vgf` works internally in a basis different from the one provided, and additional change of basis is performed.
"""

@doc "$(_doc_get_adjoint_jacobian_function_coeff)"
function get_adjoint_jacobian(
        M::AbstractManifold, vgf::AbstractFirstOrderVectorFunction, p, a::AbstractVector, B::AbstractBasis; kwargs...
    )
    c = get_coordinates(M, p, zero_vector(M, p), B)
    return add_adjoint_jacobian!(M, c, vgf, p, a, B; kwargs...)
end

@doc "$(_doc_get_adjoint_jacobian_function_coeff)"
function get_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::AbstractFirstOrderVectorFunction, p, a::AbstractVector, B::AbstractBasis; kwargs...
    )
    fill!(c, 0)
    return add_adjoint_jacobian!(M, c, vgf, p, a, B; kwargs...)
end

_doc_get_adjoint_jacobian_vector = """
    get_adjoint_jacobian(M::AbstractManifold, vgf::AbstractFirstOrderVectorFunction, p, a; kwargs...)
    get_adjoint_jacobian!(M::AbstractManifold, X, vgf::AbstractFirstOrderVectorFunction, p, a; kwargs...)

Compute the adjoint Jacobian ``J_F^*(p)[a]`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^n``
how it acts on a vector `a` at `p`, i.e., it is given by the relation

````math
$(_tex(:inner, "J_F^*(p)[a]", "X"; index = "p")) = $(_tex(:inner, "a", "J_F(p)[X]")),
````

where the inner product on the right hand side is the standard Euclidean inner product on ``ℝ^n``.
To be precise, the adjoint Jacobian is defined using the Riemannian gradients of the component functions
``F_i`` of ``F`` as

````math
J_F^*(p): ℝ^m → $(_math(:TangentSpace)),
$(_tex(:qquad))
J_F^*(p)[a] = $(_tex(:sum, "i=1", "m")) a_i $(_tex(:grad))F_i(p),
````

This can be computed in-place of `X`. To directly add a Jacobian to `X` see [`add_adjoint_jacobian!`](@ref)

!!! note
    For the case of a matrix representation, i.e. the function signature
    `get_jacobian!(M, JF, vgf, p)` the resulting matrix can just be transposed to obtain the adjoint,
    if you use an [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`).
"""

@doc "$(_doc_get_adjoint_jacobian_vector)"
function get_adjoint_jacobian(
        M::AbstractManifold, vgf::AbstractFirstOrderVectorFunction, p, a::AbstractVector; kwargs...
    )
    X = zero_vector(M, p)
    return add_adjoint_jacobian!(M, X, vgf, p, a, kwargs...)
end

@doc "$(_doc_get_adjoint_jacobian_vector)"
function get_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::AbstractFirstOrderVectorFunction, p, a::AbstractVector; kwargs...
    )
    zero_vector!(M, X, p)
    return add_adjoint_jacobian!(M, X, vgf, p, a, kwargs...)
end

@doc """
    get_gradient(M::AbstractManifold, vgf::AbstractFirstOrderVectorFunction, p, i)
    get_gradient(M::AbstractManifold, vgf::AbstractFirstOrderVectorFunction, p, i, range)
    get_gradient!(M::AbstractManifold, X, vgf::AbstractFirstOrderVectorFunction, p, i)
    get_gradient!(M::AbstractManifold, X, vgf::AbstractFirstOrderVectorFunction, p, i, range)

Evaluate the gradient(s) of the vector function `vgf` on the manifold `M` at `p` and
the values given in `range`, specifying the representation of the gradients.

Since `i` is assumed to be a linear index, you can provide
* a single integer
* a `UnitRange` to specify a range to be returned like `1:3`
* a `BitVector` specifying a selection
* a `AbstractVector{<:Integer}` to specify indices
* `:` to return the vector of all gradients
"""
get_gradient(
    M::AbstractManifold, vgf::AbstractFirstOrderVectorFunction,
    p, i, range::Union{AbstractPowerRepresentation, Nothing} = nothing,
)

_doc_get_jacobian_matrix_vgf = """
    get_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p; kwargs...)
    get_jacobian!(M::AbstractManifold, J, vgf::AbstractVectorGradientFunction, p; kwargs...)

Return the Jacobian ``J_F(p): $(_math(:TangentSpace)) → ℝ^m`` of a [`AbstractVectorGradientFunction`](@ref) `vgf`,
i.e. a function ``F: $(_math(:Manifold)) → ℝ^m``, where `p ∈ $(_math(:Manifold))`, in matrix form with respect to
a basis ``$(_tex(:Cal, "B")) = $(_tex(:set, "Y_1,…,Y_n"))``of the tangent space.

Then decomposing a tangent vector ``X = $(_tex(:displaystyle))$(_tex(:sum, "i=1", "d")) c_iX_i``
the evaluation of the Jacobian can be written as

````math
J_F(p)[X] = J c.
````

In other words, the `j`th column of ``J`` is given by ``DF(p)[Y_j]`` and this function returns
the matrix ``J``. The computation can be computed in-place of `J`.

# Keyword arguments

* `basis::AbstractBasis = `[`get_basis`](@ref)`(vgf)` basis with respect to which the matrix
  is built. For the [`CoefficientVectorialType`](@ref) of the vectorial functions gradient, this
  might lead to a change of basis, if this basis and the one the coordinates are given in do not agree.
* `range::AbstractPowerRepresentation = `[`get_range`](@ref)`(vgf.jacobian_type)`
  specify the range of the gradients in the case of a [`FunctionVectorialType`](@ref),
  that is, on which type of power manifold the gradient(s) of the function is/are given on.
"""

@doc "$(_doc_get_jacobian_matrix_vgf)"
get_jacobian(::AbstractManifold, ::AbstractFirstOrderVectorFunction, p; kwargs...)
function get_jacobian! end
@doc "$(_doc_get_jacobian_matrix_vgf)"
get_jacobian!(M::AbstractManifold, JF, vgf::AbstractFirstOrderVectorFunction, p)

function get_jacobian(
        M::AbstractManifold, vgf::VGF, p; basis::AbstractBasis = get_basis(vgf.jacobian_type), kwargs...
    ) where {FT, VGF <: AbstractFirstOrderVectorFunction{FT, <:AbstractVectorialType}}
    JF = allocate_jacobian(M, vgf, basis; T = number_eltype(p))
    return get_jacobian!(M, JF, vgf, p; basis = basis, kwargs...)
end

_doc_get_jacobian_function_vector = """
    get_jacobian(M::AbstractManifold, vgf::AbstractFirstOrderVectorFunction, p, X; kwargs...)
    get_jacobian!(M::AbstractManifold, a, vgf::AbstractFirstOrderVectorFunction, p, X; kwargs...)

Compute the Jacobian ``J_F(p)`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^m``,
to be precise how it acts on a tangent vector `X` at `p` on the manifold `M`, i.e., compute

````math
J_F(p)[X] = DF(p)[X] ∈ ℝ^m
````

If the gradient functions of the single component functions are provided, this is given by

````math
J_F(p)[X] = $(
    _tex(
        :pmatrix,
        _tex(:inner, "$(_tex(:grad))F_1(p)", "X"), _tex(:inner, "$(_tex(:grad))F_2(p)", "X"), _tex(:vdots), _tex(:inner, "$(_tex(:grad))F_m(p)", "X")
    )
) ∈ ℝ^m
````

Given a basis ``$(_tex(:set, "Y_1,…,Y_n"))`` this can also be computed in coordinates of this basis.
Then it simplifies to a matrix multiplication.

This can be computed in-place of `a`.
"""

@doc "$(_doc_get_jacobian_function_vector)"
get_jacobian(M::AbstractManifold, vgf::AbstractFirstOrderVectorFunction, p, X; kwargs...)

@doc "$(_doc_get_jacobian_function_vector)"
get_jacobian!(M::AbstractManifold, a, vgf::AbstractFirstOrderVectorFunction, p, X; kwargs...)

function get_jacobian(
        M::AbstractManifold, vgf::VGF, p, X; kwargs...
    ) where {FT, VGF <: AbstractFirstOrderVectorFunction{FT, <:AbstractVectorialType}}
    n = vgf.range_dimension
    a = zeros(number_eltype(X), n)
    return get_jacobian!(M, a, vgf, p, X; kwargs...)
end

@doc """
    AbstractVectorGradientFunction{FT, JT} <: AbstractFirstOrderVectorFunction{FT, JT}

Represent an abstract vectorial function ``f:$(_math(:Manifold)) → ℝ^n`` that provides a (component wise)
gradient.
The [`AbstractVectorialType`](@ref)s `FT` and `JT` indicate the formats in which
the function and the gradient are provided, see [`AbstractVectorFunction`](@ref) for an explanation.
"""
abstract type AbstractVectorGradientFunction{
    FT <: AbstractVectorialType, JT <: AbstractVectorialType,
} <: AbstractFirstOrderVectorFunction{FT, JT} end

#
#
# --- adjoint Jacobian – just the add_adjoint_jacobian variant
# As operator
# (a) as a function
function add_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{FT, <:FunctionVectorialType}, p, a::AbstractVector;
        Y_cache = nothing,
    ) where {FT}
    n = vgf.range_dimension
    mP = PowerManifold(M, get_range(vgf.jacobian_type), n)
    gradients = zero_vector(mP, fill(p, mP))
    vgf.jacobian!!(M, gradients, p)
    for i in 1:n
        X .+= a[i] * gradients[mP, i]
    end
    return X
end
# (b) vector of gradient functions
function add_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{FT, <:ComponentVectorialType}, p, a::AbstractVector;
        Y_cache = zero_vector(M, p)
    ) where {FT}
    n = vgf.range_dimension
    for i in 1:n
        vgf.jacobian!![i](M, Y_cache, p)
        X .+= a[i] .* Y_cache
    end
    return X
end
# (c) Jacobian function
function add_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{FT, <:CoefficientVectorialType}, p, a::AbstractVector;
        Y_cache = nothing
    ) where {FT}
    J = allocate_jacobian(M, vgf; T = eltype(a))
    add_vector!(M, X, p, adjoint(vgf.jacobian!!(M, J, p)) * a, vgf.jacobian_type.basis)
    return X
end
# in coordinates
# as a function
function add_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::VGF, p, a::AbstractVector, B::AbstractBasis; X = zero_vector(M, p), Y_cache = nothing,
    ) where {FT, VGF <: AbstractVectorGradientFunction{FT, <:FunctionVectorialType}}
    # easiest: call the one for the vector and decompose into c
    add_adjoint_jacobian!(M, X, vgf, p, a)
    return add_coordinates!(M, c, p, X, B)
end
# (b) vector of gradient functions - same as above
function add_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::VGF, p, a::AbstractVector, B::AbstractBasis; X = zero_vector(M, p), Y_cache = nothing,
    ) where {FT, VGF <: AbstractVectorGradientFunction{FT, <:ComponentVectorialType}}
    add_adjoint_jacobian!(M, X, vgf, p, a)
    add_coordinates!(M, c, p, X, B)
    return c
end
# (c) Jacobian function
function add_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::AbstractVectorGradientFunction{FT, <:CoefficientVectorialType}, p, a::AbstractVector, B::AbstractBasis; X = nothing, Y_cache = nothing
    ) where {FT}
    J = allocate_jacobian(M, vgf; T = eltype(a))
    JFt = adjoint(vgf.jacobian!!(M, J, p))
    if vgf.jacobian_type.basis === B
        mul!(c, JFt, a, one(eltype(c)), one(eltype(c)))
    else
        cX = JFt * a
        # `change_basis` may be expensive in general so we don't want to do it unless necessary
        c .+= change_basis(M, p, cX, vgf.jacobian_type.basis, B)
    end
    return c
end

#
#
# ---- Gradient

# Generic case, allocate (a) a single tangent vector
function get_gradient(
        M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    )
    X = zero_vector(M, p)
    return get_gradient!(M, X, vgf, p, i, range)
end
# (b) UnitRange and AbstractVector allow to use length for BitVector its sum
function get_gradient(
        M::AbstractManifold, vgf::AbstractVectorGradientFunction,
        p, i = :, # as long as the length can be found it should work, see _vgf_index_to_length
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    )
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    X = zero_vector(pM, fill(p, pM))
    return get_gradient!(M, X, vgf, p, i, range)
end
# (c) Special cases where allocations can be skipped
function get_gradient(
        M::AbstractManifold,
        vgf::AbstractVectorGradientFunction{FT, <:ComponentVectorialType},
        p, i::Integer, ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT <: AbstractVectorialType}
    X = zero_vector(M, p)
    return vgf.jacobian!![i](M, X, p)
end
# To evaluate the gradient, we have to distinguish whether we get an index or a range
# (a) Jacobian
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{FT, <:CoefficientVectorialType},
        p, i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    # a type wise safe way to allocate what usually should yield a n-times-d matrix
    JF = allocate_jacobian(M, vgf; T = number_eltype(X))
    vgf.jacobian!!(M, JF, p)
    get_vector!(M, X, p, view(JF, i, :), vgf.jacobian_type.basis)
    return X
end
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{FT, <:CoefficientVectorialType},
        p, i,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    # a type wise safe way to allocate what usually should yield a n-times-d matrix
    JF = allocate_jacobian(M, vgf; T = number_eltype(X))
    vgf.jacobian!!(M, JF, p)
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    for (j, k) in zip(_to_iterable_indices([JF[:, 1]...], i), 1:n)
        get_vector!(M, _write(pM, rep_size, X, (k,)), p, view(JF, j, :), vgf.jacobian_type.basis)
    end
    return X
end
#II (b) a vector of functions
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{FT, <:ComponentVectorialType},
        p, i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    return vgf.jacobian!![i](M, X, p)
end
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{FT, <:ComponentVectorialType},
        p, i,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    # In the resulting X the indices are linear,
    # in jacobian[i] have the functions f are also given n a linear sense
    for (j, f) in zip(1:n, vgf.jacobian!![i])
        f(M, _write(pM, rep_size, X, (j,)), p)
    end
    return X
end
# II(c) a single function
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{FT, <:FunctionVectorialType},
        p, i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    pM = PowerManifold(M, range, vgf.range_dimension...)
    P = fill(p, pM)
    x = zero_vector(pM, P)
    vgf.jacobian!!(M, x, p)
    copyto!(M, X, p, x[pM, i])
    return X
end
function get_gradient!(
        M::AbstractManifold, X, vgf::VGF, p, i,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType, VGF <: AbstractVectorGradientFunction{FT, <:FunctionVectorialType}}
    # Single access for function is a bit expensive
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM_out = PowerManifold(M, range, n)
    pM_temp = PowerManifold(M, range, vgf.range_dimension)
    P = fill(p, pM_temp)
    x = zero_vector(pM_temp, P)
    vgf.jacobian!!(M, x, p)
    # Luckily all documented access functions work directly on `x[pM_temp,...]`
    copyto!(pM_out, X, P[pM_temp, i], x[pM_temp, i])
    return X
end


#
#
# --- Jacobian in matrix form

# (a) We have a single gradient function
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p;
        basis::AbstractBasis = default_basis(M, typeof(p)), range::AbstractPowerRepresentation = get_range(vgf.jacobian_type),
    ) where {FT, VGF <: AbstractVectorGradientFunction{FT, <:FunctionVectorialType}}
    mP = PowerManifold(M, range, vgf.range_dimension)
    gradients = zero_vector(mP, fill(p, mP))
    vgf.jacobian!!(M, gradients, p)
    for i in 1:(vgf.range_dimension)
        get_coordinates!(M, view(JF, i, :), p, gradients[mP, i], basis)
    end
    return JF
end
# (b) a vector of functions
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p;
        basis = get_basis(vgf.jacobian_type), X = zero_vector(M, p), range = nothing,
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{FT, <:ComponentVectorialType},
    }
    for i in 1:(vgf.range_dimension)
        vgf.jacobian!![i](M, X, p)
        get_coordinates!(M, view(JF, i, :), p, X, basis)
    end
    return JF
end
# (c) a matrix
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p;
        basis::AbstractBasis = get_basis(vgf.jacobian_type), range = nothing, X = nothing, Y_cache = nothing,
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{FT, <:CoefficientVectorialType},
    }
    vgf.jacobian!!(M, JF, p)
    _change_basis!(M, JF, p, vgf.jacobian_type.basis, basis)
    return JF
end

#
#
# --- Jacobian in linear operator form
# (a) Inplace single function – skip for now since allocation not so easy? we would need a power version of the point p
function get_jacobian!(
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{FT, <:FunctionVectorialType}, p, X;
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
        Y_cache = nothing,
    ) where {FT}
    n = vgf.range_dimension
    mP = PowerManifold(M, range, n)
    gradients = zero_vector(mP, fill(p, mP))
    vgf.jacobian!!(M, gradients, p)
    for i in 1:n
        a[i] = inner(M, p, gradients[mP, i], X)
    end
    return a
end
# (b) vector of gradient functions
function get_jacobian!(
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{FT, <:ComponentVectorialType}, p, X;
        Y_cache = zero_vector(M, p), c_cache = nothing,
    ) where {FT}
    n = vgf.range_dimension
    for i in 1:n
        vgf.jacobian!![i](M, Y_cache, p)
        a[i] = inner(M, p, Y_cache, X)
    end
    return a
end


_doc_get_jacobian_function_coord = """
    get_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, c, B::AbstractBasis; kwargs...)
    get_jacobian!(M::AbstractManifold, a, vgf::AbstractVectorGradientFunction, p, c, B::AbstractBasis; kwargs...)

Compute the Jacobian ``J_F(p)`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^m``,
to be precise how it acts on a tangent vector `X` at `p` on the manifold `M`, i.e., compute

````math
J_F(p)[X] = DF(p)[X] ∈ ℝ^m
````

where a basis ``$(_tex(:set, "Y_1,…,Y_n"))`` allows to decompose / provide the tangent vector
in coordinates ``c`` given by ``X = $(_tex(:displaystyle))$(_tex(:sum, "i=1", "d")) c_iY_i``
and the computation simplifies to a matrix multiplication.

This can be computed in-place of `a`.

# Keyword arguments
$(_kwargs(:X)) used as memory to compute the interims tangent vector where necessary,
  non-allocating and/or ignored where not necessary

!!! Technical Note
  This variant only differs in the last argument from `get_jacobian(M, vgf, p, X)`,
  which works for tangent vectors X provided directly. Hence the basis is necessary to indicate
  that this method shall work in coordinates.
  For performance reasons, try to pass `get_basis(vgf.jacobian_type)` as `B` where possible.
"""

@doc "$(_doc_get_jacobian_function_coord)"
get_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, c, B::AbstractBasis; kwargs...)
@doc "$(_doc_get_jacobian_function_coord)"
get_jacobian!(M::AbstractManifold, a, vgf::AbstractVectorGradientFunction, p, c, B::AbstractBasis; kwargs...)

function get_jacobian(
        M::AbstractManifold, vgf::VGF, p, c, B::AbstractBasis; kwargs...
    ) where {VGF <: AbstractFirstOrderVectorFunction}
    n = vgf.range_dimension
    a = zeros(number_eltype(c), n)
    return get_jacobian!(M, a, vgf, p, c, B; kwargs...)
end
# (a) single gradient function
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, c, B::AbstractBasis; X = zero_vector(M, p), kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{FT, <:FunctionVectorialType}}
    # in this case it is easiest to reconstruct X and call the one for X
    get_vector!(M, X, p, c, B)
    return get_jacobian!(M, a, vgf, p, X, kwargs...)
end
# (b) vector of gradient functions
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, c, B::AbstractBasis; X = zero_vector(M, p), kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{FT, <:ComponentVectorialType}}
    # Here it is easiest to reconstruct X and call the variant for X
    get_vector!(M, X, p, c, B)
    return get_jacobian!(M, a, vgf, p, X; kwargs...)
end
# (c) Jacobian function
function get_jacobian!(
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{FT, <:CoefficientVectorialType}, p, c, B::AbstractBasis;
        X = nothing, Y_cache = nothing,
    ) where {FT}
    JF = allocate_jacobian(M, vgf; T = eltype(c))
    vgf.jacobian!!(M, JF, p)
    a .= JF * c
    return a
end


function get_jacobian_basis(vgf::AbstractVectorGradientFunction)
    return _get_jacobian_basis(vgf.jacobian_type)
end
_get_jacobian_basis(jt::AbstractVectorialType) = DefaultOrthonormalBasis()
_get_jacobian_basis(jt::CoefficientVectorialType) = jt.basis
