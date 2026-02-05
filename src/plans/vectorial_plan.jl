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
is then an actual matric, where each row ``c_i`` is the coordinate representation of the
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
or one of its ingredients is implemented in single components ``f_i,$(_tex(:quad))i=1,…,m``.

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
    AbstractVectorFunction{E, FT} <: Function

Represent an abstract vectorial function ``f:$(_math(:Manifold)) → ℝ^n`` with an
[`AbstractEvaluationType`](@ref) `E` and an [`AbstractVectorialType`](@ref) to specify the
format ``f`` is implemented as.

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
f(p) = $(_tex(:bigl))( f_1(p), f_2(p),…, f_n(p) $(_tex(:bigr)))^{$(_tex(:rm, "T"))}
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
abstract type AbstractVectorFunction{E <: AbstractEvaluationType, FT <: AbstractVectorialType} <:
Function end

@doc """
    VectorGradientFunction{E, FT, JT, F, J, I} <: AbstractManifoldObjective{E}

Represent an abstract vectorial function ``f:$(_math(:Manifold)) → ℝ^n`` that provides a (component wise)
gradient.
The [`AbstractEvaluationType`](@ref) `E` indicates the evaluation type,
and the [`AbstractVectorialType`](@ref)s `FT` and `JT` the formats in which
the function and the gradient are provided, see [`AbstractVectorFunction`](@ref) for an explanation.
"""
abstract type AbstractVectorGradientFunction{
    E <: AbstractEvaluationType, FT <: AbstractVectorialType, JT <: AbstractVectorialType,
} <: AbstractVectorFunction{E, FT} end

@doc """
    VectorGradientFunction{E, FT, JT, F, J, I} <: AbstractVectorGradientFunction{E, FT, JT}

Represent a function ``f:$(_math(:Manifold)) → ℝ^n`` including it first derivative,
either as a vector of gradients of a Jacobian

And hence has a gradient ``$(_tex(:grad)) f_i(p) ∈ $(_math(:TangentSpace))`.
Putting these gradients into a vector the same way as the functions, yields a
[`ComponentVectorialType`](@ref)

```math
$(_tex(:grad)) f(p) = $(_tex(:Bigl))( $(_tex(:grad)) f_1(p), $(_tex(:grad)) f_2(p), …, $(_tex(:grad)) f_n(p) $(_tex(:Bigr)))^$(_tex(:transp))
∈ ($(_math(:TangentSpace)))^n
```

And advantage here is, that again the single components can be evaluated individually

# Fields

* `value!!::F`:   the cost function ``f``, which can take different formats
* `cost_type::`[`AbstractVectorialType`](@ref): indicating the format how the vector function is stored,
  e.g. as a single function ([`FunctionVectorialType`](@ref), default) or as a vector of functions ([`ComponentVectorialType`](@ref))
* `jacobian!!::J`: the Jacobian ``J_f``of ``f``
* `jacobian_type::`[`AbstractVectorialType`](@ref): indicating / storing data for the type of ``J_f``, e.g.
  as a single function ([`FunctionVectorialType`](@ref), default) -,
  as a vector of functions ([`ComponentVectorialType`](@ref)), or
  as a function returning a matrix in coordinates at every point ([`CoefficientVectorialType`](@ref))
* `range_dimension`: the number `n` from, the size of the vector ``f`` returns.

# Constructor

    VectorGradientFunction(f, Jf, range_dimension;
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
        function_type::AbstractVectorialType=FunctionVectorialType(),
        jacobian_type::AbstractVectorialType=FunctionVectorialType(),
        range_dimension::Integer.
    )

Create a `VectorGradientFunction` of `f`  and its Jacobian (vector of gradients) `Jf`,
where `f` maps into the Euclidean space of dimension `range_dimension`.
Their types are specified by the `function_type`, and `jacobian_type`, respectively.
The Jacobian can further be given as an allocating variant or an in-place variant, specified
by the `evaluation=` keyword.
"""
struct VectorGradientFunction{
        E <: AbstractEvaluationType,
        FT <: AbstractVectorialType,
        JT <: AbstractVectorialType,
        F,
        J,
        I <: Integer,
    } <: AbstractVectorGradientFunction{E, FT, JT}
    value!!::F
    cost_type::FT
    jacobian!!::J
    jacobian_type::JT
    range_dimension::I
end

function VectorGradientFunction(
        f::F,
        Jf::J,
        range_dimension::I;
        evaluation::E = AllocatingEvaluation(),
        function_type::FT = FunctionVectorialType(),
        jacobian_type::JT = FunctionVectorialType(),
    ) where {
        I <: Integer,
        F,
        J,
        E <: AbstractEvaluationType,
        FT <: AbstractVectorialType,
        JT <: AbstractVectorialType,
    }
    return VectorGradientFunction{E, FT, JT, F, J, I}(
        f, function_type, Jf, jacobian_type, range_dimension
    )
end

_doc_vhf = """
    VectorHessianFunction{E, FT, JT, HT, F, J, H, I} <: AbstractVectorGradientFunction{E, FT, JT}

Represent a function ``f:$(_math(:Manifold)) M → ℝ^n`` including it first derivative,
either as a vector of gradients of a Jacobian, and the Hessian,
as a vector of Hessians of the component functions.

Both the Jacobian and the Hessian can map into either a sequence of tangent spaces
or a single tangent space of the power manifold of length `n`.

# Fields

* `value!!::F`:          the cost function ``f``, which can take different formats
* `cost_type::`[`AbstractVectorialType`](@ref):     indicating / string data for the type of `f`
* `jacobian!!::G`:     the Jacobian ``J_f`` of ``f``
* `jacobian_type::`[`AbstractVectorialType`](@ref): indicating / storing data for the type of ``J_f``
* `hessians!!::H`:     the Hessians of ``f`` (in a component wise sense)
* `hessian_type::`[`AbstractVectorialType`](@ref):  indicating / storing data for the type of ``H_f``
* `range_dimension`:    the number `n` from, the size of the vector ``f`` returns.

# Constructor

    VectorHessianFunction(f, Jf, Hess_f, range_dimension;
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
        function_type::AbstractVectorialType=FunctionVectorialType(),
        jacobian_type::AbstractVectorialType=FunctionVectorialType(),
        hessian_type::AbstractVectorialType=FunctionVectorialType(),
    )

Create a `VectorHessianFunction` of `f`  and its Jacobian (vector of gradients) `Jf`
and (vector of) Hessians, where `f` maps into the Euclidean space of dimension `range_dimension`.
Their types are specified by the `function_type`, and `jacobian_type`, and `hessian_type`,
respectively. The Jacobian and Hessian can further be given as an allocating variant or an
inplace-variant, specified by the `evaluation=` keyword.
"""

@doc "$(_doc_vhf)"
struct VectorHessianFunction{
        E <: AbstractEvaluationType,
        FT <: AbstractVectorialType,
        JT <: AbstractVectorialType,
        HT <: AbstractVectorialType,
        F,
        J,
        H,
        I <: Integer,
    } <: AbstractVectorGradientFunction{E, FT, JT}
    value!!::F
    cost_type::FT
    jacobian!!::J
    jacobian_type::JT
    hessians!!::H
    hessian_type::HT
    range_dimension::I
end

function VectorHessianFunction(
        f::F,
        Jf::J,
        Hf::H,
        range_dimension::I;
        evaluation::E = AllocatingEvaluation(),
        function_type::FT = FunctionVectorialType(),
        jacobian_type::JT = FunctionVectorialType(),
        hessian_type::HT = FunctionVectorialType(),
    ) where {
        I <: Integer,
        F,
        J,
        H,
        E <: AbstractEvaluationType,
        FT <: AbstractVectorialType,
        JT <: AbstractVectorialType,
        HT <: AbstractVectorialType,
    }
    return VectorHessianFunction{E, FT, JT, HT, F, J, H, I}(
        f, function_type, Jf, jacobian_type, Hf, hessian_type, range_dimension
    )
end

_vgf_index_to_length(b::BitVector, n) = sum(b)
_vgf_index_to_length(::Colon, n) = n
_vgf_index_to_length(i::AbstractArray{<:Integer}, n) = length(i)
_vgf_index_to_length(r::UnitRange{<:Integer}, n) = length(r)

#
#
# ---- Hessian
@doc """
    get_hessian(M::AbstractManifold, vgf::VectorHessianFunction, p, X, i)
    get_hessian(M::AbstractManifold, vgf::VectorHessianFunction, p, X, i, range)
    get_hessian!(M::AbstractManifold, X, vgf::VectorHessianFunction, p, X, i)
    get_hessian!(M::AbstractManifold, X, vgf::VectorHessianFunction, p, X, i, range)

Evaluate the Hessians of the vector function `vgf` on the manifold `M` at `p` in direction `X`
and the values given in `range`, specifying the representation of the gradients.

Since `i` is assumed to be a linear index, you can provide
* a single integer
* a `UnitRange` to specify a range to be returned like `1:3`
* a `BitVector` specifying a selection
* a `AbstractVector{<:Integer}` to specify indices
* `:` to return the vector of all Hessian evaluations
"""
get_hessian(
    M::AbstractManifold,
    vgf::VectorHessianFunction,
    p,
    X,
    i,
    range::Union{AbstractPowerRepresentation, Nothing} = nothing,
)

# Generic case, allocate (a) a single tangent vector
function get_hessian(
        M::AbstractManifold,
        vhf::VectorHessianFunction,
        p,
        X,
        i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    )
    Y = zero_vector(M, p)
    return get_hessian!(M, Y, vhf, p, X, i, range)
end
# (b) UnitRange and AbstractVector allow to use length for BitVector its sum
function get_hessian(
        M::AbstractManifold,
        vhf::VectorHessianFunction,
        p,
        X,
        i = :, # as long as the length can be found it should work, see _vgf_index_to_length
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    )
    n = _vgf_index_to_length(i, vhf.range_dimension)
    pM = PowerManifold(M, range, n)
    P = fill(p, pM)
    Y = zero_vector(pM, P)
    return get_hessian!(M, Y, vhf, p, X, i, range)
end

#
#
# Part I: allocation
# I (a) a vector of functions
function get_hessian!(
        M::AbstractManifold,
        Y,
        vhf::VectorHessianFunction{<:AllocatingEvaluation, FT, JT, <:ComponentVectorialType},
        p,
        X,
        i::Integer,
        (::Union{AbstractPowerRepresentation, Nothing}) = nothing,
    ) where {FT, JT}
    return copyto!(M, Y, p, vhf.hessians!![i](M, p, X))
end
function get_hessian!(
        M::AbstractManifold,
        Y,
        vhf::VectorHessianFunction{<:AllocatingEvaluation, FT, JT, <:ComponentVectorialType},
        p,
        X,
        i,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    ) where {FT, JT}
    n = _vgf_index_to_length(i, vhf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    # In the resulting `X` the indices  are linear,
    # in `jacobian[i]` the functions f are ordered in a linear sense
    for (j, f) in zip(1:n, vhf.hessians!![i])
        copyto!(M, _write(pM, rep_size, Y, (j,)), f(M, p, X))
    end
    return Y
end
function get_hessian!(
        M::AbstractManifold,
        Y,
        vhf::VectorHessianFunction{<:AllocatingEvaluation, FT, JT, <:ComponentVectorialType},
        p,
        X,
        i::Colon,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    ) where {FT, JT}
    n = _vgf_index_to_length(i, vhf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    for (j, f) in enumerate(vhf.hessians!!)
        copyto!(M, _write(pM, rep_size, Y, (j,)), p, f(M, p, X))
    end
    return Y
end
# Part I(c) A single gradient function
function get_hessian!(
        M::AbstractManifold,
        Y,
        vhf::VectorHessianFunction{<:AllocatingEvaluation, FT, JT, <:FunctionVectorialType},
        p,
        X,
        i,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    ) where {FT, JT}
    n = _vgf_index_to_length(i, vhf.range_dimension)
    mP = PowerManifold(M, range, n)
    copyto!(mP, Y, vhf.hessians!!(M, p, X)[mP, i])
    return Y
end
function get_hessian!(
        M::AbstractManifold,
        Y,
        vhf::VectorHessianFunction{<:AllocatingEvaluation, FT, JT, <:FunctionVectorialType},
        p,
        X,
        i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    ) where {FT, JT}
    mP = PowerManifold(M, range, vhf.range_dimension)
    copyto!(M, Y, p, vhf.hessians!!(M, p, X)[mP, i])
    return Y
end
#
#
# Part II: in-place evaluations
# (a) a vector of functions
function get_hessian!(
        M::AbstractManifold,
        Y,
        vhf::VectorHessianFunction{<:InplaceEvaluation, FT, JT, <:ComponentVectorialType},
        p,
        X,
        i::Integer,
        (::Union{AbstractPowerRepresentation, Nothing}) = nothing,
    ) where {FT, JT}
    return vhf.hessians!![i](M, Y, p, X)
end
function get_hessian!(
        M::AbstractManifold,
        Y,
        vhf::VectorHessianFunction{<:InplaceEvaluation, FT, JT, <:ComponentVectorialType},
        p,
        X,
        i,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    ) where {FT, JT}
    n = _vgf_index_to_length(i, vhf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    # In the resulting X the indices are linear,
    # in jacobian[i] have the functions f are also given n a linear sense
    for (j, f) in zip(1:n, vhf.hessians!![i])
        f(M, _write(pM, rep_size, Y, (j,)), p, X)
    end
    return Y
end
# II(b) a single function
function get_hessian!(
        M::AbstractManifold,
        Y,
        vhf::VectorHessianFunction{<:InplaceEvaluation, FT, JT, <:FunctionVectorialType},
        p,
        X,
        i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    ) where {FT, JT}
    pM = PowerManifold(M, range, vhf.range_dimension...)
    P = fill(p, pM)
    y = zero_vector(pM, P)
    vhf.hessians!!(M, y, p, X)
    copyto!(M, Y, p, y[pM, i])
    return Y
end
function get_hessian!(
        M::AbstractManifold,
        Y,
        vhf::VectorHessianFunction{<:InplaceEvaluation, FT, JT, <:FunctionVectorialType},
        p,
        X,
        i,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    ) where {FT, JT}
    #Single access for function is a bit expensive
    n = _vgf_index_to_length(i, vhf.range_dimension)
    pM_out = PowerManifold(M, range, n)
    pM_temp = PowerManifold(M, range, vhf.range_dimension)
    P = fill(p, pM_temp)
    y = zero_vector(pM_temp, P)
    vhf.hessians!!(M, y, p, X)
    # Luckily all documented access functions work directly on `x[pM_temp,...]`
    copyto!(pM_out, Y, P[pM_temp, i], y[pM_temp, i])
    return Y
end

get_hessian_function(vgf::VectorHessianFunction, recursive::Bool = false) = vgf.hessians!!

#
#
# --- Jacobian - matrix representation

# A small helper function to change the basis of a Jacobian
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
        get_vector!(M, X, p, JF[i, :], from_basis)
        get_coordinates!(M, JF[i, :], p, X, to_basis)
    end
    return JF
end
# case we have the same basis: nothing to do, just return JF
function _change_basis!(
        M, JF, p, from_basis::B, to_basis_new::B; kwargs...
    ) where {B <: AbstractBasis}
    return JF
end

_doc_get_jacobian_matrix_vgf = """
    get_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p; kwargs...)
    get_jacobian!(M::AbstractManifold, J, vgf::AbstractVectorGradientFunction, p; kwargs...)

Return the Jacobian ``J_F(p): $(_math(:TangentSpace)) → ℝ^m`` of a [`AbstractVectorGradientFunction`](@ref) `vgf`,
i.e. a function ``F: $(_math(:Manifold)) → ℝ^m``, where `p ∈ $(_math(:Manifold))`, in matric form with respect to
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
get_jacobian(::AbstractManifold, ::AbstractVectorGradientFunction, p; kwargs...)
function get_jacobian! end
@doc "$(_doc_get_jacobian_matrix_vgf)"
get_jacobian!(M::AbstractManifold, JF, vgf::AbstractVectorGradientFunction, p)

function get_jacobian(
        M::AbstractManifold, vgf::VGF, p; basis::AbstractBasis = get_basis(vgf.jacobian_type), kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:AbstractVectorialType}}
    n = vgf.range_dimension
    # Can we avoid this allocation?
    c = get_coordinates(M, p, zero_vector(M, p), basis)
    JF = zeros(eltype(c), n, length(c))
    return get_jacobian!(M, JF, vgf, p; basis = basis, kwargs...)
end
# Part I: allocating vgf
# (a) We have a single gradient function
function get_jacobian!(
        M::AbstractManifold, J, vgf::VGF, p;
        basis::AbstractBasis = get_basis(vgf.jacobian_type), range::AbstractPowerRepresentation = get_range(vgf.jacobian_type),
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType}}
    n = vgf.range_dimension
    gradients = vgf.jacobian!!(M, p)
    mP = PowerManifold(M, range, vgf.range_dimension)
    for i in 1:n
        c = @view J[i, :]
        get_coordinates!(M, c, p, gradients[mP, i], basis)
    end
    return J
end
# (b) We have a vector of gradient functions
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p; basis::AbstractBasis = get_basis(vgf.jacobian_type), range = nothing,
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType}}
    for i in 1:(vgf.range_dimension)
        JF[i, :] .= get_coordinates(M, p, vgf.jacobian!![i](M, p), basis)
    end
    return JF
end
# (c) Jacobian function
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p; basis::AbstractBasis = get_basis(vgf.jacobian_type), range = nothing,
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:CoefficientVectorialType}}
    JF = vgf.jacobian!!(M, p)
    _change_basis!(M, JF, p, vgf.jacobian_type.basis, basis)
    return JF
end

# Part II: mutating vgf
# (a) We have a single gradient function
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p;
        basis::AbstractBasis = default_basis(M, typeof(p)), range::AbstractPowerRepresentation = get_range(vgf.jacobian_type),
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType}}
    mP = PowerManifold(M, range, vgf.range_dimension)
    gradients = zero_vector(mP, fill(p, mP))
    vgf.jacobian!!(M, gradients, p)
    for i in 1:(vgf.range_dimension)
        JF[i, :] .= get_coordinates(M, p, gradients[mP, i], basis)
    end
    return JF
end
# (b) a vector of functions
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p;
        basis = get_basis(vgf.jacobian_type), X = zero_vector(M, p), range = nothing,
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType},
    }
    for i in 1:(vgf.range_dimension)
        vgf.jacobian!![i](M, X, p)
        JF[i, :] .= get_coordinates(M, p, X, basis)
    end
    return JF
end
# (c) a matrix
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p; basis = get_basis(vgf.jacobian_type), range = nothing, X = nothing,
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType},
    }
    vgf.jacobian!!(M, JF, p)
    _change_basis!(M, JF, p, vgf.jacobian_type.basis, basis)
    return JF
end

#
#
# --- Jacobian function in terms of gradients as a 1-1 tensor (basis free) ---

_doc_get_jacobian_function_vector = """
    get_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, X; kwargs...)
    get_jacobian!(M::AbstractManifold, a, vgf::AbstractVectorGradientFunction, p, X; kwargs...)

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
Then it simplifies to a matric multiplication.

This can be computed in-place of `a`.
"""

@doc "$(_doc_get_jacobian_function_vector)"
get_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, X; kwargs...)

@doc "$(_doc_get_jacobian_function_vector)"
get_jacobian!(M::AbstractManifold, a, vgf::AbstractVectorGradientFunction, p, X; kwargs...)

# For the allocating one, we just need to allocate a
function get_jacobian(
        M::AbstractManifold, vgf::VGF, p, X; kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:AbstractVectorialType}}
    n = vgf.range_dimension
    a = zeros(eltype(X), n)
    return get_jacobian!(M, a, vgf, p, X; kwargs...)
end
# (a) a single function, allocating
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, X
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType}}
    mP = PowerManifold(M, get_range(vgf.jacobian_type), n)
    gradients = vgf.jacobian!!(M, p)
    for i in 1:n
        a[i] = inner(M, p, gradients[mP, i], X)
    end
    return a
end
# (b) vector of gradient functions
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, X
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType}}
    n = vgf.range_dimension
    for i in 1:n
        a[i] = inner(M, p, vgf.jacobian!![i](M, p), X)
    end
    return a
end
# (c) Jacobian function – easiest: Decompose X and call the other, for both
function get_jacobian!(
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:CoefficientVectorialType}, p, X;
        range = nothing
    ) where {FT}
    B = vgf.jacobian_type.basis
    return get_jacobian!(M, a, vgf, p, get_coordinates(M, p, X, B), B)
end
# II (a) Inplace single function – skip for now since allocation not so easy? we would need a power version of the point p
function get_jacobian!(
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType}, p, X;
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
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
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType}, p, X,
    ) where {FT}
    n = vgf.range_dimension
    Y = zero_vector(M, p)
    for i in 1:n
        vgf.jacobian!![i](M, Y, p)
        a[i] = inner(M, p, Y, X)
    end
    return a
end

# --- Jacobian function in terms of gradients as a 1-1 tensor in a basis (hence in matrix form) ---

_doc_get_jacobian_function_coord = """
    get_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, c, B::AbstractBasis; kwargs...)
    get_jacobian!(M::AbstractManifold, a, vgf::AbstractVectorGradientFunction, p, c, B::AbstractBasis; kwargs...)

Compute the Jacobian ``J_F(p)`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^m``,
to be precise how it acts on a tangent vector `X` at `p` on the manifold `M`, i.e., compute

````math
J_F(p)[X] = DF(p)[X] ∈ ℝ^m
````

where a basis ``$(_tex(:set, "Y_1,…,Y_n"))`` allows to decompose / provide the tangent vector
in coordincates ``c`` given by ``X = $(_tex(:displaystyle))$(_tex(:sum, "i=1", "d")) c_iY_i``
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

# Part I: allocating/inplace vgf – allocating/inplace work the same here jacobian (a) single gradient function
function get_jacobian(
        M::AbstractManifold, vgf::VGF, p, c, B::AbstractBasis; X = zero_vector(M, p), kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:FunctionVectorialType}}
    n = vgf.range_dimension
    a = zeros(eltype(X), n)
    return get_jacobian!(M, a, vgf, p, c, B; X = X, kwargs...)
end
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, c, B::AbstractBasis; X = zero_vector(M, p), kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:FunctionVectorialType}}
    # in this case it is easiest to reconstruct X and call the one for X
    get_vector!(M, X, p, c, B)
    return get_jacobian!(M, a, vgf, p, X, kwargs...)
end
# (b) vector of gradient functions
function get_jacobian(
        M::AbstractManifold, vgf::VGF, p, c, B::AbstractBasis, X = zero_vector(M, p), kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:ComponentVectorialType}}
    n = vgf.range_dimension
    a = zeros(eltype(X), n)
    return get_jacobian!(M, a, vgf, p, c, B; X = X, kwargs...)
end
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, c, B::AbstractBasis; X = zero_vector(M, p), kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:ComponentVectorialType}}
    # Here it is easiest to reconstruct X and call the variant for X
    get_vector!(M, X, p, c, B)
    return get_jacobian!(M, a, vgf, p, X; kwargs...)
end
# Part I: allocating VGF, (c) Jacobian function
function get_jacobian(
        M::AbstractManifold, vgf::VGF, p, c, B::AbstractBasis; X = nothing, kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:CoefficientVectorialType}}
    n = vgf.range_dimension
    a = zeros(eltype(X), n)
    return get_jacobian!(M, a, vgf, p, c, B; X = X, kwargs...)
end
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, c, B::AbstractBasis; kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:CoefficientVectorialType}}
    c2 = change_basis(M, p, c, B, vgf.jacobian_type.basis) # only allocates if basis changed
    a .= vgf.jacobian!!(M, p) * c
    return a
end
# Part II: mutating vgf
# (c) Jacobian function
function get_jacobian!(
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType}, p, c, B::AbstractBasis;
        X = nothing,
    ) where {FT}
    n = vgf.range_dimension
    JF = zeros(eltype(c), length(a), length(c))
    vgf.jacobian!!(M, JF, p)
    a .= JF * c
    return a
end

function get_jacobian_basis(vgf::AbstractVectorGradientFunction)
    return _get_jacobian_basis(vgf.jacobian_type)
end
_get_jacobian_basis(jt::AbstractVectorialType) = DefaultOrthonormalBasis()
_get_jacobian_basis(jt::CoefficientVectorialType) = jt.basis


#
#
# --- Adjoint Jacobian function in terms of gradients as a 1-1 tensor (basis free)

_doc_get_adjoint_jacobian_function_vector = """
    get_adjoint_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, a; kwargs...)
    get_adjoint_jacobian!(M::AbstractManifold, X, vgf::AbstractVectorGradientFunction, p, a; kwargs...)

Compute the adjoint Jacobian ``J_F^*(p)[a]`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^n``
``how it acts on a vector `a` at `p`, i.e., it is given by the relation

````math
$(_tex(:inner, "J_F^*(p)[a]", "X"; index = "p")) = $(_tex(:inner, "a", "J_F(p)[X]")),
````

where the inner product on the right hand side is the standard Euclidean inner product on ``ℝ^m``.

To be precise, the adjoint Jacobian is defined using the Riemannian gradients of the component functions
``F_i`` of ``F`` as

````math
J_F^*(p): ℝ^m → $(_math(:TangentSpace)),
$(_tex(:qquad))
J_F^*(p)[a] = $(_tex(:sum, "i=1", "m")) a_i $(_tex(:grad))F_i(p),
````

This can be computed in-place of `X`.

!!! note
    For the case of a matrix representation, i.e. the function signature
    `get_jacobian!(M, JF, vgf, p)` the resulting matrix can just be transposed to obtain the adjoint,
    if you used an [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`).
"""

@doc "$(_doc_get_adjoint_jacobian_function_vector)"
function get_adjoint_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, a; kwargs...)
    X = zero_vector(M, p)
    return get_adjoint_jacobian!(M, X, vgf, p, a; kwargs...)
end

@doc "$(_doc_get_adjoint_jacobian_function_vector)"
get_adjoint_jacobian!(M::AbstractManifold, X, vgf::AbstractVectorGradientFunction, p, a; kwargs...)

# Part I: allocating vgf (a) single gradient function
function get_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::VGF, p, a
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType},
    }
    n = vgf.range_dimension
    mP = PowerManifold(M, get_range(vgf.jacobian_type), n)
    gradients = vgf.jacobian!!(M, p)
    zero_vector!(M, X, p)
    for i in 1:n
        X .+= a[i] * gradients[mP, i]
    end
    return X
end
# (b) vector of gradient functions
function get_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::VGF, p, a
    ) where {
        FT,
        VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType},
    }
    n = vgf.range_dimension
    zero_vector!(M, X, p)
    for i in 1:n
        X .+= a[i] * vgf.jacobian!![i](M, p)
    end
    return X
end
# (c) Jacobian function
function get_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:CoefficientVectorialType}, p, a;
        basis::B = default_basis(M, typeof(p))
    ) where {FT, B <: AbstractBasis}
    n = vgf.range_dimension
    JF = vgf.jacobian!!(M, p)
    c = adjoint(JF) * a
    get_vector!(M, X, p, c, basis)
    return X
end

# Part II: mutating vgf (a) single gradient function
function get_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType}, p, a
    ) where {FT}
    n = vgf.range_dimension
    mP = PowerManifold(M, get_range(vgf.jacobian_type), n)
    gradients = zero_vector(mP, fill(p, mP))
    vgf.jacobian!!(M, gradients, p)
    zero_vector!(M, X, p)
    for i in 1:n
        X .+= a[i] * gradients[mP, i]
    end
    return X
end
# (b) vector of gradient functions
function get_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType}, p, a
    ) where {FT}
    n = vgf.range_dimension
    Y = zero_vector(M, p)
    zero_vector!(M, X, p)
    for i in 1:n
        vgf.jacobian!![i](M, Y, p)
        X .+= a[i] * Y
    end
    return X
end
# (c) Jacobian function
function get_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType}, p, a
    ) where {FT}
    n = vgf.range_dimension
    d = number_of_coordinates(vgf.jacobian_type.basis)
    J = zeros(n, d)
    get_vector!(M, X, p, adjoint(vgf.jacobian!!(M, J, p)) * a)
    return X
end

#
#
# --- Adjoint Jacobian function in terms of gradients as a matric-vector product in a basis

_doc_get_adjoint_jacobian_function_coeff = """
    get_adjoint_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, a, B::AbstractBasis; kwargs...)
    get_adjoint_jacobian!(M::AbstractManifold, c, vgf::AbstractVectorGradientFunction, p, a, B::AbstractBasis; kwargs...)

Compute the adjoint Jacobian ``J_F^*(p)[a]`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^n``
``how it acts on a vector `a` at `p`, i.e., it is given by the relation

````math
$(_tex(:inner, "J_F^*(p)[a]", "X"; index = "p")) = $(_tex(:inner, "a", "J_F(p)[X]")),
````

where the basis indicates that the result should be given in coordinates `c` with respect to that basis.
This can be done in-place of `c`.

Note that if `vgf` works in ternally in a basis different from the one provided, and additional change of basis is performed.
"""

@doc "$(_doc_get_adjoint_jacobian_function_coeff)"
function get_adjoint_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, a, B::AbstractBasis; X = zero_vector(M, p), kwargs...)
    c = get_coordinates(M, p, X, B)
    return get_adjoint_jacobian!(M, c, vgf, p, a; X = X, kwargs...)
end

@doc "$(_doc_get_adjoint_jacobian_function_coeff)"
get_adjoint_jacobian!(M::AbstractManifold, c, vgf::AbstractVectorGradientFunction, p, a, B::AbstractBasis; kwargs...)

# Part I: allocating vgf (a) single gradient function
function get_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::VGF, p, a, B::AbstractBasis; X = zero_vector(M, p)
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:FunctionVectorialType},
    }
    # easiest: call the one for the vector and decompose into c
    get_adjoint_jacobian!(M, X, vgf, p, a)
    return get_coordinates!(M, c, p, X, B)
end
# (b) vector of gradient functions - same as above
function get_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::VGF, p, a, B::AbstractBasis; X = zero_vector(M, p)
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:ComponentVectorialType},
    }
    get_adjoint_jacobian!(M, X, vgf, p, a)
    return get_coordinates!(M, c, p, X, B)
end
# I allocating, (c) Jacobian function
function get_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:CoefficientVectorialType}, p, a, B::AbstractBasis; X = nothing,
    ) where {FT}
    n = vgf.range_dimension
    JF = vgf.jacobian!!(M, p)
    c .= adjoint(JF) * a
    return change_basis!(M, c, p, c, vgf.jacobian_type.basis, B)
end
# mutating, (c) Jacobian function
function get_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType}, p, a, B::AbstractBasis; X = nothing
    ) where {FT}
    n = vgf.range_dimension
    J = zeros(eltype(c), n, length(c))
    return change_basis!(M, c, p, adjoint(vgf.jacobian!!(M, J, p)) * a, vgf.jacobian_type.basis, B)
end

#
#
# ---- Gradient
@doc """
    get_gradient(M::AbstractManifold, vgf::VectorGradientFunction, p, i)
    get_gradient(M::AbstractManifold, vgf::VectorGradientFunction, p, i, range)
    get_gradient!(M::AbstractManifold, X, vgf::VectorGradientFunction, p, i)
    get_gradient!(M::AbstractManifold, X, vgf::VectorGradientFunction, p, i, range)

Evaluate the gradients of the vector function `vgf` on the manifold `M` at `p` and
the values given in `range`, specifying the representation of the gradients.

Since `i` is assumed to be a linear index, you can provide
* a single integer
* a `UnitRange` to specify a range to be returned like `1:3`
* a `BitVector` specifying a selection
* a `AbstractVector{<:Integer}` to specify indices
* `:` to return the vector of all gradients
"""
get_gradient(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction,
    p,
    i,
    range::Union{AbstractPowerRepresentation, Nothing} = nothing,
)

# Generic case, allocate (a) a single tangent vector
function get_gradient(
        M::AbstractManifold,
        vgf::AbstractVectorGradientFunction,
        p,
        i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    )
    X = zero_vector(M, p)
    return get_gradient!(M, X, vgf, p, i, range)
end
# (b) UnitRange and AbstractVector allow to use length for BitVector its sum
function get_gradient(
        M::AbstractManifold,
        vgf::AbstractVectorGradientFunction,
        p,
        i = :, # as long as the length can be found it should work, see _vgf_index_to_length
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
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType},
        p,
        i::Integer,
        (::Union{AbstractPowerRepresentation, Nothing}) = nothing,
    ) where {FT <: AbstractVectorialType}
    return vgf.jacobian!![i](M, p)
end
function get_gradient(
        M::AbstractManifold,
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType},
        p,
        i::Integer,
        (::Union{AbstractPowerRepresentation, Nothing}) = nothing,
    ) where {FT <: AbstractVectorialType}
    X = zero_vector(M, p)
    return vgf.jacobian!![i](M, X, p)
end

#
#
# Part I: allocation
# I (a) Internally a Jacobian
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{
            <:AllocatingEvaluation, FT, <:CoefficientVectorialType,
        },
        p,
        i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    JF = vgf.jacobian!!(M, p)
    get_vector!(M, X, p, JF[i, :], vgf.jacobian_type.basis) #convert rows to gradients
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{
            <:AllocatingEvaluation, FT, <:CoefficientVectorialType,
        },
        p,
        i = :,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    JF = vgf.jacobian!!(M, p) # yields a full Jacobian
    for (j, k) in zip(_to_iterable_indices([JF[:, 1]...], i), 1:n)
        get_vector!(M, _write(pM, rep_size, X, (k,)), p, JF[j, :], vgf.jacobian_type.basis)
    end
    return X
end
# Part I(b) a vector of functions
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType},
        p,
        i::Integer,
        (::Union{AbstractPowerRepresentation, Nothing}) = nothing,
    ) where {FT <: AbstractVectorialType}
    return copyto!(M, X, p, vgf.jacobian!![i](M, p))
end
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType},
        p,
        i,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    # In the resulting `X` the indices  are linear,
    # in `jacobian[i]` the functions f are ordered in a linear sense
    for (j, f) in zip(1:n, vgf.jacobian!![i])
        copyto!(M, _write(pM, rep_size, X, (j,)), f(M, p))
    end
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType},
        p,
        i::Colon,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    for (j, f) in enumerate(vgf.jacobian!!)
        copyto!(M, _write(pM, rep_size, X, (j,)), p, f(M, p))
    end
    return X
end
# Part I(c) A single gradient function
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType},
        p,
        i,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    mP = PowerManifold(M, range, n)
    copyto!(mP, X, vgf.jacobian!!(M, p)[mP, i])
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType},
        p,
        i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    mP = PowerManifold(M, range, vgf.range_dimension)
    copyto!(M, X, p, vgf.jacobian!!(M, p)[mP, i])
    return X
end
#
#
# Part II: in-place evaluations
# (a) Jacobian
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType},
        p,
        i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    # a type wise safe way to allocate what usually should yield a n-times-d matrix
    pM = PowerManifold(M, range, vgf.range_dimension...)
    Y = zero_vector(pM, fill(p, pM))
    JF = reshape(
        get_coordinates(pM, fill(p, pM), Y, vgf.jacobian_type.basis),
        power_dimensions(pM)...,
        :,
    )
    vgf.jacobian!!(M, JF, p)
    get_vector!(M, X, p, JF[i, :], vgf.jacobian_type.basis)
    return X
end
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType},
        p,
        i,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    # a type wise safe way to allocate what usually should yield a n-times-d matrix
    pM = PowerManifold(M, range, vgf.range_dimension...)
    JF = reshape(
        get_coordinates(pM, fill(p, pM), X, vgf.jacobian_type.basis),
        power_dimensions(pM)...,
        :,
    )
    vgf.jacobian!!(M, JF, p)
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    for (j, k) in zip(_to_iterable_indices([JF[:, 1]...], i), 1:n)
        get_vector!(M, _write(pM, rep_size, X, (k,)), p, JF[j, :], vgf.jacobian_type.basis)
    end
    return X
end
#II (b) a vector of functions
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType},
        p,
        i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    return vgf.jacobian!![i](M, X, p)
end
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType},
        p,
        i,
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
        M::AbstractManifold,
        X,
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType},
        p,
        i::Integer,
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
    ) where {FT <: AbstractVectorialType, VGF <: AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType}}
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

get_gradient_function(vgf::VectorGradientFunction, recursive = false) = vgf.jacobian!!

#
#
# ---- Value
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
        M::AbstractManifold, vgf::AbstractVectorFunction{E, <:FunctionVectorialType}, p, i = :
    ) where {E <: AllocatingEvaluation}
    c = vgf.value!!(M, p)
    if isa(c, Number)
        return c
    else
        return c[i]
    end
end
function get_value(
        M::AbstractManifold,
        vgf::AbstractVectorFunction{E, <:ComponentVectorialType},
        p,
        i::Integer,
    ) where {E <: AbstractEvaluationType}
    return vgf.value!![i](M, p)
end
function get_value(
        M::AbstractManifold, vgf::AbstractVectorFunction{E, <:ComponentVectorialType}, p, i = :
    ) where {E <: AbstractEvaluationType}
    return [f(M, p) for f in vgf.value!![i]]
end
function get_value(
        M::AbstractManifold,
        vgf::AbstractVectorFunction{E, <:FunctionVectorialType},
        p,
        i = :;
        value_cache = zeros(vgf.range_dimension),
    ) where {E <: InplaceEvaluation}
    vgf.value!!(M, value_cache, p)
    return value_cache[i]
end

function get_value!(
        M::AbstractManifold,
        V,
        vgf::AbstractVectorFunction{AllocatingEvaluation, <:FunctionVectorialType},
        p,
        i = :,
    )
    c = vgf.value!!(M, p)
    V .= c[i]
    return V
end

function get_value!(
        M::AbstractManifold, V, vgf::AbstractVectorFunction{InplaceEvaluation, <:FunctionVectorialType}, p, i = :;
        value_cache = zeros(vgf.range_dimension),
    )
    vgf.value!!(M, value_cache, p)
    V .= value_cache[i]
    return V
end

function get_value!(
        M::AbstractManifold, V, vgf::AbstractVectorFunction{E, <:ComponentVectorialType}, p, i = :
    ) where {E <: AbstractEvaluationType}
    V .= [f(M, p) for f in vgf.value!![i]]
    return V
end


@doc """
    get_value_function(vgf::VectorGradientFunction, recursive=false)

return the internally stored function computing [`get_value`](@ref).
"""
function get_value_function(vgf::VectorGradientFunction, recursive = false)
    return vgf.value!!
end

@doc """
    length(vgf::AbstractVectorFunction)

Return the length of the vector the function ``f: $(_math(:Manifold)) → ℝ^n`` maps into,
that is the number `n`.
"""
Base.length(vgf::AbstractVectorFunction) = vgf.range_dimension
