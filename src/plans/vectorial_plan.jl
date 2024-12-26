@doc """
    AbstractVectorialType

An abstract type for different representations of a vectorial function
    ``f: $(_math(:M)) → ℝ`` and its (component-wise) gradient/Jacobian
"""
abstract type AbstractVectorialType end

@doc raw"""
    CoordinateVectorialType{B<:AbstractBasis} <: AbstractVectorialType

A type to indicate that gradient of the constraints is implemented as a
Jacobian matrix with respect to a certain basis, that is if the vectfor function
are is ``f: \mathcal M → ℝ^m`` with and we have a asis ``\mathcal B`` of ``T_p\mathcal M``, at ``p∈ \mathcal M``
This can be written as ``J_g(p) = (c_1^{\mathrm{T}},…,c_m^{\mathrm{T}})^{\mathrm{T}} \in ℝ^{m,d}``, that is,
every row ``c_i`` of this matrix is a set of coefficients such that
`get_coefficients(M, p, c, B)` is the tangent vector ``\oepratorname{grad} g_i(p)``
for example ``g_i(p) ∈ ℝ^m`` or ``\operatorname{grad} g_i(p) ∈ T_p\mathcal M``,
    ``i=1,…,m``.

# Fields

* `basis` an [`AbstractBasis`](@extref `ManifoldsBase.AbstractBasis`) to indicate the default representation.
"""
struct CoordinateVectorialType{B<:AbstractBasis} <: AbstractVectorialType
    basis::B
end

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

@doc raw"""
    ComponentVectorialType <: AbstractVectorialType

A type to indicate that constraints are implemented as component functions,
for example ``g_i(p) ∈ ℝ^m`` or ``\operatorname{grad} g_i(p) ∈ T_p\mathcal M``, ``i=1,…,m``.
"""
struct ComponentVectorialType <: AbstractVectorialType end

@doc raw"""
    FunctionVectorialType <: AbstractVectorialType

 A type to indicate that constraints are implemented one whole functions,
for example ``g(p) ∈ ℝ^m`` or ``\operatorname{grad} g(p) ∈ (T_p\mathcal M)^m``.

This type internally stores the [`AbstractPowerRepresentation`](@ref),
when it makes sense, especially for Hessian and gradient functions.
"""
struct FunctionVectorialType <: AbstractVectorialType end

# TODO store range type in this type instead of having a keyword argument in so many function.

@doc raw"""
    AbstractVectorFunction{E, FT} <: Function

Represent an abstract vectorial function ``f:\mathcal M → ℝ^n`` with an
[`AbstractEvaluationType`](@ref) `E` and an [`AbstractVectorialType`](@ref) to specify the
format ``f`` is implemented as.

# Representations of ``f``

There are three different representations of ``f``, which might be beneficial in one or
the other situation:
* the [`FunctionVectorialType`](@ref) storing a single function ``f`` that returns a vector,
* the [`ComponentVectorialType`](@ref) storing a vector of functions ``f_i`` that return a single value each,
* the [`CoordinateVectorialType`](@ref) storing functions with respect to a specific basis of the tangent space for gradients and Hessians.
  Gradients of this type are usually referred to as Jacobians.

For the [`ComponentVectorialType`](@ref) imagine that ``f`` could also be written
using its component functions,

```math
f(p) = \bigl( f_1(p), f_2(p), \ldots, f_n(p) \bigr)^{\mathrm{T}}
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
abstract type AbstractVectorFunction{E<:AbstractEvaluationType,FT<:AbstractVectorialType} <:
              Function end

@doc raw"""
    VectorGradientFunction{E, FT, JT, F, J, I} <: AbstractManifoldObjective{E}

Represent an abstract vectorial function ``f:\mathcal M → ℝ^n`` that provides a (component wise)
gradient.
The [`AbstractEvaluationType`](@ref) `E` indicates the evaluation type,
and the [`AbstractVectorialType`](@ref)s `FT` and `JT` the formats in which
the function and the gradient are provided, see [`AbstractVectorFunction`](@ref) for an explanation.
"""
abstract type AbstractVectorGradientFunction{
    E<:AbstractEvaluationType,FT<:AbstractVectorialType,JT<:AbstractVectorialType
} <: AbstractVectorFunction{E,FT} end

@doc """
    VectorGradientFunction{E, FT, JT, F, J, I} <: AbstractVectorGradientFunction{E, FT, JT}

Represent a function ``f:$(_math(:M)) → ℝ^n`` including it first derivative,
either as a vector of gradients of a Jacobian

And hence has a gradient ``$(_tex(:grad)) f_i(p) ∈ $(_math(:TpM))`.
Putting these gradients into a vector the same way as the functions, yields a
[`ComponentVectorialType`](@ref)

```math
$(_tex(:grad)) f(p) = $(_tex(:Bigl))( $(_tex(:grad)) f_1(p), $(_tex(:grad)) f_2(p), …, $(_tex(:grad)) f_n(p) $(_tex(:Bigr)))^$(_tex(:transp))
∈ ($(_math(:TpM)))^n
```

And advantage here is, that again the single components can be evaluated individually

# Fields

* `value!!::F`:          the cost function ``f``, which can take different formats
* `cost_type::`[`AbstractVectorialType`](@ref):     indicating / string data for the type of `f`
* `jacobian!!::G`:     the Jacobian of ``f``
* `jacobian_type::`[`AbstractVectorialType`](@ref): indicating / storing data for the type of ``J_f``
* `parameters`:    the number `n` from, the size of the vector ``f`` returns.

# Constructor

    VectorGradientFunction(f, Jf, range_dimension;
        evaluation::AbstractEvaluationType=AllocatingEvaluation(),
        function_type::AbstractVectorialType=FunctionVectorialType(),
        jacobian_type::AbstractVectorialType=FunctionVectorialType(),
    )

Create a `VectorGradientFunction` of `f`  and its Jacobian (vector of gradients) `Jf`,
where `f` maps into the Euclidean space of dimension `range_dimension`.
Their types are specified by the `function_type`, and `jacobian_type`, respectively.
The Jacobian can further be given as an allocating variant or an in-place variant, specified
by the `evaluation=` keyword.
"""
struct VectorGradientFunction{
    E<:AbstractEvaluationType,
    FT<:AbstractVectorialType,
    JT<:AbstractVectorialType,
    F,
    J,
    I<:Integer,
} <: AbstractVectorGradientFunction{E,FT,JT}
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
    evaluation::E=AllocatingEvaluation(),
    function_type::FT=FunctionVectorialType(),
    jacobian_type::JT=FunctionVectorialType(),
) where {
    I<:Integer,
    F,
    J,
    E<:AbstractEvaluationType,
    FT<:AbstractVectorialType,
    JT<:AbstractVectorialType,
}
    return VectorGradientFunction{E,FT,JT,F,J,I}(
        f, function_type, Jf, jacobian_type, range_dimension
    )
end

_doc_vhf = """
    VectorHessianFunction{E, FT, JT, HT, F, J, H, I} <: AbstractVectorGradientFunction{E, FT, JT}

Represent a function ``f:$(_math(:M)) M → ℝ^n`` including it first derivative,
either as a vector of gradients of a Jacobian, and the Hessian,
as a vector of Hessians of the component functions.

Both the Jacobian and the Hessian can map into either a sequence of tangent spaces
or a single tangent space of the power manifold of lenth `n`.

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
    E<:AbstractEvaluationType,
    FT<:AbstractVectorialType,
    JT<:AbstractVectorialType,
    HT<:AbstractVectorialType,
    F,
    J,
    H,
    I<:Integer,
} <: AbstractVectorGradientFunction{E,FT,JT}
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
    evaluation::E=AllocatingEvaluation(),
    function_type::FT=FunctionVectorialType(),
    jacobian_type::JT=FunctionVectorialType(),
    hessian_type::HT=FunctionVectorialType(),
) where {
    I<:Integer,
    F,
    J,
    H,
    E<:AbstractEvaluationType,
    FT<:AbstractVectorialType,
    JT<:AbstractVectorialType,
    HT<:AbstractVectorialType,
}
    return VectorHessianFunction{E,FT,JT,HT,F,J,H,I}(
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
@doc raw"""
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
* `:` to return the vector of all gradients
"""
get_hessian(
    M::AbstractManifold,
    vgf::VectorHessianFunction,
    p,
    X,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=nothing,
)

# Generic case, allocate (a) a single tangent vector
function get_hessian(
    M::AbstractManifold,
    vhf::VectorHessianFunction,
    p,
    X,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
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
    i=:, # as long as the length can be found it should work, see _vgf_index_to_length
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
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
    vhf::VectorHessianFunction{<:AllocatingEvaluation,FT,JT,<:ComponentVectorialType},
    p,
    X,
    i::Integer,
    ::Union{AbstractPowerRepresentation,Nothing}=nothing,
) where {FT,JT}
    return copyto!(M, Y, p, vhf.hessians!![i](M, p, X))
end
function get_hessian!(
    M::AbstractManifold,
    Y,
    vhf::VectorHessianFunction{<:AllocatingEvaluation,FT,JT,<:ComponentVectorialType},
    p,
    X,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT,JT}
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
    vgf::VectorHessianFunction{<:AllocatingEvaluation,FT,JT,<:ComponentVectorialType},
    p,
    X,
    i::Colon,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT,JT}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    for (j, f) in enumerate(vgf.hessians!!)
        copyto!(M, _write(pM, rep_size, Y, (j,)), p, f(M, p, X))
    end
    return Y
end
# Part I(c) A single gradient function
function get_hessian!(
    M::AbstractManifold,
    Y,
    vhf::VectorHessianFunction{<:AllocatingEvaluation,FT,JT,<:FunctionVectorialType},
    p,
    X,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT,JT}
    n = _vgf_index_to_length(i, vhf.range_dimension)
    mP = PowerManifold(M, range, n)
    copyto!(mP, Y, vhf.hessians!!(M, p, X)[mP, i])
    return Y
end
function get_hessian!(
    M::AbstractManifold,
    Y,
    vhf::VectorHessianFunction{<:AllocatingEvaluation,FT,JT,<:FunctionVectorialType},
    p,
    X,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT,JT}
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
    vhf::VectorHessianFunction{<:InplaceEvaluation,FT,JT,<:ComponentVectorialType},
    p,
    X,
    i::Integer,
    ::Union{AbstractPowerRepresentation,Nothing}=nothing,
) where {FT,JT}
    return vhf.hessians!![i](M, Y, p, X)
end
function get_hessian!(
    M::AbstractManifold,
    Y,
    vhf::VectorHessianFunction{<:InplaceEvaluation,FT,JT,<:ComponentVectorialType},
    p,
    X,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT,JT}
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
    vhf::VectorHessianFunction{<:InplaceEvaluation,FT,JT,<:FunctionVectorialType},
    p,
    X,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT,JT}
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
    vhf::VectorHessianFunction{<:InplaceEvaluation,FT,JT,<:FunctionVectorialType},
    p,
    X,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT,JT}
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

get_hessian_function(vgf::VectorHessianFunction, recursive::Bool=false) = vgf.hessians!!

#
#
# ---- Jacobian
_doc_get_jacobian = """
    get_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p; basis = get_javobian_basis(vgf) )
    get_jacobian!(M::AbstractManifold, J, vgf::AbstractVectorGradientFunction, p; basis = get_javobian_basis(vgf) )

Evaluate the Jacobian of the vector function `vgf` on the manifold `M` at `p`.
Let ``n`` be the manifold dimension of `M`.
Representing every gradient ``$(_tex(:grad)) f_i(p) ∈ $(_math(:TpM))` as a vector of coefficients
``c_i = (c_{i,j})_{j=0}^n`` with respect to some basis, the Jacobian is given by

```math
 J_f = $(_tex(:pmatrix,
    "c_{1,1} & $(_tex(:cdots)) & c_{1,n}",
    "$(_tex(:vdots)) & $(_tex(:ddots)) & $(_tex(:vdots))",
    "c_{n,1} & $(_tex(:cdots)) & c_{n,n}"
    )) ∈ ℝ^{n×n},
```

in other words, the Jacobian consists of the gradients in coordinates stored row-wise.

For the [`ComponentVectorialType`](@ref) and the [`FunctionVectorialType`](@ref),
this is done in the tangent space basis provided by `basis=` directly.
For the [`CoordinateVectorialType`](@ref) the Jacobian is computed in the basis of this type,
if the basis does not agree with the `basis=`, a change of basis is performed

# Keyword Arguments
* `basis` basis with respect to which the Jacobian is computed.
* `range` for the [`FunctionVectorialType`](@ref) specify the range type of the resulting gradient,
  that is, the [`AbstractPowerRepresentation`](@extref `ManifoldsBase.AbstractPowerRepresentation`) the resulting power manifold has.
"""

function get_jacobian end
@doc "$(_doc_get_jacobian)"
get_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p)

# A small helper function to change the basis of a Jacobian
"""
    _change_basis!(M::AbstractManifold, p, JF, from_basis::B1, to_basis::B; X=zero_vector(M,p))

Given a jacobian matrix `JF` on a manifold `M` at `p` with respect to the `from_basis`
in the tangent space of `p` on `M`. Change the basis of the Jacobian to `to_basis` in place of `JF`.

# Keyword Arguments
* `X` a temporary vector to store a generated vector, before decomposing it again with respect to the new basis
"""
function _change_basis!(
    M, p, JF, from_basis::B1, to_basis::B2; X=zero_vector(M, p)
) where {B1<:AbstractBasis,B2<:AbstractBasis}
    # change every row to new basis
    for i in 1:size(JF, 1) # every row
        get_vector!(M, X, p, JF[i, :], from_basis)
        get_coordinates!(M, JF[i, :], p, X, to_basis)
    end
    return JF
end
# case we have the same basis: nothing to do, just return JF
_change_basis!(M, p, JF, from_basis::B, to_basis_new) where {B<:AbstractBasis} = JF

# Part I: allocating vgf – allocating jacobian
# (a) We have a single gradient function
function get_jacobian(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation,FT,<:FunctionVectorialType},
    p;
    basis::B=DefaultOrthonormalBasis(),
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT,B<:AbstractBasis}
    n = vgf.range_dimension
    d = manifold_dimension(M, p)
    gradients = vgf.jacobian!!(M, p)
    mP = PowerManifold(M, range, vgf.range_dimension)
    # generate the first row to get an eltype
    c1 = get_coordinates(M, p, gradients[mP, 1], basis)
    JF = zeros(eltype(c1), n, d)
    JF[1, :] .= c1
    for i in 2:n
        JF[i, :] .= get_coordinates(M, p, gradients[mP, i], basis)
    end
    return JF
end
# (b) We have a vector of gradient functions
function get_jacobian(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation,FT,<:ComponentVectorialType},
    p;
    basis=get_jacobian_basis(vgf),
    kwargs...,
) where {FT}
    n = vgf.range_dimension
    d = manifold_dimension(M, p)
    # generate the first row to get an eltype
    c1 = get_coordinates(M, p, vgf.jacobian!![1](M, p), basis)
    JF = zeros(eltype(c1), n, d)
    JF[1, :] .= c1
    for i in 2:n
        JF[i, :] .= get_coordinates(M, p, vgf.jacobian!![i](M, p), basis)
    end
    return JF
end
# (c) We have a Jacobian function
function get_jacobian(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction{
        <:AllocatingEvaluation,FT,<:CoordinateVectorialType
    },
    p;
    basis=get_jacobian_basis(vgf),
    kwargs...,
) where {FT}
    JF = vgf.jacobian!!(M, p)
    _change_basis!(JF, vgf.jacobian_type.basis, basis)
    return JF
end

# Part II: mutating vgf – allocating jacobian
# (a) We have a single gradient function
function get_jacobian(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:FunctionVectorialType},
    p;
    basis::B=DefaultOrthonormalBasis(),
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT,B<:AbstractBasis}
    n = vgf.range_dimension
    d = manifold_dimension(M, p)
    mP = PowerManifold(M, range, vgf.range_dimension)
    gradients = zero_vector(mP, fill(p, pM))
    vgf.jacobian!!(M, gradients, p)
    # generate the first row to get an eltype
    c1 = get_coordinates(M, p, gradients[mP, 1], basis)
    JF = zeros(eltype(c1), n, d)
    JF[1, :] .= c1
    for i in 2:n
        JF[i, :] .= get_coordinates(M, p, gradients[mP, i], basis)
    end
    return JF
end
# (b) We have a vector of gradient functions
function get_jacobian(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:ComponentVectorialType},
    p;
    basis=get_jacobian_basis(vgf),
) where {FT}
    n = vgf.range_dimension
    d = manifold_dimension(M, p)
    # generate the first row to get an eltype
    X = zero_vector(M, p)
    vgf.jacobian!![1](M, X, p)
    c1 = get_coordinates(M, p, X, basis)
    JF = zeros(eltype(c1), n, d)
    JF[1, :] .= c1
    for i in 2:n
        vgf.jacobian!![i](M, X, p)
        JF[i, :] .= get_coordinates(M, p, X, basis)
    end
    return JF
    JF = vgf.jacobian!!(M, p)
    _change_basis!(JF, vgf.jacobian_type.basis, basis)
    return JF
end
# (c) We have a Jacobian function
function get_jacobian(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:CoordinateVectorialType},
    p;
    basis=get_jacobian_basis(vgf),
) where {FT}
    c = get_coordinats(M, p, zero_vector(M, p))
    JF = zeros(eltype(c), vgf.range_dimension, manifold_dimension(M, p))
    vgf.jacobian!!(M, JF, p)
    _change_basis!(JF, vgf.jacobian_type.basis, basis)
    return JF
end

function get_jacobian! end
@doc "$(_doc_get_jacobian)"
get_jacobian!(M::AbstractManifold, JF, vgf::AbstractVectorGradientFunction, p)

# Part I: allocating vgf – inplace jacobian
# (a) We have a single gradient function
function get_jacobian!(
    M::AbstractManifold,
    JF,
    vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation,FT,<:FunctionVectorialType},
    p;
    basis::B=DefaultOrthonormalBasis(),
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT,B<:AbstractBasis}
    n = vgf.range_dimension
    d = manifold_dimension(M, p)
    gradients = vgf.jacobian!!(M, p)
    mP = PowerManifold(M, range, vgf.range_dimension)
    for i in 1:n
        JF[i, :] .= get_coordinates(M, p, gradients[mP, i], basis)
    end
    return JF
end
# (b) We have a vector of gradient functions
function get_jacobian!(
    M::AbstractManifold,
    JF,
    vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation,FT,<:ComponentVectorialType},
    p;
    basis=get_jacobian_basis(vgf),
) where {FT}
    for i in 1:(vgf.range_dimension)
        JF[i, :] .= get_coordinates(M, p, vgf.jacobian!![i](M, p), basis)
    end
    return JF
end
# (c) We have a Jacobian function
function get_jacobian!(
    M::AbstractManifold,
    JF,
    vgf::AbstractVectorGradientFunction{
        <:AllocatingEvaluation,FT,<:CoordinateVectorialType
    },
    p;
    basis=get_jacobian_basis(vgf),
) where {FT}
    JF .= vgf.jacobian!!(M, p)
    _change_basis!(JF, vgf.jacobian_type.basis, basis)
    return JF
end

# Part II: mutating vgf – allocating jacobian
# (a) We have a single gradient function
function get_jacobian!(
    M::AbstractManifold,
    JF,
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:FunctionVectorialType},
    p;
    basis::B=DefaultOrthonormalBasis(),
) where {FT,B<:AbstractBasis}
    gradients = zero_vector(mP, fill(p, pM))
    vgf.jacobian!!(M, gradients, p)
    for i in 1:n
        JF[i, :] .= get_coordinates(M, p, gradients[mP, i], basis)
    end
    return JF
end
# (b) We have a vector of gradient functions
function get_jacobian!(
    M::AbstractManifold,
    JF,
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:ComponentVectorialType},
    p;
    basis=get_jacobian_basis(vgf),
    X=zero_vector(M, p),
) where {FT}
    for i in 1:n
        vgf.jacobian!![i](M, X, p)
        JF[i, :] .= get_coordinates(M, p, X, basis)
    end
    return JF
    JF = vgf.jacobian!!(M, p)
    _change_basis!(JF, vgf.jacobian_type.basis, basis)
    return JF
end
# (c) We have a Jacobian function
function get_jacobian!(
    M::AbstractManifold,
    JF,
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:CoordinateVectorialType},
    p;
    basis=get_jacobian_basis(vgf),
) where {FT}
    vgf.jacobian!!(M, JF, p)
    _change_basis!(JF, vgf.jacobian_type.basis, basis)
    return JF
end

get_jacobian_basis(vgf::AbstractVectorGradientFunction) = DefaultBasis()
function get_jacobian_basis(
    vgf::AbstractVectorGradientFunction{F,G,<:CoordinateVectorialType}
) where {F,G}
    return vgf.jacobian_type.basis
end

#
#
# ---- Gradient
@doc raw"""
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
    range::Union{AbstractPowerRepresentation,Nothing}=nothing,
)

# Generic case, allocate (a) a single tangent vector
function get_gradient(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction,
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
)
    X = zero_vector(M, p)
    return get_gradient!(M, X, vgf, p, i, range)
end
# (b) UnitRange and AbstractVector allow to use length for BitVector its sum
function get_gradient(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction,
    p,
    i=:, # as long as the length can be found it should work, see _vgf_index_to_length
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
)
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    P = fill(p, pM)
    X = zero_vector(pM, P)
    return get_gradient!(M, X, vgf, p, i, range)
end
# (c) Special cases where allocations can be skipped
function get_gradient(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation,FT,<:ComponentVectorialType},
    p,
    i::Integer,
    ::Union{AbstractPowerRepresentation,Nothing}=nothing,
) where {FT<:AbstractVectorialType}
    return vgf.jacobian!![i](M, p)
end
function get_gradient(
    M::AbstractManifold,
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:ComponentVectorialType},
    p,
    i::Integer,
    ::Union{AbstractPowerRepresentation,Nothing}=nothing,
) where {FT<:AbstractVectorialType}
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
        <:AllocatingEvaluation,FT,<:CoordinateVectorialType
    },
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT<:AbstractVectorialType}
    JF = vgf.jacobian!!(M, p)
    get_vector!(M, X, p, JF[i, :], vgf.jacobian_type.basis) #convert rows to gradients
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::AbstractVectorGradientFunction{
        <:AllocatingEvaluation,FT,<:CoordinateVectorialType
    },
    p,
    i=:,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT<:AbstractVectorialType}
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
    vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation,FT,<:ComponentVectorialType},
    p,
    i::Integer,
    ::Union{AbstractPowerRepresentation,Nothing}=nothing,
) where {FT<:AbstractVectorialType}
    return copyto!(M, X, p, vgf.jacobian!![i](M, p))
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation,FT,<:ComponentVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
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
    vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation,FT,<:ComponentVectorialType},
    p,
    i::Colon,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT<:AbstractVectorialType}
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
    vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation,FT,<:FunctionVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT<:AbstractVectorialType}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    mP = PowerManifold(M, range, n)
    copyto!(mP, X, vgf.jacobian!!(M, p)[mP, i])
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation,FT,<:FunctionVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT<:AbstractVectorialType}
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
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:CoordinateVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT<:AbstractVectorialType}
    # a type wise safe way to allocate what usually should yield a n-times-d matrix
    pM = PowerManifold(M, range, vgf.range_dimension...)
    P = fill(p, pM)
    Y = zero_vector(pM, P)
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
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:CoordinateVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT<:AbstractVectorialType}
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
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:ComponentVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=nothing,
) where {FT<:AbstractVectorialType}
    return vgf.jacobian!![i](M, X, p)
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:ComponentVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT<:AbstractVectorialType}
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
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:FunctionVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT<:AbstractVectorialType}
    pM = PowerManifold(M, range, vgf.range_dimension...)
    P = fill(p, pM)
    x = zero_vector(pM, P)
    vgf.jacobian!!(M, x, p)
    copyto!(M, X, p, x[pM, i])
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::AbstractVectorGradientFunction{<:InplaceEvaluation,FT,<:FunctionVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT<:AbstractVectorialType}
    #Single access for function is a bit expensive
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

get_gradient_function(vgf::VectorGradientFunction, recursive=false) = vgf.jacobian!!

#
#
# ---- Value
@doc raw"""
    get_value(M::AbstractManifold, vgf::AbstractVectorFunction, p[, i=:])

Evaluate the vector function [`VectorGradientFunction`](@ref) `vgf` at `p`.
The `range` can be used to specify a potential range, but is currently only present for consistency.

The `i` can be a linear index, you can provide

* a single integer
* a `UnitRange` to specify a range to be returned like `1:3`
* a `BitVector` specifying a selection
* a `AbstractVector{<:Integer}` to specify indices
* `:` to return the vector of all gradients, which is also the default
"""
get_value(M::AbstractManifold, vgf::AbstractVectorFunction, p, i)
function get_value(
    M::AbstractManifold, vgf::AbstractVectorFunction{E,<:FunctionVectorialType}, p, i=:
) where {E<:AbstractEvaluationType}
    c = vgf.value!!(M, p)
    if isa(c, Number)
        return c
    else
        return c[i]
    end
end
function get_value(
    M::AbstractManifold,
    vgf::AbstractVectorFunction{E,<:ComponentVectorialType},
    p,
    i::Integer,
) where {E<:AbstractEvaluationType}
    return vgf.value!![i](M, p)
end
function get_value(
    M::AbstractManifold, vgf::AbstractVectorFunction{E,<:ComponentVectorialType}, p, i=:
) where {E<:AbstractEvaluationType}
    return [f(M, p) for f in vgf.value!![i]]
end

@doc raw"""
    get_value_function(vgf::VectorGradientFunction, recursive=false)

return the internally stored function computing [`get_value`](@ref).
"""
function get_value_function(vgf::VectorGradientFunction, recursive=false)
    return vgf.value!!
end

@doc raw"""
    length(vgf::AbstractVectorFunction)

Return the length of the vector the function ``f: \mathcal M → ℝ^n`` maps into,
that is the number `n`.
"""
Base.length(vgf::AbstractVectorFunction) = vgf.range_dimension
