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

@doc """
    VectorGradientFunction{FT, JT, F, J, I} <: AbstractVectorGradientFunction{E, FT, JT}

Represent a function ``f:$(_math(:Manifold)) → ℝ^n`` including it first derivative,
either as a vector of gradients of a Jacobian

And hence has a gradient ``$(_tex(:grad)) f_i(p) ∈ $(_math(:TangentSpace))``.
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
        FT <: AbstractVectorialType, JT <: AbstractVectorialType, F, J, I <: Integer,
    } <: AbstractVectorGradientFunction{FT, JT}
    value!!::F
    cost_type::FT
    jacobian!!::J
    jacobian_type::JT
    range_dimension::I
end
function VectorGradientFunction(
        f::F, Jf::J, range_dimension::I;
        function_type::FT = FunctionVectorialType(), jacobian_type::JT = FunctionVectorialType(),
    ) where {I <: Integer, F, J, FT <: AbstractVectorialType, JT <: AbstractVectorialType}
    return VectorGradientFunction{FT, JT, F, J, I}(f, function_type, Jf, jacobian_type, range_dimension)
end

@doc """
    VectorDifferentialFunction{FT, JT, AJT, F, J, A, I} <: AbstractFirstOrderVectorFunction{E, FT, JT}

Represent a function ``f:$(_math(:Manifold)) → ℝ^n`` including it first derivative information
as its differential, and optionally its adjoint differential.

All 3 can be either single functions [`FunctionVectorialType`](@ref) or vector of functions ([`ComponentVectorialType`](@ref))

# Fields

* `value!!::F`: the cost function ``f``, which can take different formats
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
        function_type::AbstractVectorialType=FunctionVectorialType(),
        jacobian_type::AbstractVectorialType=FunctionVectorialType(),
        range_dimension::Integer.
    )

    VectorGradientFunction(f, Jf, Jsf, range_dimension;
        function_type::AbstractVectorialType=FunctionVectorialType(),
        jacobian_type::AbstractVectorialType=FunctionVectorialType(),
        adjoint_jacobian_type::AbstractVectorialType=FunctionVectorialType(),
        range_dimension::Integer.
    )

Create a `VectorGradientFunction` of `f`  and its Jacobian `Jf`, and optionally its adjoint Jacobian.
If the adjoint is not provided, its type is also set to `Nothing`
"""
struct VectorDifferentialFunction{
        FT <: AbstractVectorialType, JT <: AbstractVectorialType, AT <: Union{<:AbstractVectorialType, Missing},
        F, J, A, I <: Integer,
    } <: AbstractFirstOrderVectorFunction{FT, JT}
    value!!::F
    cost_type::FT
    jacobian!!::J
    jacobian_type::JT
    adjoint_jacobian!!::A
    adjoint_jacobian_type::AT
    range_dimension::I
end
function VectorDifferentialFunction(
        f::F, Jf::J, range_dimension::I;
        function_type::FT = FunctionVectorialType(), jacobian_type::JT = FunctionVectorialType(),
    ) where {
        I <: Integer, F, J, FT <: AbstractVectorialType,
        JT <: AbstractVectorialType,
    }
    return VectorDifferentialFunction{FT, JT, Missing, F, J, Missing, I}(
        f, function_type, Jf, jacobian_type, missing, missing, range_dimension
    )
end
function VectorDifferentialFunction(
        f::F, Jf::J, AJf::A, range_dimension::I;
        function_type::FT = FunctionVectorialType(),
        jacobian_type::JT = FunctionVectorialType(), adjoint_jacobian_type::AJT = FunctionVectorialType(),
    ) where {
        I <: Integer, F, J, A, FT <: AbstractVectorialType,
        JT <: AbstractVectorialType, AJT <: Union{<:AbstractVectorialType, Missing},
    }
    return VectorDifferentialFunction{FT, JT, AJT, F, J, A, I}(
        f, function_type, Jf, jacobian_type, AJf, adjoint_jacobian_type, range_dimension
    )
end

function status_summary(vgf::VectorDifferentialFunction; context::Symbol = :default)
    _is_inline(context) && (return "A vectorial function including its differential $(length(vgf)) represented as $(vgf.cost_type) and its differential as $(vgf.jacobian_type) (adjoint: $(vgf.adjoint_jacobian_type))")
    return """
    A function defined on a manifold that maps into a vector space including its differential and the adjoint differential.

    ## Components
    * cost:             $(_MANOPT_INDENT)$(vgf.value!!)$(_MANOPT_INDENT)(as $(vgf.cost_type)),
    * Jacobian:         $(_MANOPT_INDENT)$(vgf.jacobian!!)$(_MANOPT_INDENT)(as $(vgf.jacobian_type))
    * adjoint Jacobian: $(_MANOPT_INDENT)$(vgf.adjoint_jacobian!!)$(_MANOPT_INDENT)(as $(vgf.adjoint_jacobian_type))
    * dimension:        $(_MANOPT_INDENT)$(length(vgf))"""
end
function show(io::IO, vgf::VectorDifferentialFunction)
    print(io, "VectorDifferentialFunction("); print(io, vgf.value!!); print(io, ", ")
    print(io, vgf.jacobian!!)
    if !ismissing(vgf.adjoint_jacobian!!)
        print(io, ", "); print(io, vgf.adjoint_jacobian!!)
    end
    print(io, ", "); print(io, vgf.range_dimension)
    print(io, "; ");
    print(io, ", function_type = "); print(io, vgf.cost_type)
    if !ismissing(vgf.adjoint_jacobian_type)
        print(io, ", adjoint_jacobian_type = "); print(io, vgf.adjoint_jacobian_type)
    end
    print(io, ", jacobian_type = ")
    return print(io, vgf.jacobian_type)
end

_doc_vhf = """
    VectorHessianFunction{FT, JT, HT, F, J, H, I} <: AbstractVectorGradientFunction{E, FT, JT}

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
        FT <: AbstractVectorialType, JT <: AbstractVectorialType, HT <: AbstractVectorialType,
        F, J, H, I <: Integer,
    } <: AbstractVectorGradientFunction{FT, JT}
    value!!::F
    cost_type::FT
    jacobian!!::J
    jacobian_type::JT
    hessians!!::H
    hessian_type::HT
    range_dimension::I
end

function VectorHessianFunction(
        f::F, Jf::J, Hf::H, range_dimension::I;
        function_type::FT = FunctionVectorialType(),
        jacobian_type::JT = FunctionVectorialType(), hessian_type::HT = FunctionVectorialType(),
    ) where {
        I <: Integer, F, J, H,
        FT <: AbstractVectorialType, JT <: AbstractVectorialType, HT <: AbstractVectorialType,
    }
    return VectorHessianFunction{FT, JT, HT, F, J, H, I}(
        f, function_type, Jf, jacobian_type, Hf, hessian_type, range_dimension
    )
end

_vgf_index_to_length(b::BitVector, n) = sum(b)
_vgf_index_to_length(::Colon, n) = n
_vgf_index_to_length(i::AbstractArray{<:Integer}, n) = length(i)
_vgf_index_to_length(r::UnitRange{<:Integer}, n) = length(r)

function status_summary(vhf::VectorHessianFunction; context::Symbol = :default)
    _is_inline(context) && (return "A vectorial function of length $(length(vhf)) including gradients and Hessians represented as $(vhf.cost_type), gradients as $(vhf.jacobian_type), and Hessians as $(vhf.hessian_type).")
    return """
    A function defined on a manifold that maps into a vector space including gradients and Hessians of the component functions.

    * cost:$(_MANOPT_INDENT)$(vhf.value!!)$(_MANOPT_INDENT)(represented as $(vhf.cost_type)),
    * gradient(s) or Jacobian:$(_MANOPT_INDENT)$(vhf.jacobian!!)$(_MANOPT_INDENT)(represented as $(vhf.jacobian_type))
    * Hessian(s):$(_MANOPT_INDENT)$(vhf.hessians!!)$(_MANOPT_INDENT)(represented as $(vhf.hessian_type))
    * dimension:$(_MANOPT_INDENT)$(length(vhf))"""
end
function show(io::IO, vhf::VectorHessianFunction)
    print(io, "VectorGradientFunction("); print(io, vhf.value!!); print(io, ", ")
    print(io, vhf.jacobian!!); print(io, ", "); print(io, vhf.hessians!!); print(io, ", ")
    print(io, vhf.range_dimension); print(io, "; ");
    print(io, ", function_type = "); print(io, vhf.cost_type)
    print(io, ", jacobian_type = "); print(io, vhf.jacobian_type)
    print(io, ", hessian_type = ")
    return print(io, vhf.hessian_type)
end
