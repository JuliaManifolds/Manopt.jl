@doc raw"""
    AbstractVectorialType

An abstract type for different representations of a vectorial function
    ``f: \mathcal → \mathbb R^m`` and its (component-wise) gradient/Jacobian
"""
abstract type AbstractVectorialType end

@doc raw"""
    CoefficientVectorialType{B<:AbstractBasis} <: AbstractVectorialType

    A type to indicate that gradient of the constraints is implemented as a
Jacobian matrix with respect to a certain basis, that is if we have constraints
``g: \mathcal M → ℝ^m`` and a basis ``\mathcal B`` of ``T_p\mathcal M``, at ``p∈ \mathcal M``

We have ``J_g(p) = (c_1^{\mathrm{T)},…,c_m^{\mathrm{T)})^{\mathrm{T)} \in ℝ^{m,d}``, that is,
every row ``c_i`` of this matrix is a set of coefficients such that
`get_coefficients(M, p, c, B)` is the tangent vector ``\oepratorname{grad} g_i(p)``

for example ``g_i(p) ∈ ℝ^m`` or ``\operatorname{grad} g_(p) ∈ T_p\mathcal M``,
    ``i=1,…,m``.

# Fields

* `basis` an [`AbstractBasis`](@extref) to indicate the default representation.
"""
struct CoefficientVectorialType{B<:AbstractBasis} <: AbstractVectorialType
    basis::B
end

@doc raw"""
    ComponentVectorialType <: AbstractVectorialType

A type to indicate that constraints are implemented as component functions,
for example ``g_i(p) ∈ ℝ^m`` or ``\operatorname{grad} g_(p) ∈ T_p\mathcal M``, ``i=1,…,m``.
"""
struct ComponentVectorialType <: AbstractVectorialType end

@doc raw"""
    FunctionVectorialType <: AbstractVectorialType

 A type to indicate that constraints are implemented one whole functions,
for example ``g(p) ∈ ℝ^m`` or ``\operatorname{grad} g(p) ∈ (T_p\mathcal M)^m``.
"""
struct FunctionVectorialType <: AbstractVectorialType end

@doc raw"""
    VectorGradientFunction{E, FT, JT, F, J, I} <: <: AbstractManifoldObjective{E}

Represent a function ``f:\mathcal M → ℝ^n`` including it first derivatie,
either as a vector of gradients of a Jacobian

# Representations of ``f``

There are htree different representations of ``f``, which might be benefictial in one or
the other situation:
* the [`FunctionVectorialType`](@ref),
* the [`ComponentVectorialType`](@ref),
* the [`CoefficientVectorialType`](@ref) with respect to a specific basis of the tangent space.

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

For the [`ComponentVectorialType`](@ref) of `f`, each of the component functions
is a (classical) objective and hence has a gradient ``\oepratorname{grad} f_i(p) ∈ T_p\mathcal M``.
Putting these gradients into a vector the same way as the functions, we obtain a
[`ComponentVectorialType`](@ref)

```math
\operatorname{grad} f(p) = \Bigl( \operatorname{grad} f_1(p), \operatorname{grad} f_2(p), …, \operatorname{grad} f_n(p) \Bigr)^{\mathrm{T}}
∈ (T_p\mathcal M)^n
```

And advantage here is, that again the single components can be evaluated individually

# Fields

* `costs!!`:          the cost function ``f``, which can take different formats
* `cost_type`:     indicating / string data for the type of `f`
* `jacobian!!:     the jacobian of ``f``
* `jacobian_type`: indicating / storing data for the type of ``J_f``
* `parameters`:    the nunmber `n` from above, that is the size of the vector ``f`` returns.

# Constructor

VectorGradientFunction(f, Jf, range_dimension;
    evaluation::AbstractEvaluationType=AllocatingEvaluation(),
    function_type::AbstractVectorialType=FunctionVectorialType(),
    jacobian_type::AbstractVectorialType=FunctionVectorialType(),
)
"""
struct VectorGradientFunction{
    E<:AbstractEvaluationType,
    FT<:AbstractVectorialType,
    JT<:AbstractVectorialType,
    F,
    J,
    I<:Integer,
} <: Function
    costs!!::F
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
    I,F,J,E<:AbstractEvaluationType,FT<:AbstractVectorialType,JT<:AbstractVectorialType
}
    return VectorGradientFunction{E,FT,JT,F,J,I}(
        f, function_type, Jf, jacobian_type, range_dimension
    )
end

@doc raw"""
    get_cost(M::AbstractManifold, vgf::VectorGradientFunction, p, i, range=nothing)

Evaluate the ``i``th component or components of the [`VectorGradientFunction`](@ref)
cost at `p`. The `range` can be used to speficy a potential range.
This is currently ignored, but kept for consistency with the gradient acces functions.

Since `i` is assumed to be a linear index, you can provide

* a single integer
* a `UnitRange` to specify a range to be returned like `1:3`
* a `BitVector` specifying a selection
* a `AbstractVector{<:Integer}` to specify indices
* `:` to return the vector of all gradients

"""
get_cost(M::AbstractManifold, vgf::VectorGradientFunction, p, i, range=nothing)
function get_cost(
    M::AbstractManifold,
    vgf::VectorGradientFunction{E,<:FunctionVectorialType},
    p,
    i,
    range=nothing,
) where {E}
    c = vgf.costs!!(M, p)
    if isa(c, Number)
        return c
    else
        return c[i]
    end
end
function get_cost(
    M::AbstractManifold,
    vgf::VectorGradientFunction{E,<:ComponentVectorialType},
    p,
    i::Integer,
    range=nothing,
) where {E}
    return vgf.costs!![i](M, p)
end
function get_cost(
    M::AbstractManifold,
    vgf::VectorGradientFunction{E,<:ComponentVectorialType},
    p,
    i,
    range=nothing,
) where {E}
    return [f(M, p) for f in vgf.costs!![i]]
end

@doc raw"""
    get_cost_function(vgf::VectorGradientFunction, recursive=false)

return the internally stored cost function.
"""
function get_cost_function(vgf::VectorGradientFunction, recursive=false)
    return vgf.costs!!
end

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
get_gradient(M::AbstractManifold, vgf::VectorGradientFunction, p, i, range=nothing)

_vgf_index_to_length(b::BitVector, n) = sum(b)
_vgf_index_to_length(::Colon, n) = n
_vgf_index_to_length(i::AbstractArray{<:Integer}) = length(i)
_vgf_index_to_length(r::UnitRange{<:Integer}) = lengh(r)

# Generic case, we allocate (a) a single tangent vector
function get_gradient(
    M::AbstractManifold,
    vgf::VectorGradientFunction,
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
    vgf::VectorGradientFunction,
    p,
    i, # as long as the length can be found it should work, see _vgf_index_to_length
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
)
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    P = fill(p, pM)
    X = zero_vector(pM, P)
    return get_gradient!(M, X, vgf, p, i, range)
end
# (c) Special cases where we can skip allocations and/or the power manifold
function get_gradient(
    M::AbstractManifold,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:ComponentVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=nothing,
) where {FT}
    return vgf.jacobian!![i](M, p)
end
function get_gradient(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:ComponentVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=nothing,
) where {FT}
    X = zero_vector(M, p)
    return vgf.jacobian!![i](M, X, p)
end

#
#
# Part I: Allocation
# I (a) Internally a Jacobian
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:CoefficientVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
    pM = PowerManifold(M, range, vgf.range_dimension...)
    JF = vgf.jacobian!!(M, p)
    get_vector!(M, X[pM, i], p, JF[i, :]) #convert rows to gradients
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:CoefficientVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    JF = vgf.jacobian!!(M, p)[i, :] #yields a n x d matrix where n is the number of indices
    for j in 1:n
        get_vector!(M, X[pM, j], p, JF[j, :])
    end
    return X
end
# I (b) a vector of functions
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:ComponentVectorialType},
    p,
    i::Integer,
    ::Union{AbstractPowerRepresentation,Nothing}=nothing,
) where {FT}
    return copyto!(M, X, p, vgf.jacobian!![i](M, p))
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:ComponentVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    for (j, f) in zip(i, vgf.jacobian!![i])
        copyto!(M, X[pM, j], p, f(M, p))
    end
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:ComponentVectorialType},
    p,
    i::Colon,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    for (j, f) in enumerate(vgf.jacobian!!)
        copyto!(M, X[pM, j], p, f(M, p))
    end
    return X
end
# I (c) A single gradient function
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:FunctionVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    mP = PowerManifold(M, range, n)
    copyto!(mP, X, vgf.jacobian!!(M, p)[mP, i])
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:AllocatingEvaluation,FT,<:FunctionVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
    mP = PowerManifold(M, range, vgf.range_dimension)
    copyto!(mP, X, vgf.jacobian!!(M, p)[mP, i])
    return X
end
#
#
# Part II: In-place evaluations
# II (a) Jacobian
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:CoefficientVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
    # We could also allocate a nxd matrix, but this might be nicer type wise
    pM = PowerManifold(M, range, vgf.range_dimension...)
    JF = reshape(
        get_coefficients(pM, fill(p, pM), X, vgf.jacobian_type.basis),
        power_dimensions(pM)...,
        :,
    )
    vgf.jacobian!!(M, JF, p)
    get_vector!(M, X, p, JF[i..., :])
    return X
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:CoefficientVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
    # We could also allocate a nxd matrix, but this might be nicer type wise
    pM = PowerManifold(M, range, vgf.range_dimension...)
    JF = reshape(
        get_coefficients(pM, fill(p, pM), X, vgf.jacobian_type.basis),
        power_dimensions(pM)...,
        :,
    )
    vgf.jacobian!!(M, JF, p)
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    JFi = vgf.jacobian!!(M, p)[i, :] #yields a n x d matrix where n is the number of indices
    for j in 1:n
        get_vector!(M, X[pM, j], p, JFi[j, :])
    end
    return X
end
#II (b) a vector of functions
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:ComponentVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=nothing,
) where {FT}
    return vgf.jacobian!![i](M, X, p)
end
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:ComponentVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    for (j, f) in zip(i, vgf.jacobian!![i])
        f(M, X[pM, j], p)
    end
    return X
end
# II(c) a sungle function
function get_gradient!(
    M::AbstractManifold,
    X,
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:FunctionVectorialType},
    p,
    i::Integer,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
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
    vgf::VectorGradientFunction{<:InplaceEvaluation,FT,<:FunctionVectorialType},
    p,
    i,
    range::Union{AbstractPowerRepresentation,Nothing}=NestedPowerRepresentation(),
) where {FT}
    #Singel access for function is a bit expensive
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM_out = PowerManifold(M, range, n)
    pM_temp = PowerManifold(M, range, vgf.range_dimension)
    P = fill(p, pM_temp)
    x = zero_vector(pM_temp, P)
    vgf.jacobian!!(M, x, p)
    # Luckily all documented access functions work directly on x[pM_temp,...]
    copyto!(pM_out, X, P[pM_temp, i], x[pM_temp, i])
    return X
end

@doc raw"""
    length(vgf::VectorGradientFunction)

Return the length of the vector the function ``f: \mathcal M → ℝ^n`` maps into,
that is the number `n`.
"""
length(vgf::VectorGradientFunction) = vgf.range_dimension
#
#
# TODO: Jacobian Evaluations
# TODO: Maybe generate Jacobian from Gradients?

#
#
# Maybe something for a bit later when we know that the approach above works just fine.
function get_jacobian end

@doc raw"""
    get_jacobian(M::AbstractManifold, vgf::VectorGradientFunction, p, B)

Evaluate and return the Jacobian with respect to a basis of the tangent space.
"""
get_jacobian(M::AbstractManifold, vgf::VectorGradientFunction, p, B)

#TODO Check all cases and decompose vectors in basis B to get Jacobian
# For jacobian representation check to maybe do a change of basis if the indernal basis does not agree with B? Or error?

function get_jacobian_function(vgf::VectorGradientFunction, recursive=false)
    return vgf.jacobian!!
end
