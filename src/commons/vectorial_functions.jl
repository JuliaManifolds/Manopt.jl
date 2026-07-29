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
    get_value_function(vgf::VectorGradientFunction, recursive=false)

return the internally stored function computing [`get_value`](@ref).
"""
function get_value_function(vgf::VectorGradientFunction, recursive = false)
    return vgf.value!!
end

function status_summary(vgf::VectorGradientFunction; context::Symbol = :default)
    _is_inline(context) && (return "A vectorial function including gradients of length $(length(vgf)) represented as $(vgf.cost_type) and gradients as $(vgf.jacobian_type)")
    return """
    A function defined on a manifold that maps into a vector space including gradients of the component functions.

    ## Components
    * cost:                   $(_MANOPT_INDENT)$(vgf.value!!)$(_MANOPT_INDENT)(as $(vgf.cost_type)),
    * gradient(s) or Jacobian:$(_MANOPT_INDENT)$(vgf.jacobian!!)$(_MANOPT_INDENT)(as $(vgf.jacobian_type))
    * dimension:              $(_MANOPT_INDENT)$(length(vgf))"""
end
function show(io::IO, vgf::VectorGradientFunction)
    print(io, "VectorGradientFunction("); print(io, vgf.value!!); print(io, ", ")
    print(io, vgf.jacobian!!); print(io, ", "); print(io, vgf.range_dimension)
    print(io, "; ")
    print(io, ", function_type = "); print(io, vgf.cost_type); print(io, ", jacobian_type = ")
    return print(io, vgf.jacobian_type)
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
# As an operator
function add_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::VectorDifferentialFunction{FT, JT, <:FunctionVectorialType}, p, a::AbstractVector;
        Y_cache = zero_vector(M, p)
    ) where {FT, JT}
    zero_vector!(M, Y_cache, p)
    vgf.adjoint_jacobian!!(M, Y_cache, p, a)
    X .+= Y_cache
    return X
end
# in coordinates
function add_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::VectorDifferentialFunction{FT, JT, <:FunctionVectorialType}, p, a::AbstractVector, B::AbstractBasis; X = nothing, Y_cache = zero_vector(M, p)
    ) where {FT, JT}
    vgf.adjoint_jacobian!!(M, Y_cache, p, a)
    add_coordinates!(M, c, p, Y_cache, B)
    return c
end

function get_gradient(
        M::AbstractManifold, vgf::VectorDifferentialFunction{FT, JT, <:FunctionVectorialType},
        p, i::Integer, ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT <: AbstractVectorialType, JT <: AbstractVectorialType}
    n = vgf.range_dimension
    ei = zeros(n); ei[i] = 1
    X = zero_vector(M, p)
    return vgf.adjoint_jacobian!!(M, X, p, ei)
end
function get_gradient!(
        M::AbstractManifold, X, vgf::VectorDifferentialFunction{FT, JT, <:FunctionVectorialType},
        p, i::Integer, ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT <: AbstractVectorialType, JT <: AbstractVectorialType}
    n = vgf.range_dimension
    ei = zeros(n); ei[i] = 1
    return vgf.adjoint_jacobian!!(M, X, p, ei)
end

# Jacobian in matrix form JF
# Jacobian as a differential -> build matrix column by column by passing the basis vectors in
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p; basis::AbstractBasis = get_basis(vgf.jacobian_type), range = nothing,
    ) where {FT, VGF <: VectorDifferentialFunction{FT, <:FunctionVectorialType}}
    V = get_vectors(M, p, get_basis(M, p, basis))
    for i in 1:length(V)
        vgf.jacobian!!(M, view(JF, :, i), p, V[i])
    end
    return JF
end
# Jacobian in linear operator form
function get_jacobian!(
        M::AbstractManifold, a, vgf::VectorDifferentialFunction{FT, <:FunctionVectorialType}, p, X;
        range = nothing, Y_cache = nothing, c_cache = allocate_result(M, get_coordinates, p, X, get_basis(vgf.jacobian_type))
    ) where {FT}
    return vgf.jacobian!!(M, a, p, X)
end
# linear operator in coordinates
function get_jacobian!(
        M::AbstractManifold, a, vgf::VectorDifferentialFunction{FT, <:FunctionVectorialType}, p, c, B::AbstractBasis;
        X = nothing, Y_cache = nothing,
    ) where {FT}
    return vgf.jacobian!!(M, a, p, get_vector(M, p, c, B))
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
    print(io, "; ")
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
    # TODO: Readd evaluation keyword and wrap accordingly
    return VectorHessianFunction{FT, JT, HT, F, J, H, I}(
        f, function_type, Jf, jacobian_type, Hf, hessian_type, range_dimension
    )
end

_vgf_index_to_length(b::BitVector, n) = sum(b)
_vgf_index_to_length(::Colon, n) = n
_vgf_index_to_length(i::AbstractArray{<:Integer}, n) = length(i)
_vgf_index_to_length(r::UnitRange{<:Integer}, n) = length(r)

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
    M::AbstractManifold, vgf::VectorHessianFunction, p, X, i,
    range::Union{AbstractPowerRepresentation, Nothing} = nothing,
)

# Generic case, allocate (a) a single tangent vector
function get_hessian(
        M::AbstractManifold, vhf::VectorHessianFunction, p, X, i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    )
    Y = zero_vector(M, p)
    return get_hessian!(M, Y, vhf, p, X, i, range)
end
# (b) UnitRange and AbstractVector allow to use length for BitVector its sum
function get_hessian(
        M::AbstractManifold, vhf::VectorHessianFunction, p, X,
        i = :, # as long as the length can be found it should work, see _vgf_index_to_length
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    )
    n = _vgf_index_to_length(i, vhf.range_dimension)
    pM = PowerManifold(M, range, n)
    P = fill(p, pM)
    Y = zero_vector(pM, P)
    return get_hessian!(M, Y, vhf, p, X, i, range)
end
# Actual implementations
# (a) a vector of functions, single access i
function get_hessian!(
        M::AbstractManifold, Y, vhf::VectorHessianFunction{FT, JT, <:ComponentVectorialType},
        p, X, i::Integer, ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT, JT}
    return vhf.hessians!![i](M, Y, p, X)
end
# (a) arbitrary i
function get_hessian!(
        M::AbstractManifold, Y, vhf::VectorHessianFunction{FT, JT, <:ComponentVectorialType},
        p, X, i, range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
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
# (b) a single function
function get_hessian!(
        M::AbstractManifold, Y, vhf::VectorHessianFunction{FT, JT, <:FunctionVectorialType},
        p, X, i::Integer, range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    ) where {FT, JT}
    pM = PowerManifold(M, range, vhf.range_dimension...)
    P = fill(p, pM)
    y = zero_vector(pM, P)
    vhf.hessians!!(M, y, p, X)
    copyto!(M, Y, p, y[pM, i])
    return Y
end
function get_hessian!(
        M::AbstractManifold, Y, vhf::VectorHessianFunction{FT, JT, <:FunctionVectorialType},
        p, X, i, range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
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
    print(io, vhf.range_dimension); print(io, "; ")
    print(io, ", function_type = "); print(io, vhf.cost_type)
    print(io, ", jacobian_type = "); print(io, vhf.jacobian_type)
    print(io, ", hessian_type = ")
    return print(io, vhf.hessian_type)
end
