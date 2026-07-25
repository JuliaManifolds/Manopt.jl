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
function status_summary(vgf::VectorGradientFunction; context::Symbol = :default)
    _is_inline(context) && (return "A vectorial function including gradients of length $(length(vgf)) represented as $(vgf.cost_type) and gradients as $(vgf.jacobian_type)")
    return """
    A function defined on a manifold that maps into a vector space including gradients of the component functions.

    ## Components
    * cost:                   $(_MANOPT_INDENT)$(vgf.value!!)$(_MANOPT_INDENT)(as $(vgf.cost_type)),
    * gradient(s) or Jacobian:$(_MANOPT_INDENT)$(vgf.jacobian!!)$(_MANOPT_INDENT)(as $(vgf.jacobian_type))
    * dimension:              $(_MANOPT_INDENT)$(length(vgf))"""
end
function show(io::IO, vgf::VectorGradientFunction{E}) where {E}
    print(io, "VectorGradientFunction("); print(io, vgf.value!!); print(io, ", ")
    print(io, vgf.jacobian!!); print(io, ", "); print(io, vgf.range_dimension)
    print(io, "; "); print(io, _to_kw(E))
    print(io, ", function_type = "); print(io, vgf.cost_type); print(io, ", jacobian_type = ")
    return print(io, vgf.jacobian_type)
end



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

#
#
# Part I: allocation
# I (a) a vector of functions
function get_hessian!(
        M::AbstractManifold, Y, vhf::VectorHessianFunction{<:AllocatingEvaluation, FT, JT, <:ComponentVectorialType},
        p, X, i::Integer, ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT, JT}
    return copyto!(M, Y, p, vhf.hessians!![i](M, p, X))
end
function get_hessian!(
        M::AbstractManifold, Y, vhf::VectorHessianFunction{<:AllocatingEvaluation, FT, JT, <:ComponentVectorialType},
        p, X, i,
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
        M::AbstractManifold, Y, vhf::VectorHessianFunction{<:AllocatingEvaluation, FT, JT, <:ComponentVectorialType},
        p, X, i::Colon,
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
        M::AbstractManifold, Y, vhf::VectorHessianFunction{<:AllocatingEvaluation, FT, JT, <:FunctionVectorialType},
        p, X, i, range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
    ) where {FT, JT}
    n = _vgf_index_to_length(i, vhf.range_dimension)
    mP = PowerManifold(M, range, n)
    copyto!(mP, Y, vhf.hessians!!(M, p, X)[mP, i])
    return Y
end
function get_hessian!(
        M::AbstractManifold, Y, vhf::VectorHessianFunction{<:AllocatingEvaluation, FT, JT, <:FunctionVectorialType},
        p, X, i::Integer, range::Union{AbstractPowerRepresentation, Nothing} = get_range(vhf.hessian_type),
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
        M::AbstractManifold, Y, vhf::VectorHessianFunction{<:InplaceEvaluation, FT, JT, <:ComponentVectorialType},
        p, X, i::Integer, ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT, JT}
    return vhf.hessians!![i](M, Y, p, X)
end
function get_hessian!(
        M::AbstractManifold, Y, vhf::VectorHessianFunction{<:InplaceEvaluation, FT, JT, <:ComponentVectorialType},
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
# II(b) a single function
function get_hessian!(
        M::AbstractManifold, Y, vhf::VectorHessianFunction{<:InplaceEvaluation, FT, JT, <:FunctionVectorialType},
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
        M::AbstractManifold, Y, vhf::VectorHessianFunction{<:InplaceEvaluation, FT, JT, <:FunctionVectorialType},
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
get_jacobian(::AbstractManifold, ::AbstractVectorGradientFunction, p; kwargs...)
function get_jacobian! end
@doc "$(_doc_get_jacobian_matrix_vgf)"
get_jacobian!(M::AbstractManifold, JF, vgf::AbstractVectorGradientFunction, p)

function get_jacobian(
        M::AbstractManifold, vgf::VGF, p; basis::AbstractBasis = get_basis(vgf.jacobian_type), kwargs...
    ) where {FT, VGF <: AbstractFirstOrderVectorFunction{<:AbstractEvaluationType, FT, <:AbstractVectorialType}}
    JF = allocate_jacobian(M, vgf, basis; T = number_eltype(p))
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
        get_coordinates!(M, view(JF, i, :), p, vgf.jacobian!![i](M, p), basis)
    end
    return JF
end
# (c) Jacobian matrix function in a basis
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p; basis::AbstractBasis = get_basis(vgf.jacobian_type), range = nothing,
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:CoefficientVectorialType}}
    JF .= vgf.jacobian!!(M, p)
    _change_basis!(M, JF, p, vgf.jacobian_type.basis, basis)
    return JF
end
# (d) Jacobian as a differential -> build matrix column by column by passing the basis vectors in
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p; basis::AbstractBasis = get_basis(vgf.jacobian_type), range = nothing,
    ) where {FT, VGF <: VectorDifferentialFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType}}
    V = get_vectors(M, p, get_basis(M, p, basis))
    for i in 1:length(V)
        c = @view JF[:, i]
        c .= vgf.jacobian!!(M, p, V[i])
    end
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
        get_coordinates!(M, view(JF, i, :), p, gradients[mP, i], basis)
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
        get_coordinates!(M, view(JF, i, :), p, X, basis)
    end
    return JF
end
# (c) a matrix
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p;
        basis::AbstractBasis = get_basis(vgf.jacobian_type), range = nothing, X = nothing, Y_cache = nothing,
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType},
    }
    vgf.jacobian!!(M, JF, p)
    _change_basis!(M, JF, p, vgf.jacobian_type.basis, basis)
    return JF
end
# (d) Jacobian as a differential -> build matrix column by column by passing the basis vectors in
function get_jacobian!(
        M::AbstractManifold, JF, vgf::VGF, p; basis::AbstractBasis = get_basis(vgf.jacobian_type), range = nothing,
    ) where {FT, VGF <: VectorDifferentialFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType}}
    V = get_vectors(M, p, get_basis(M, p, basis))
    for i in 1:length(V)
        vgf.jacobian!!(M, view(JF, :, i), p, V[i])
    end
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
Then it simplifies to a matrix multiplication.

This can be computed in-place of `a`.
"""

@doc "$(_doc_get_jacobian_function_vector)"
get_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, X; kwargs...)

@doc "$(_doc_get_jacobian_function_vector)"
get_jacobian!(M::AbstractManifold, a, vgf::AbstractVectorGradientFunction, p, X; kwargs...)

# For the allocating one, we just need to allocate a
function get_jacobian(
        M::AbstractManifold, vgf::VGF, p, X; kwargs...
    ) where {FT, VGF <: AbstractFirstOrderVectorFunction{<:AbstractEvaluationType, FT, <:AbstractVectorialType}}
    n = vgf.range_dimension
    a = zeros(number_eltype(X), n)
    return get_jacobian!(M, a, vgf, p, X; kwargs...)
end
# (a) a single function, allocating
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, X; Y_cache = nothing, c_cache = nothing
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType}}
    n = vgf.range_dimension
    mP = PowerManifold(M, get_range(vgf.jacobian_type), n)
    gradients = vgf.jacobian!!(M, p)
    for i in 1:n
        a[i] = inner(M, p, gradients[mP, i], X)
    end
    return a
end
# (b) vector of gradient functions
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, X; Y_cache = nothing, c_cache = nothing,
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType}}
    n = vgf.range_dimension
    for i in 1:n
        a[i] = inner(M, p, vgf.jacobian!![i](M, p), X)
    end
    return a
end
# (c) Jacobian function – easy: Decompose X and call the other, for both
function get_jacobian!(
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:CoefficientVectorialType}, p, X;
        range = nothing, Y_cache = nothing, c_cache = allocate_result(M, get_coordinates, p, X, vgf.jacobian_type.basis)
    ) where {FT}
    B = vgf.jacobian_type.basis
    get_coordinates!(M, c_cache, p, X, B)
    return get_jacobian!(M, a, vgf, p, c_cache, B)
end
# (d) Jacobian differential – easiest: just call it
function get_jacobian!(
        M::AbstractManifold, a, vgf::VectorDifferentialFunction{<:AbstractEvaluationType, FT, <:FunctionVectorialType}, p, X;
        range = nothing, Y_cache = nothing, c_cache = nothing
    ) where {FT}
    a .= vgf.jacobian!!(M, p, X)
    return a
end
# II (a) Inplace single function – skip for now since allocation not so easy? we would need a power version of the point p
function get_jacobian!(
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType}, p, X;
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
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType}, p, X;
        Y_cache = zero_vector(M, p), c_cache = nothing,
    ) where {FT}
    n = vgf.range_dimension
    for i in 1:n
        vgf.jacobian!![i](M, Y_cache, p)
        a[i] = inner(M, p, Y_cache, X)
    end
    return a
end
# (c) Jacobian function – for now not provided
# (d) Jacobian differential – easiest: just call it
function get_jacobian!(
        M::AbstractManifold, a, vgf::VectorDifferentialFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType}, p, X;
        range = nothing, Y_cache = nothing, c_cache = allocate_result(M, get_coordinates, p, X, get_basis(vgf.jacobian_type))
    ) where {FT}
    return vgf.jacobian!!(M, a, p, X)
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

# Part I: allocating/inplace vgf – allocating/inplace work the same here jacobian (a) single gradient function
function get_jacobian(
        M::AbstractManifold, vgf::VGF, p, c, B::AbstractBasis; X = zero_vector(M, p), kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:FunctionVectorialType}}
    n = vgf.range_dimension
    a = zeros(number_eltype(X), n)
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
    a = zeros(number_eltype(X), n)
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
    a = zeros(number_eltype(isnothing(X) ? c : X), n)
    return get_jacobian!(M, a, vgf, p, c, B; X = X, kwargs...)
end
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, c, B::AbstractBasis; kwargs...
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:CoefficientVectorialType}}
    c2 = change_basis(M, p, c, B, vgf.jacobian_type.basis) # only allocates if basis changed
    a .= vgf.jacobian!!(M, p) * c
    return a
end
# Part I: allocating VGF, (d) Jacobian differential function
function get_jacobian(
        M::AbstractManifold, vgf::VGF, p, c, B::AbstractBasis; X = nothing, kwargs...
    ) where {FT, VGF <: VectorDifferentialFunction{<:AbstractEvaluationType, FT, <:FunctionVectorialType}}
    n = vgf.range_dimension
    a = zeros(number_eltype(isnothing(X) ? c : X), n)
    return get_jacobian!(M, a, vgf, p, c, B; X = X, kwargs...)
end
function get_jacobian!(
        M::AbstractManifold, a, vgf::VGF, p, c, B::AbstractBasis; kwargs...
    ) where {FT, VGF <: VectorDifferentialFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType}}
    a .= vgf.jacobian!!(M, p, get_vector(M, p, c, B))
    return a
end
# Part II: mutating vgf
# (c) Jacobian function
function get_jacobian!(
        M::AbstractManifold, a, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType}, p, c, B::AbstractBasis;
        X = nothing, Y_cache = nothing,
    ) where {FT}
    JF = allocate_jacobian(M, vgf; T = eltype(c))
    vgf.jacobian!!(M, JF, p)
    a .= JF * c
    return a
end
# (d) Jacobian differential
function get_jacobian!(
        M::AbstractManifold, a, vgf::VectorDifferentialFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType}, p, c, B::AbstractBasis;
        X = nothing, Y_cache = nothing,
    ) where {FT}
    return vgf.jacobian!!(M, a, p, get_vector(M, p, c, B))
end

function get_jacobian_basis(vgf::AbstractVectorGradientFunction)
    return _get_jacobian_basis(vgf.jacobian_type)
end
_get_jacobian_basis(jt::AbstractVectorialType) = DefaultOrthonormalBasis()
_get_jacobian_basis(jt::CoefficientVectorialType) = jt.basis

function add_vector!(M::AbstractManifold, X, p, c, basis::AbstractBasis)
    Y = get_vector(M, p, c, basis)
    X .+= Y
    return X
end
function add_vector!(M::ProductManifold, X, p, c, basis::AbstractBasis)
    dims = map(manifold_dimension, M.manifolds)
    @assert length(c) == sum(dims)
    dim_ranges = ManifoldsBase._get_dim_ranges(dims)
    tc = map(dr -> (@inbounds view(c, dr)), dim_ranges)
    ts = ManifoldsBase.ziptuples(
        M.manifolds,
        submanifold_components(M, X),
        submanifold_components(M, p),
        tc,
    )
    map(ts) do t
        return add_vector!(t..., basis)
    end
    return X
end
function add_coordinates!(M::AbstractManifold, c, p, X, basis::AbstractBasis)
    cX = get_coordinates(M, p, X, basis)
    c .+= cX
    return c
end

#
#
# --- Adjoint Jacobian function in terms of gradients as a 1-1 tensor (basis free)

_doc_get_adjoint_jacobian_vector = """
    get_adjoint_jacobian(M::AbstractManifold, vgf::AbstractVectorGradientFunction, p, a; kwargs...)
    get_adjoint_jacobian!(M::AbstractManifold, X, vgf::AbstractVectorGradientFunction, p, a; kwargs...)

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

This can be computed in-place of `X`.
To directly add a Jacobian to `X` see [`add_adjoint_jacobian!`](@ref)

!!! note
    For the case of a matrix representation, i.e. the function signature
    `get_jacobian!(M, JF, vgf, p)` the resulting matrix can just be transposed to obtain the adjoint,
    if you used an [`DefaultOrthonormalBasis`](@extref `ManifoldsBase.DefaultOrthonormalBasis`).
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

_doc_add_adjoint_jacobian_function_vector = """
    add_adjoint_jacobian!(M::AbstractManifold, X, vgf::AbstractVectorGradientFunction, p, a; kwargs...)

Compute the adjoint Jacobian ``J_F^*(p)[a]`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^n``
and add it to `X`.
For more details see [`get_adjoint_jacobian`](@ref).
"""

@doc "$(_doc_add_adjoint_jacobian_function_vector)"
add_adjoint_jacobian!(M::AbstractManifold, X, vgf::AbstractVectorGradientFunction, p, a; kwargs...)

# Part I: allocating vgf (a) single gradient function
function add_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::VGF, p, a::AbstractVector; Y_cache = nothing,
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType},
    }
    n = vgf.range_dimension
    mP = PowerManifold(M, get_range(vgf.jacobian_type), n)
    gradients = vgf.jacobian!!(M, p)
    for i in 1:n
        X .+= a[i] * gradients[mP, i]
    end
    return X
end
# (b) vector of gradient functions
function add_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::VGF, p, a::AbstractVector; Y_cache = nothing,
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType},
    }
    n = vgf.range_dimension
    for i in 1:n
        X .+= a[i] * vgf.jacobian!![i](M, p)
    end
    return X
end
# (c) Jacobian function
function add_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:CoefficientVectorialType}, p, a;
        basis::B = default_basis(M, typeof(p)), Y_cache = nothing,
    ) where {FT, B <: AbstractBasis}
    n = vgf.range_dimension
    JF = vgf.jacobian!!(M, p)
    c = adjoint(JF) * a
    add_vector!(M, X, p, c, basis)
    return X
end
# (d) Jacobian function
function add_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::VectorDifferentialFunction{<:AllocatingEvaluation, FT, JT, <:FunctionVectorialType}, p, a;
        Y_cache = nothing,
    ) where {FT, JT}
    X .+= vgf.adjoint_jacobian!!(M, p, a)
    return X
end
# Part II: mutating vgf (a) single gradient function
function add_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType}, p, a::AbstractVector;
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
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType}, p, a::AbstractVector;
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
        M::AbstractManifold, X, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType}, p, a::AbstractVector;
        Y_cache = nothing
    ) where {FT}
    J = allocate_jacobian(M, vgf; T = eltype(a))
    add_vector!(M, X, p, adjoint(vgf.jacobian!!(M, J, p)) * a, vgf.jacobian_type.basis)
    return X
end
# (d) Jacobian function
function add_adjoint_jacobian!(
        M::AbstractManifold, X, vgf::VectorDifferentialFunction{<:InplaceEvaluation, FT, JT, <:FunctionVectorialType}, p, a::AbstractVector;
        Y_cache = zero_vector(M, p)
    ) where {FT, JT}
    zero_vector!(M, Y_cache, p)
    vgf.adjoint_jacobian!!(M, Y_cache, p, a)
    X .+= Y_cache
    return X
end
#
#
# --- Adjoint Jacobian function in terms of gradients as a matrix-vector product in a basis

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
    return add_adjoint_jacobian!(M, c, vgf, p, a, B, kwargs...)
end

@doc "$(_doc_get_adjoint_jacobian_function_coeff)"
function get_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::AbstractFirstOrderVectorFunction, p, a::AbstractVector, B::AbstractBasis; kwargs...
    )
    fill!(c, 0)
    return add_adjoint_jacobian!(M, c, vgf, p, a, kwargs...)
end

_doc_add_adjoint_jacobian_function_coeff = """
    add_adjoint_jacobian!(M::AbstractManifold, c, vgf::AbstractVectorGradientFunction, p, a, B::AbstractBasis; kwargs...)

Compute the adjoint Jacobian ``J_F^*(p)[a]`` of a vectorial function ``F: $(_math(:Manifold)) → ℝ^n`` as a matrix in a tangent space.
For more details see [`get_adjoint_jacobian`](@ref).
"""

@doc "$(_doc_add_adjoint_jacobian_function_coeff)"
add_adjoint_jacobian!(M::AbstractManifold, c, vgf::AbstractVectorGradientFunction, p, a::AbstractVector, B::AbstractBasis; kwargs...)

# Part I: allocating vgf (a) single gradient function
function add_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::VGF, p, a::AbstractVector, B::AbstractBasis; X = zero_vector(M, p), Y_cache = nothing,
    ) where {
        FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:FunctionVectorialType},
    }
    # easiest: call the one for the vector and decompose into c
    add_adjoint_jacobian!(M, X, vgf, p, a)
    return add_coordinates!(M, c, p, X, B)
end
# (b) vector of gradient functions - same as above
function add_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::VGF, p, a::AbstractVector, B::AbstractBasis; X = zero_vector(M, p), Y_cache = nothing,
    ) where {FT, VGF <: AbstractVectorGradientFunction{<:AbstractEvaluationType, FT, <:ComponentVectorialType}}
    add_adjoint_jacobian!(M, X, vgf, p, a)
    add_coordinates!(M, c, p, X, B)
    return c
end
# I allocating, (c) Jacobian function
function add_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:CoefficientVectorialType}, p, a::AbstractVector, B::AbstractBasis; X = nothing, Y_cache = nothing,
    ) where {FT}
    JF = vgf.jacobian!!(M, p)
    cX = adjoint(JF) * a
    if vgf.jacobian_type.basis === B
        c .+= cX
    else
        c .+= change_basis(M, p, cX, vgf.jacobian_type.basis, B)
    end
    return c
end
# mutating, (c) Jacobian function
function add_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType}, p, a::AbstractVector, B::AbstractBasis; X = nothing, Y_cache = nothing
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
# allocating (d) Jacobian function
function add_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::VectorDifferentialFunction{<:AllocatingEvaluation, FT, JT, <:FunctionVectorialType}, p, a::AbstractVector, B::AbstractBasis; X = nothing, Y_cache = nothing
    ) where {FT, JT}
    add_coordinates!(M, c, p, vgf.adjoint_jacobian!!(M, p, a), B)
    return c
end
# (d) Jacobian function
function add_adjoint_jacobian!(
        M::AbstractManifold, c, vgf::VectorDifferentialFunction{<:InplaceEvaluation, FT, JT, <:FunctionVectorialType}, p, a::AbstractVector, B::AbstractBasis; X = nothing, Y_cache = zero_vector(M, p)
    ) where {FT, JT}
    vgf.adjoint_jacobian!!(M, Y_cache, p, a)
    add_coordinates!(M, c, p, Y_cache, B)
    return c
end
#
#
# ---- Gradient
@doc """
    get_gradient(M::AbstractManifold, vgf::VectorGradientFunction, p, i)
    get_gradient(M::AbstractManifold, vgf::VectorGradientFunction, p, i, range)
    get_gradient!(M::AbstractManifold, X, vgf::VectorGradientFunction, p, i)
    get_gradient!(M::AbstractManifold, X, vgf::VectorGradientFunction, p, i, range)

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
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType},
        p, i::Integer, ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT <: AbstractVectorialType}
    return vgf.jacobian!![i](M, p)
end
function get_gradient(
        M::AbstractManifold,
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType},
        p, i::Integer, ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT <: AbstractVectorialType}
    X = zero_vector(M, p)
    return vgf.jacobian!![i](M, X, p)
end
# (d) diff and adjoint diff
function get_gradient(
        M::AbstractManifold,
        vgf::VectorDifferentialFunction{<:AllocatingEvaluation, FT, JT, <:FunctionVectorialType},
        p, i::Integer, ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT <: AbstractVectorialType, JT <: AbstractVectorialType}
    n = vgf.range_dimension
    ei = zeros(n); ei[i] = 1
    return vgf.adjoint_jacobian!!(M, p, ei)
end
function get_gradient(
        M::AbstractManifold,
        vgf::VectorDifferentialFunction{<:InplaceEvaluation, FT, JT, <:FunctionVectorialType},
        p, i::Integer,
        ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT <: AbstractVectorialType, JT <: AbstractVectorialType}
    n = vgf.range_dimension
    ei = zeros(n); ei[i] = 1
    X = zero_vector(M, p)
    return vgf.adjoint_jacobian!!(M, X, p, ei)
end
#
#
# Part I: allocation
# I (a) Internally a Jacobian
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:CoefficientVectorialType},
        p, i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    JF = vgf.jacobian!!(M, p)
    get_vector!(M, X, p, view(JF, i, :), vgf.jacobian_type.basis) #convert rows to gradients
    return X
end
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:CoefficientVectorialType},
        p, i = :, range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    pM = PowerManifold(M, range, n)
    rep_size = representation_size(M)
    JF = vgf.jacobian!!(M, p) # yields a full Jacobian
    for (j, k) in zip(_to_iterable_indices([JF[:, 1]...], i), 1:n)
        get_vector!(M, _write(pM, rep_size, X, (k,)), p, view(JF, j, :), vgf.jacobian_type.basis)
    end
    return X
end
# Part I(b) a vector of functions
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType},
        p, i::Integer,
        (::Union{AbstractPowerRepresentation, Nothing}) = nothing,
    ) where {FT <: AbstractVectorialType}
    return copyto!(M, X, p, vgf.jacobian!![i](M, p))
end
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType},
        p, i, range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
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
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:ComponentVectorialType},
        p, i::Colon,
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
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType},
        p, i, range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    n = _vgf_index_to_length(i, vgf.range_dimension)
    mP = PowerManifold(M, range, n)
    copyto!(mP, X, vgf.jacobian!!(M, p)[mP, i])
    return X
end
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{<:AllocatingEvaluation, FT, <:FunctionVectorialType},
        p, i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    mP = PowerManifold(M, range, vgf.range_dimension)
    copyto!(M, X, p, vgf.jacobian!!(M, p)[mP, i])
    return X
end
# Part I(d) adjoint differentials
function get_gradient!(
        M::AbstractManifold, X,
        vgf::VectorDifferentialFunction{<:AllocatingEvaluation, FT, JT, <:FunctionVectorialType},
        p, i::Integer,
        ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT <: AbstractVectorialType, JT <: AbstractVectorialType}
    n = vgf.range_dimension
    ei = zeros(n); ei[i] = 1
    return copyto!(M, X, p, vgf.adjoint_jacobian!!(M, p, ei))
end
#
#
# Part II: in-place evaluations
# (a) Jacobian
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType},
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
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:CoefficientVectorialType},
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
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType},
        p, i::Integer,
        range::Union{AbstractPowerRepresentation, Nothing} = get_range(vgf.jacobian_type),
    ) where {FT <: AbstractVectorialType}
    return vgf.jacobian!![i](M, X, p)
end
function get_gradient!(
        M::AbstractManifold, X,
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:ComponentVectorialType},
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
        vgf::AbstractVectorGradientFunction{<:InplaceEvaluation, FT, <:FunctionVectorialType},
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
# II(d) adjoint
function get_gradient!(
        M::AbstractManifold,
        X,
        vgf::VectorDifferentialFunction{<:InplaceEvaluation, FT, JT, <:FunctionVectorialType},
        p, i::Integer,
        ::Union{AbstractPowerRepresentation, Nothing} = nothing,
    ) where {FT <: AbstractVectorialType, JT <: AbstractVectorialType}
    n = vgf.range_dimension
    ei = zeros(n); ei[i] = 1
    return vgf.adjoint_jacobian!!(M, X, p, ei)
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
    return isa(c, Number) ? c : c[i]
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
    return i === Colon() ? [f(M, p) for f in vgf.value!!] : [f(M, p) for f in vgf.value!![i]]
end
function get_value(
        M::AbstractManifold, vgf::AbstractVectorFunction{E, <:FunctionVectorialType},
        p, i = :; value_cache = zeros(vgf.range_dimension),
    ) where {E <: InplaceEvaluation}
    vgf.value!!(M, value_cache, p)
    return value_cache[i]
end

function get_value!(
        M::AbstractManifold, V, vgf::AbstractVectorFunction{AllocatingEvaluation, <:FunctionVectorialType},
        p, i = :,
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
    if i === Colon()
        for i in eachindex(vgf.value!!, V)
            V[i] = vgf.value!![i](M, p)
        end
    else
        V .= i isa Number ? [vgf.value!![i](M, p)] : [f(M, p) for f in vgf.value!![i]]
    end

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
