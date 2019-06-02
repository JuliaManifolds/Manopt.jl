#
#      Gr(k,V) - The space of all k-dimensional linear subspaces
#           of the n-dimensional vector space C^n
#  A point on this manifold is a vector subspace of dimension k embedded in C^n
#  represented by a n×k matrix with full rank.
#  A point in an tangential space is a matrix.
#
import LinearAlgebra: norm, nullspace, det, tr, qr, triu, rank, svd, Diagonal, diag
import Base: exp, log, show, cat, rand, Matrix, real, atan
export Grassmannian, GrPoint, GrTVector, getValue
export distance, dot, exp, log, manifoldDimension, norm, retraction, inverseRetraction
export parallelTransport, randomTVector, randomMPoint, validateMPoint, validateTVector
export projection, zeroTVector
#
# Type definitions
#

@doc doc"""
    Grassmannian <: Manifold

The manifold $\mathcal M = \mathrm{Gr}(k,n)$ of the set of k-dimensional
subspaces in $\mathbb{K}^{n}$ represented by $n\times k$
orthonormal matrices, that are either real- or complex-valued.

# Abbreviation
`Gr`

# Constructor
    Grassmannian(k, n[, d=Float64])

generates the manifold $\mathrm{Gr}(k,n)$ where the integer `n` is the
number of rows and `k` is the number of columns of the matrices and the optional
parameter `d` sets the `DataType` of the matrix entries.
"""
struct Grassmannian{T<:Union{U, Complex{U}} where U<:AbstractFloat} <: Manifold
  name::String
  dimensionsubspace::Int
  dimensionvecspace::Int
  abbreviation::String
  function Grassmannian{T}(dimensionsubspace::Int, dimensionvecspace::Int) where T<:Union{U, Complex{U}} where U<:AbstractFloat
    if dimensionsubspace > dimensionvecspace
      throw(ErrorException("dimensionsubspace can't be bigger than dimensionvecspace: $dimensionsubspace > $dimensionvecspace"))
    else
     new("of the set of $dimensionsubspace-dimensional subspaces in K$dimensionvecspace in $T", dimensionsubspace, dimensionvecspace,"Gr($dimensionsubspace,$dimensionvecspace)")
    end
  end
end
Grassmannian(dimensionsubspace::Int, dimensionvecspace::Int, D::DataType=Float64) = Grassmannian{D}(dimensionsubspace, dimensionvecspace)

@doc doc"""
    GrPoint <: MPoint

A point $x$ on the manifold $\mathcal M = \mathrm{Gr}(k,n)$ is an orthonormal
matrix of size $ n×k$. The matrix is a numerical representation of the vector
subspace its columns span.

# Constructor

    GrPoint(p)

where `p::Matrix` is an orthonormal matrix of dimension $n×k$.
"""
struct GrPoint{T<:Union{U, Complex{U}} where U<:AbstractFloat} <: MPoint
  value::Matrix{T}
  GrPoint{T}(value::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat = new(value)
end
GrPoint(value::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat = GrPoint{T}(value)
getValue(x::GrPoint) = x.value;

@doc doc"""
    GrTVector <: TVector

A tangent vector $\xi \in T_x\mathcal M$ on the manifold
$\mathcal M = \mathrm{Gr}(k,n)$. The tangent space is given by as

```math
T_x\mathrm{Gr}(k,n) = \bigl\{
\xi \in \mathbb{K}^{n\times k} \ \big|
\ {\bar ξ}^\mathrm{T}x = {\bar x}^\mathrm{T}ξ = 0_{k×k} \bigr\}.
```

# Constructor

    GrTVector(ξ)

where `ξ` is an $n\times k$ `Matrix` that satisfies the above.
"""
struct GrTVector{T<:Union{U, Complex{U}} where U<:AbstractFloat} <: TVector
  value::Matrix{T}
  GrTVector{T}(value::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat = new(value)
end
GrTVector(value::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat = GrTVector{T}(value)
getValue(ξ::GrTVector) = ξ.value;

@traitimpl IsMatrixM{Grassmannian}
@traitimpl IsMatrixP{GrPoint}
@traitimpl IsMatrixTV{GrTVector}

@doc doc"""
    dot(M,x,ξ,ν)

compute the Riemannian inner product for two [`GrTVector`](@ref)s `ξ` and `ν`
from $T_x\mathcal M$ of the [`Grassmannian`](@ref) manifold `M` given by

$\langle \xi, \nu \rangle_x = \operatorname{Re}(\operatorname{tr}(\xi^{\mathrm{T}} ν)),$
"""
function dot(M::Grassmannian{T}, x::GrPoint{T}, ξ::GrTVector{T}, ν::GrTVector{T})::Float64 where T<:Union{U, Complex{U}} where U<:AbstractFloat
  return real(tr(getValue(ξ)' * getValue(ν)))
end

@doc doc"""
    distance(M,x,y)

compute the Riemannian distance on [`Grassmannian`](@ref) manifold `M`$= \mathrm{Gr}(k,n)$ embedded in
$\mathbb R$. Let $USV = {\bar x}^\mathrm{T}y$ denote the SVD decomposition of
$x'y$. Then we compute

$d_{\mathrm{GR}(k,n)}(x,y) = \operatorname{norm}(\operatorname{Re}(b))$

where

$b_{i}=\begin{cases} 0 & \text{if} \; S_i≧1 \\ \operatorname{acos}(S_i) & \, \text{if} \; S_i<1 \end{cases}.$
"""
function distance(M::Grassmannian{T},x::GrPoint{T},y::GrPoint{T})::Float64 where T<:Union{U, Complex{U}} where U<:AbstractFloat
  if x==y
	return 0
  else
    z = getValue(x)'*getValue(y)
    a = svd(z).S
    b = zero(a)
    b[a.<1] = acos.(a[a.<1])
	return norm(real(b), 2)
  end
end

@doc doc"""
    exp(M,x,ξ,[t=1.0])

compute the exponential map on the [`Grassmannian`](@ref) manifold `M`$= \mathrm{Gr}(k,n)$ with
respect to the [`GrPoint`](@ref) `x` and the [`GrTVector`](@ref) `ξ`, which can
be shortened with `t` to `tξ`. Let $USV = t\xi$ denote the SVD decomposition of
$t\xi$. Then we compute

$A = x\cdot V\cdot \cos(S)\cdot {\bar V}^\mathrm{T} + U \cdot \sin(S) \cdot {\bar V}^\mathrm{T},$

where cosine and sine are applied element wise to the diagonal entries of $S$.
The resulting point $y$ of the exponential map is then the matrix $Q$ of the
QR decomposition $A=QR$ of $A$.
"""
function exp(M::Grassmannian{T},x::GrPoint{T},ξ::GrTVector{T},t::Float64=1.0) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  ξ = t*ξ
  d = svd(getValue(ξ))
  Y = getValue(x) * d.V * cos(Diagonal(d.S)) * (d.V)' + (d.U) * sin(Diagonal(d.S)) * (d.V)'
  GrPoint( Matrix( qr(Y).Q ) )
end

@doc doc"""
    inverseRetraction(M,x,y)

return a [`GrTVector`](@ref) `ξ` of the tagent space $T_x\mathrm{Gr}(k,n)$
with which the [`GrPoint`](@ref) `y` can be reached on the
[`Grassmannian`](@ref) manifold `M`by the
[`retraction`](@ref) from the [`GrPoint`](@ref) `x` after time 1.
The formula reads

$\xi = y \cdot (x^{\mathrm{T}}y)^{-1} - x$

This function is implemented only for the case $\mathbb{K}=\mathbb{R}$.
This is also the standard retraction.
"""
function inverseRetraction(M::Grassmannian{T},x::GrPoint{T},y::GrPoint{T}) where T<:AbstractFloat
  U = getValue(y)/(transpose(getValue(x))*getValue(y))
  A = U-getValue(x)
  GrPoint{T}(A)
end

@doc doc"""
    log(M,x,y)

compute the logarithmic map on the [`Grassmannian`](@ref) manifold
$\mathcal M=\mathrm{Gr}(k,n)$, i.e. the [`GrTVector`](@ref) whose corresponding
[`geodesic`](@ref) starting from [`GrPoint`](@ref) `x` reaches the
[`GrPoint`](@ref) `y` after time 1 on the [`Grassmannian`](@ref) manifold `M`.
The formula reads

$\log_x y = V\cdot \operatorname{atan}(S) \cdot {\bar U}^\mathrm{T}$

where $U$ and $V$ are the unitary matrices and $S$ is a diagonal matrix containing the
singular values of the SVD-decomposition of

$USV = ({\bar y}^\mathrm{T}x)^{-1} ( {\bar y}^\mathrm{T} - {\bar y}^\mathrm{T}x{\bar x}^\mathrm{T} )$

and the $\operatorname{atan}$ is meant elementwise.
"""
function log(M::Grassmannian{T},x::GrPoint{T},y::GrPoint{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  Z = getValue(y)'*getValue(x)
  if det(Z)≠0
   W = Z\(getValue(y)'-Z*getValue(x)')
   d = svd(W, full = false)
   A = (d.V)*atan(Diagonal(svd(W, full = false).S))*(d.U)'
   GrTVector{T}(A)
  else
   throw( ErrorException("The points $x and $y are antipodal, thus these input parameters are invalid.") )
  end
end

@doc doc"""
    manifoldDimension(x)

return the dimension of the [`Grassmannian`](@ref) manifold `M`$= \mathrm{Gr}(k,n)$, the
[`GrPoint`](@ref) `x`, itself embedded in $\mathbb{K}^{n\times k}$, belongs to.
The dimension for $\mathbb{K}=\mathbb{R}$ is defined by

$k(n-k)$

and for $\mathbb{K}=\mathbb{C}$

$2k \cdot (n-k).$
"""
function manifoldDimension(x::GrPoint{T}) where T<:AbstractFloat
	return Int(size(getValue(x),2) * (size(getValue(x),1) - size(getValue(x),2)))
end

function manifoldDimension(x::GrPoint{T}) where T<:Complex{U} where U<:AbstractFloat
	return Int(2 * size(getValue(x),2) * (size(getValue(x),1) - size(getValue(x),2)))
end

@doc doc"""
    manifoldDimension(M)

return the dimension of the [`Grassmannian`](@ref) manifold `M`.
The dimension for $\mathbb{K}=\mathbb{R}$ is defined by

$k(n-k)$

and for $\mathbb{K}=\mathbb{C}$

$2k \cdot (n-k).$
"""
function manifoldDimension(M::Grassmannian{T}) where T<:AbstractFloat
	return Int(M.dimensionsubspace * (M.dimensionvecspace - M.dimensionsubspace))
end

function manifoldDimension(M::Grassmannian{T}) where T<:Complex{U} where U<:AbstractFloat
	return Int(2 * M.dimensionsubspace * (M.dimensionvecspace - M.dimensionsubspace))
end

@doc doc"""
    norm(M,x,ξ)

compute the norm of the [`GrTVector`](@ref) `ξ` in the tangent space
$T_x\mathcal M$ at [`GrPoint`](@ref) `x` of the [`Grassmannian`](@ref) manifold `M`.

$\lVert \xi \rVert_x = \sqrt{\sum_{i,j=0}^n \xi_{ij}^2}$

where $\xi_{ij}$ are the entries of the matrix `ξ`, i.e. the norm is given by
the Frobenius norm of `ξ`.
"""
function norm(M::Grassmannian{T}, x::GrPoint{T}, ξ::GrTVector{T})::Float64 where T<:Union{U, Complex{U}} where U<:AbstractFloat
  norm(getValue(ξ))
end

@doc doc"""
    parallelTransport(M,x,y,ξ)

compute the paralllel transport of the [`GrTVector`](@ref) `ξ` from
the tangent space $T_x\mathcal M$ at [`GrPoint`](@ref) `x` to
$T_y\mathcal M$ at [`GrPoint`](@ref) `y` on the [`Grassmannian`](@ref) manifold `M` provided
that the corresponding [`geodesic`](@ref) $g(\cdot;x,y)$ is unique.
The formula reads

$P_{x\to y}(\xi) = \operatorname{proj}_{\mathcal M}(y,\xi).$

where $\operatorname{proj}_{\mathcal M}$ is the [`projection`](@ref) onto the
tangent space $T_y\mathcal M$.
"""
function parallelTransport(M::Grassmannian{T}, x::GrPoint{T}, y::GrPoint{T}, ξ::GrTVector{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
 GrTVector(projection(M,y,getValue(ξ)))
end

@doc doc"""
    projection(M,x,q)

project a matrix q orthogonally on the [`GrPoint`](@ref) `x` of the manifold
[`Grassmannian`](@ref) manifold `M`. The formula reads

$\operatorname{proj}_{\mathcal M}(x,q) = q-x({\bar x}^\mathrm{T}q),$

i.e. the difference matrix of the image and the output matrix lies in
the orthogonal complement of all [`GrTVector`](@ref)s from the tangent space
$T_x\mathcal M$ at [`GrPoint`](@ref) `x`.

# see also
[`parallelTransport`](@ref), [`randomTVector`](@ref)
"""
function projection(M::Grassmannian{T},x::GrPoint{T},q::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  A = getValue(x)'*q
  B = q - getValue(x)*A
  return B
end

@doc doc"""
    randomMPoint(M [,type=:Gaussian, σ=1.0])

return a random [`GrPoint`](@ref) `x` on [`Grassmannian`](@ref) manifold `M` by
generating a random (Gaussian) matrix with standard deviation `σ` in matching
size, which is orthonormal.
"""
function randomMPoint(M::Grassmannian{T}, ::Val{:Gaussian}, σ::Float64=1.0) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  V = σ * randn(T, (M.dimensionvecspace, M.dimensionsubspace))
  A = qr(V).Q[:,1:M.dimensionsubspace]
  GrPoint{T}( A )
end


@doc doc"""
    randomTVector(M,x [,type=:Gaussian, σ=1.0])

return a (Gaussian) random vector [`GrTVector`](@ref) in the tangential space
$T_x\mathrm{Gr}(k,n)$ with mean zero and standard deviation `σ` by projecting
a random Matrix onto the  [`GrPoint`](@ref) `x` with [`projection`](@ref).
"""
function randomTVector(M::Grassmannian{T},x::GrPoint{T}, ::Val{:Gaussian}, σ::Float64=1.0) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  y = σ * randn(T, (M.dimensionvecspace,M.dimensionsubspace))
  Y = projection(M, x, y)
  Y = Y/norm(Y)
  GrTVector(Y)
end

@doc doc"""
    retraction(M,x,ξ,[t=1.0])

move the [`GrPoint`](@ref) `x` in the direction of the [`GrTVector`](@ref) `ξ`
on the  [`Grassmannian`](@ref) manifold `M`. This SVD-based retraction is an
approximation of the exponential map [`exp`](@ref). Let

$USV = x + tξ$

be the singular value decomposition, then the formula reads

$\operatorname{retr}_x\xi = U{\bar V}^\mathrm{T}.$
"""
function retraction(M::Grassmannian{T},x::GrPoint{T},ξ::GrTVector{T},t::Float64=1.0) where T<:Union{U, Complex{U}} where U<:AbstractFloat
	y = getValue(x) + t * getValue(ξ)
    S = svd(y)
    A = S.U * S.V'
    GrPoint{T}(A)
end

@doc doc"""
    zeroTVector(M,x)

return a zero tangent vector in the tangent space of the [`GrPoint`](@ref) on
the [`Grassmannian`](@ref) manifold `M`.
"""
function zeroTVector(M::Grassmannian{T}, x::GrPoint{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  return GrTVector{T}(zeros(M.dimensionvecspace,M.dimensionsubspace))
end

function validateMPoint(M::Grassmannian{T}, x::GrPoint{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  if size(getValue(x), 1) ≠ M.dimensionvecspace
    throw( ErrorException("The dimension of $x must be $(M.dimensionvecspace) × $(M.dimensionsubspace) but it is $(size(getValue(x), 1)) × $(size(getValue(x), 2)).") )
  end
  if size(getValue(x), 2) ≠ M.dimensionsubspace
    throw( ErrorException("The dimension of $x must be $(M.dimensionvecspace) × $(M.dimensionsubspace) but it is $(size(getValue(x), 1)) × $(size(getValue(x), 2)).") )
  end
  if rank(getValue(x)) ≠ M.dimensionsubspace
    throw( ErrorException("$x must have  full rank.") )
  end

  z=getValue(x)'*getValue(x)
  if z ≉ one(z)
    throw( ErrorException("$x has to be orthonormal but it's not.") )
  end
end

function validateTVector(M::Grassmannian{T}, x::GrPoint{T}, ξ::GrTVector{T},) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  if size(getValue(ξ), 1) ≠ M.dimensionvecspace
    throw( ErrorException("The dimension of $ξ must be $(M.dimensionvecspace) × $(M.dimensionsubspace) but it is $(size(getValue(ξ), 1)) × $(size(getValue(ξ), 2))") )
  end
  if size(getValue(ξ), 2) ≠ M.dimensionsubspace
    throw( ErrorException("The dimension of $ξ must be $(M.dimensionvecspace) × $(M.dimensionsubspace) but it is $(size(getValue(ξ), 1)) × $(size(getValue(ξ), 2))") )
  end
  if norm(getValue(x)'*getValue(ξ) + getValue(ξ)'*getValue(x)) > 10^(-15)
    throw( ErrorException("The matrix $x'$ξ must be skew-symmetric!") )
  end
end
