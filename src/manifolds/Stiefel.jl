#
#      V(k,n) - The Stiefel-Manifold
#  A point on this manifold is a n×k matrix.
#  A point in an tangential space is a n×k matrix.
#
import LinearAlgebra: norm, dot, nullspace, det, tr, qr, triu, rank, svd, diag, sylvester, lyap, Diagonal
import Base: exp, log, show, cat, rand, Matrix, real, atan
export Stiefel, StPoint, StTVector, getValue
export dot, exp, log, manifoldDimension, norm, parallelTransport, randomTVector, randomMPoint, retractionQR, retractionPolar, inverseRetractionPolar, inverseRetractionQR, projection, retraction, inverseRetraction
export zeroTVector
#
# Type definitions
#

@doc doc"""
    Stiefel{T<:Union{U, Complex{U}} <: Manifold

The manifold $\mathcal M = \mathrm{St}(k,n)$ represented by $n\times k$
orthogonal matrices, that are either real- or complex-valued.

# Abbreviation
  `St`

# Constructor
     Stiefel(k, n[, d=Float64])

generate the manifold $\mathrm{St}(k,n)$ where the integer `n` is the
number of rows and `k` is the number of columns of the matrices and the optional
parameter `d` sets the `DataType` of the matrix entries.
"""
struct Stiefel{T<:Union{U, Complex{U}} where U<:AbstractFloat} <: Manifold
  name::String
  dimensionlines::Int
  dimensioncolumns::Int
  abbreviation::String
  function Stiefel{T}(dimensioncolumns::Int, dimensionlines::Int) where T<:Union{U, Complex{U}} where U<:AbstractFloat
    if dimensioncolumns > dimensionlines
      throw(ErrorException("dimensioncolumns can't be bigger than dimensionlines: $dimensioncolumns > $dimensionlines"))
    else
     new("Stiefel-Manifold St($dimensioncolumns,$dimensionlines) in $T", dimensionlines, dimensioncolumns,"V($dimensioncolumns,$dimensionlines)")
    end
  end
end
Stiefel(dimensionlines::Int, dimensioncolumns::Int, D::DataType=Float64) = Stiefel{D}(dimensionlines::Int, dimensioncolumns::Int)

@doc doc"""
    StPoint <: MPoint

A point $x$ on the manifold $\mathcal M = \mathrm{St}(k,n)$ is represented by an
orthogonal matrix from $\mathbb{K}^{n\times k}.$

# Constructor
    StPoint(Matrix)

where Matrix is an orthogonal matrix of dimension $n×k$.
"""
struct StPoint{T<:Union{U, Complex{U}} where U<:AbstractFloat} <: MPoint
  value::Matrix{T}
  StPoint{T}(value::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat = new(value)
end
getValue(x::StPoint) = x.value;
StPoint(value::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat = StPoint{T}(value)

@doc doc"""
    StTVector <: TVector

A tangent vector $\xi \in T_x\mathcal M$ on the manifold
$\mathcal M = \mathrm{St}(k,n)$. The tangent space is given by as

$T_x\mathrm{St}(k,n) = \bigl\{\xi \in \mathbb{K}^{n\times k} \big| x^{\mathrm{T}}ξ+ξ^{\mathrm{T}}x=0 \bigr\}$.

# Constructor

    StTVector(ξ)

where `ξ` is an $n\times k$ `Matrix` that satisfies the above.
"""
struct StTVector{T<:Union{U, Complex{U}} where U<:AbstractFloat} <: TVector
  value::Matrix{T}
  StTVector{T}(value::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat = new(value)
end
getValue(ξ::StTVector) = ξ.value;
StTVector(value::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat = StTVector{T}(value)

#
# Traits
#
@traitimpl IsMatrixM{Stiefel}
@traitimpl IsMatrixP{StPoint}
@traitimpl IsMatrixTV{StTVector}

#
# Functions
#
@doc doc"""
    dot(M,x,ξ,ν)

compute the Riemannian inner product for two [`StTVector`](@ref)s `ξ` and `ν`
from $T_x\mathcal M$ of the [`Stiefel`](@ref) manifold `M` given by

$\langle \xi, \nu \rangle_x = \operatorname{trace}({\bar \xi}^{\mathrm{T}}\nu).$
"""
function dot(M::Stiefel{T}, x::StPoint{T}, ξ::StTVector{T}, ν::StTVector{T})::Float64 where T<:Union{U, Complex{U}} where U<:AbstractFloat
  return real(tr(getValue(ξ)'*getValue(ν)))
end

@doc doc"""
    exp(M,x,ξ [,t=1.0])

compute the exponential map on the [`Stiefel`](@ref) manifold `M`$=\mathrm{SO}(n)$ with
respect to the [`StPoint`](@ref) `x` and the [`StTVector`](@ref) `ξ`, which can
be shortened with `t` to `tξ`. The formula reads

$\operatorname{exp}_{x} tξ = \begin{pmatrix}
   x\\t\xi
 \end{pmatrix}
 \operatorname{Exp}
 \left(
 \begin{pmatrix} {\bar x}^{\mathrm{T}}\xi & -{\bar \xi}^{\mathrm{T}}\xi\\
 I_{k×k} & {\bar x}^{\mathrm{T}}\xi\end{pmatrix}
 \right)
\begin{pmatrix}  \operatorname{Exp}( -{\bar x}^{\mathrm{T}}\xi) \\ 0_{k×k}\end{pmatrix}$

where $\operatorname{Exp}$ denotes matrix exponential, and $I_{k×k}$ and
$0_{k×k} are the identity matrix and the zero matrix of dimension $k×k$,
respectively.
"""
function exp(M::Stiefel{T},x::StPoint{T},ξ::StTVector{T},t::Float64=1.0) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  Z = zeros(M.dimensioncolumns,M.dimensioncolumns)
  I = one(Z)
  Ξ = t*getValue(ξ)
  Y = [getValue(x) Ξ] * exp([getValue(x)'*Ξ -Ξ'*Ξ; I getValue(x)'*Ξ]) * [exp(-getValue(x)'*Ξ); Z]
  StPoint{T}(Y)
end

@doc doc"""
    inverseRetractionPolar(M,x,y)

return a [`StTVector`](@ref) `ξ` of the tagent space $T_x\mathrm{SO}(n)$
of the [`StPoint`](@ref) `x` on the [`Stiefel`](@ref) manifold `M`
with which the [`StPoint`](@ref) `y` can be reached by the
[`retractionPolar`](@ref)  after time 1.
The formula reads

$ξ = ys-x$

where $s$ is the solution to the Lyapunov equation

$x^{\mathrm{T}}ys + s(x^{\mathrm{T}}y)^{\mathrm{T}} + 2\mathrm{I}_k = 0.$

This function is implemented only for the case $\mathbb{K}=\mathbb{R}$.
"""
function inverseRetractionPolar(M::Stiefel{T},x::StPoint{T},y::StPoint{T}) where T<:AbstractFloat
  A = transpose(getValue(x)) * getValue(y)
  H = -2 * one(transpose(getValue(x))*getValue(x))
  B = lyap(A, H)
  C = getValue(y)*B - getValue(x)
  StTVector{T}( C )
end

@doc doc"""
    inverseRetractionQR(M,x,y)

return a [`StTVector`](@ref) `ξ` of the tagent space $T_x\mathrm{SO}(n)$
of the [`StPoint`](@ref) `x` on the [`Stiefel`](@ref) manifold `M`
with which the [`StPoint`](@ref) `y` can be reached by the
[`retractionQR`](@ref) after time 1.
This function is implemented only for the case $\mathbb{K}=\mathbb{R}$.
This is also the standard retraction.
"""
function inverseRetractionQR(M::Stiefel{T},x::StPoint{T},y::StPoint{T}) where T<:AbstractFloat
  A = transpose(getValue(x)) * getValue(y)
  R = zeros(M.dimensioncolumns, M.dimensioncolumns)
  for i = 1:M.dimensioncolumns
    b = zeros(i)
    b[end] = 1
    b[1:(end-1)] = - transpose(R[1:(i-1), 1:(i-1)]) * A[i, 1:(i-1)]
    R[1:i, i] = A[1:i, 1:i] \ b
  end
  C =  getValue(y)*R-getValue(x)
  StTVector{T}( C )
end

function inverseRetraction(M::Stiefel{T},x::StPoint{T},y::StPoint{T}) where T<:AbstractFloat
  return inverseRetractionQR(M,x,y)
end

@doc doc"""
    manifoldDimension(x)

return the dimension of the [`Stiefel`](@ref) manifold `M`$= \mathrm{St}(k,n)$, the
[`StPoint`](@ref) `x`, itself embedded in $\mathbb R^{n\times k}$, belongs to.
The dimension for $\mathbb{K}=\mathbb{R}$ is given by

$nk - \frac{1}{2}k(k+1)$

and for $\mathbb{K}=\mathbb{C}$

$2nk - k^2.$
"""
function manifoldDimension(x::StPoint{T}) where T<:AbstractFloat
  return Int(size(getValue(x),1) * size(getValue(x),2) - 0.5 * size(getValue(x),2) * (size(getValue(x),2) + 1))
end

function manifoldDimension(x::StPoint{T}) where T<:Complex{U} where U<:AbstractFloat
  return Int(2 * size(getValue(x),1) * size(getValue(x),2) - size(getValue(x),2)^2)
end

@doc doc"""
    manifoldDimension(M)

return the dimension of the [`Stiefel`](@ref) manifold `M`.
The dimension for $\mathbb{K}=\mathbb{R}$ is given by

$nk - \frac{1}{2}k(k+1)$

and for $\mathbb{K}=\mathbb{C}$

$2nk - k^2.$
"""
function manifoldDimension(M::Stiefel{T}) where T<:AbstractFloat
  return Int(M.dimensionlines * M.dimensioncolumns - 0.5 * M.dimensioncolumns * (M.dimensioncolumns + 1))
end
function manifoldDimension(M::Stiefel{T}) where T<:Complex{U} where U<:AbstractFloat
  return Int(2 * M.dimensionlines * M.dimensioncolumns - (M.dimensioncolumns)^2)
end

@doc doc"""
    norm(M,x,ξ)

compute the norm of the [`StTVector`](@ref) `ξ` in the tangent space
$T_x\mathcal M$ at [`StPoint`](@ref) `x` of the [`Stiefel`](@ref) manifold `M`.

$\lVert \xi \rVert_x = \sqrt{\sum_{i,j=0}^n \xi_{ij}^2}$

where $\xi_{ij}$ are the entries of the matrix `ξ`, i.e. the norm is given by
the Frobenius norm of `ξ`.
"""
function norm(M::Stiefel{T}, x::StPoint{T}, ξ::StTVector{T})::Float64 where T<:Union{U, Complex{U}} where U<:AbstractFloat
  norm(getValue(ξ))
end

@doc doc"""
    parallelTransport(M,x,y,ξ)

compute the paralllel transport of the [`StTVector`](@ref) `ξ` from
the tangent space $T_x\mathcal M$ at [`StPoint`](@ref) `x` to
$T_y\mathcal M$ at [`StPoint`](@ref) `y` on the [`Stiefel`](@ref) manifold `M`
provided that the corresponding [`geodesic`](@ref) $g(\cdot;x,y)$ is unique.
The formula reads

$P_{x\to y}(\xi) = \operatorname{proj}_{\mathcal M}(y,\xi).$

where $\operatorname{proj}_{\mathcal M}$ is the projection onto the
tangent space $T_y\mathcal M$.
"""
function parallelTransport(M::Stiefel{T}, x::StPoint{T}, y::StPoint{T}, ξ::StTVector{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
    projection(M,y,getValue(ξ))
end

@doc doc"""
    projection(M,x,q)

project a `Matrix q` orthogonally on the tangent space of the
[`StPoint`](@ref) `x` on the [`Stiefel`](@ref) manifold `M`. The formula reads

$\operatorname{proj}_{\mathcal M}(x,q) = q-xB$

where

$B=\frac{1}{2} (x^{\mathrm{T}}{\bar q})^{\mathrm{T}} {\bar x}^{\mathrm{T}}q.$

# see also
[`parallelTransport`](@ref), [`randomTVector`](@ref)
"""
function projection(M::Stiefel{T}, x::StPoint{T}, q::Matrix{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
    A = getValue(x)'*q
    B = 0.5 * (A' + A)
    return StTVector{T}(q - getValue(x) * B)
end

@doc doc"""
    randomMPoint(M [,:Gaussian, σ=1.0])

return a random (Gaussian) [`StPoint`](@ref) `x` on the manifold
[`Stiefel`](@ref) manifold `M` by generating a (Gaussian) matrix with standard deviation
`σ` and return the orthogonalized version, i.e. return ​​the Q
component of the QR decomposition of the random matrix of size $n×k$.
"""
function randomMPoint(M::Stiefel{T}, ::Val{:Gaussian}, σ::Float64=1.0) where T<:Union{U, Complex{U}} where U<:AbstractFloat
    A = σ*randn(T, (M.dimensionlines, M.dimensioncolumns))
    return StPoint{T}( Matrix(qr(A).Q) )
end

@doc doc"""
    randomTVector(M,x [,:Gaussian, σ=1.0])

return a random vector [`StTVector`](@ref) in the tangential space
$T_x\mathrm{St}(k,n)$ by generating a random matrix of size $n×k$ and projecting
it onto [`StPoint`](@ref) `x` with [`projection`](@ref).
"""
function randomTVector(M::Stiefel{T}, x::StPoint{T}, ::Val{:Gaussian}, σ::Float64=1.0) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  y = σ * randn(T, (M.dimensionlines,M.dimensioncolumns))
  Y = projection(M, x, y)
  return 1/(norm(getValue(Y))) * Y
end

@doc doc"""
    retractionPolar(M,x,ξ,[t=1.0])

move the [`StPoint`](@ref) `x` in the direction of the [`StTVector`](@ref) `ξ`
on the  [`Stiefel`](@ref) manifold `M`. This SVD-based retraction is a second-order
approximation of the exponential map [`exp`](@ref). Let

$USV = x + tξ$

be the singular value decomposition, then the formula reads

$\operatorname{retr}_x\xi = UV^\mathrm{T}.$

This function is implemented only for the case $\mathbb{K}=\mathbb{R}$.
"""
function retractionPolar(M::Stiefel{T},x::StPoint{T},ξ::StTVector{T},t::Float64=1.0) where T<:AbstractFloat
  y = getValue(x) + t * getValue(ξ)
  S = svd(y)
  A = S.U * transpose(S.V)
  StPoint{T}(A)
end

@doc doc"""
    retractionQR(M,x,ξ,[t=1.0])

move the [`StPoint`](@ref) `x` in the direction of the [`StTVector`](@ref) `ξ`
on the  [`Stiefel`](@ref) manifold `M`. This QR-based retraction is a
first-order approximation of the exponential map [`exp`](@ref). Let

$QR = x + tξ$

be the QR decomposition, then the formula reads

$\operatorname{retr}_x\xi = QD$

where D is a $n×k$ matrix with the signs of the diagonal entries of $R$ plus
$0.5$ on the upper diagonal, i.e.

$D_{ij}=\begin{cases} \operatorname{sgn}(R_{ij}+0,5) & \text{if} \; i=j \\ 0 & \, \text{otherwise} \end{cases}.$

This is also the standard retraction.
"""
function retractionQR(M::Stiefel{T},x::StPoint{T},ξ::StTVector{T},t::Float64=1.0) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  y = getValue(x) + t * getValue(ξ)
  QRdecomp = qr(y)
  d = diag(QRdecomp.R)
  D = Diagonal( sign.( sign.(d .+ 0.5) ) )
  B = zeros(M.dimensionlines,M.dimensioncolumns)
  B[1:M.dimensioncolumns,1:M.dimensioncolumns] = D
  A = QRdecomp.Q * B
  StPoint{T}(A)
end

function retraction(M::Stiefel{T},x::StPoint{T},ξ::StTVector{T},t::Float64=1.0) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  return retractionQR(M,x,ξ,t)
end

@doc doc"""
    ξ = zeroTVector(M,x)

return a zero vector in the tangent space $T_x\mathcal M$ of the
[`StPoint`](@ref) $x\in\mathrm{St}(k,n)$ on the [`Stiefel`](@ref) manifold `M`.
"""
function zeroTVector(M::Stiefel{T},x::StPoint{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  return StTVector{T}( zero(getValue(x)) )
end

@doc doc"""
    validateMPoint(M,x)

validate that the [`StPoint`](@ref) `x` is a valid point on the
[`Stiefel`](@ref) manifold `M`, i.e. that both dimensions and the rank are correct
as well as that all columns are orthonormal.
"""
function validateMPoint(M::Stiefel{T}, x::StPoint{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  if size(getValue(x), 1) ≠ M.dimensionlines
    throw( ErrorException("The dimension of $x must be $(M.dimensionlines) × $(M.dimensioncolumns) but it is $(size(getValue(x), 1)) × $(size(getValue(x), 2)).") )
  end
  if size(getValue(x), 2) ≠ M.dimensioncolumns
    throw( ErrorException("The dimension of $x must be $(M.dimensionlines) × $(M.dimensioncolumns) but it is $(size(getValue(x), 1)) × $(size(getValue(x), 2)).") )
  end
if rank(getValue(x)) ≠ M.dimensioncolumns #is this necessary?
    throw( ErrorException("$x must have  full rank.") )
  end

  z=getValue(x)'*getValue(x)
  if z ≉ one(z)
    throw( ErrorException("$x has to be orthonormal but it's not.") )
  end
end
@doc doc"""
    validateMPoint(M,x,ξ)

validate that the [`StTVector`](@ref) `ξ` is a valid tangent vector to
[`StPoint`](@ref) `x` on the [`Stiefel`](@ref) manifold `M`, i.e. that both
dimensions agree and ${\bar x}^{\mathrm{T}}$ is skew symmetric.
"""
function validateTVector(M::Stiefel{T}, x::StPoint{T}, ξ::StTVector{T}) where T<:Union{U, Complex{U}} where U<:AbstractFloat
  if size(getValue(ξ), 1) ≠ M.dimensionlines
    throw( ErrorException("The dimension of $ξ must be $(M.dimensionlines) × $(M.dimensioncolumns) but it is $(size(getValue(ξ), 1)) × $(size(getValue(ξ), 2))") )
  end
  if size(getValue(ξ), 2) ≠ M.dimensioncolumns
    throw( ErrorException("The dimension of $ξ must be $(M.dimensionlines) × $(M.dimensioncolumns) but it is $(size(getValue(ξ), 1)) × $(size(getValue(ξ), 2))") )
  end
  if norm(getValue(x)'*getValue(ξ) + getValue(ξ)'*getValue(x)) > 10^(-15)
    throw( ErrorException("The matrix $x'$ξ must be skew-symmetric!") )
  end
end
