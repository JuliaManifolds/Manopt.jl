#
#      SO(n) - The manifold of the special orthogonal group
#  A point on the manifold is a orthogonal matrix with determinant +1.
#  A point in an tangential space is a skew-symmetric matrix.
#
import LinearAlgebra: Diagonal, norm, dot, nullspace, det, tr, qr, triu, eigvals, diag, svd, sylvester
import Base: exp, log, show, rand, Matrix
export Rotations, SOPoint, SOTVector, getValue
export addNoise, distance, dot, exp, log, manifoldDimension, norm, parallelTransport, randomTVector, randomMPoint, retractionQR, retractionPolar, inverseRetractionPolar, inverseRetractionQR, retraction, inverseRetraction
export zeroTVector
#
# Type definitions
#

@doc doc"""
    Rotations <: Manifold

The manifold $\mathcal M = \mathrm{SO}(n)$ represented by $n\times n$
real-valued orthogonal matrices with determinant $+1$.

# Abbreviation
`SO`

# Constructor
    Rotations(n)

generates the manifold $\mathrm{SO}(n)$ where the integer `n` is the
number of rows or columns of the matrices.
"""
struct Rotations <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Rotations(n::Int) = new("of the set of rotations in R$n",n, "SO($n)")
end

@doc doc"""
    SOPoint <: MPoint

A point $x$ on the manifold $\mathcal M = \mathrm{SO}(n)$ is represented by an
orthogonal matrix with determinant $+1$ from $\mathbb R^{n\times n}.$

# Constructor
    SOPoint(x)

where `x` is an orthogonal matrix with determinant $+1$ of dimension $n×n$.
"""
struct SOPoint <: MPoint
  value::Matrix
  SOPoint(value::Matrix) = new(value)
end
getValue(x::SOPoint) = x.value;

@doc doc"""
    SOTVector <: TVector

A tangent vector $\xi \in T_x\mathcal M$ on the manifold
$\mathcal M = \mathrm{SO}(n)$. The tangent space is given by as

$T_x\mathrm{SO}(n) = \bigl\{x\xi \in \mathbb R^{n\times n}
\big| \xi + \xi^T = 0 \bigr\}.$

Since the manifold of rotations is a Lie group, it suffices to store just the
skew-symmetric matrix $\xi$. This has to be taken into account in all formulae.


# Constructor
    SOTVector(ξ)

where `ξ` is an $n\times n$ `Matrix` that is skew-symmetric.
"""
struct SOTVector <: TVector
  value::Matrix
  SOTVector(value::Matrix) = new(value)
end
getValue(ξ::SOTVector) = ξ.value;

# Traits
# ---
# (a) SO(n) is a MatrixManifold
@traitimpl IsMatrixM{Rotations}
@traitimpl IsMatrixP{SOPoint}
@traitimpl IsMatrixTV{SOTVector}
# (b) SO(n) is Embedded
@traitimpl IsEmbeddedM{Rotations}
@traitimpl IsEmbeddedP{SOPoint}
@traitimpl IsEmbeddedV{SOTVector}
# (c) SO(n) is a LieGroup
@traitimpl IsLieGroupM{Rotations}
@traitimpl IsLieGroupP{SOPoint}
@traitimpl IsLieGroupV{SOTVector}
#
LieGroupOp(x::SOPoint, y::SOPoint) = SOPoint( transpose(getValue(x))*getValue(y) )

@doc doc"""
    dot(M,x,ξ,ν)

compute the Riemannian inner product for two [`SOTVector`](@ref)s `ξ` and `ν`
from $T_x\mathcal M$ of the [`Rotations`](@ref) manifold `M` given by

$\langle \xi, \nu \rangle_x = \operatorname{tr}(\xi^T\nu)$

i.e. the inner product in the embedded space $\mathbb R^{n\times n}$.
"""
dot(M::Rotations, x::SOPoint, ξ::SOTVector, ν::SOTVector)::Float64 = tr( transpose(getValue(ξ))*getValue(ν) )

@doc doc"""
    distance(M,x,y)

compute the Riemannian distance on [`Rotations`](@ref) manifold `M`
$= \mathrm{SO}(n)$ embedded in $\mathbb R^{n\times n}$, which is given by

$d(x,y) = \lVert \operatorname{log}_{x}y \rVert_x$

where $\operatorname{log}_{\cdot}\cdot$ denotes the logarithmic map on
the [`Rotations`](@ref) $\mathcal M=\mathrm{SO}(n)$.
"""
function distance(M::Rotations,x::SOPoint,y::SOPoint)
  U = transpose(getValue(x)) * getValue(y)
  if (getValue(x) != getValue(y)) && (abs.(U) == one(U))
    return π
  end
  return norm(M, x, log(M,x,y))
end


@doc doc"""
    exp(M,x,ξ,[t=1.0])

compute the exponential map on the [`Rotations`](@ref) manifold `M`$=\mathrm{SO}(n)$ with
respect to the [`SOPoint`](@ref) `x` and the [`SOTVector`](@ref) `ξ`, which can
be shortened with `t` to `tξ`. The formula reads

```math
\operatorname{exp}_{x}(tξ) = x \cdot \operatorname{Exp}(tξ)
```

where $\operatorname{Exp}$ denotes matrix exponential.
"""
exp(M::Rotations,x::SOPoint,ξ::SOTVector,t::Float64=1.0) = SOPoint(getValue(x) * exp( t * getValue(ξ)))

@doc doc"""
    log(M,x,y)

compute the logarithmic map on the [`Rotations`](@ref) manifold
`M`$=\mathrm{SO}(n)$, which is given by

```math
\operatorname{log}_{x} y =
  \frac{1}{2} \bigl(\operatorname{Log}(x^{\mathrm{T}}y)
  - (\operatorname{Log} x^{\mathrm{T}}y)^{\mathrm{T}}),
```

where $\operatorname{Log}$ denotes the matrix logarithm.
"""
function log(M::Rotations,x::SOPoint,y::SOPoint)
  U = transpose(getValue(x)) * getValue(y)
  if (getValue(x) != getValue(y)) && (abs.(U) == one(U))
    throw( ErrorException("The points $x and $y are antipodal, thus these input parameters are invalid.") )
  end
  U1 = real(log(U))
  U2 = 0.5 * (U1 - transpose(U1))
  return SOTVector(U2)
end

@doc doc"""
    manifoldDimension(x)

return the dimension of the [`Rotations`](@ref) manifold `M`$= \mathrm{SO}(n)$, the
[`SOPoint`](@ref) `x`, itself embedded in $\mathbb R^{n\times n}$, belongs to.
The dimension is defined by

$\frac{n(n-1)}{2}.$
"""
manifoldDimension(x::SOPoint)::Integer = size(getValue(x), 1)*(size(getValue(x), 2)-1)/2

@doc doc"""
    manifoldDimension(M)

return the dimension of the [`Rotations`](@ref) manifold `M`$= \mathrm{SO}(n)$.
The dimension is defined by

$\frac{n(n-1)}{2}.$
"""
manifoldDimension(M::Rotations)::Integer = (M.dimension*(M.dimension-1))/2

@doc doc"""
    norm(M,x,ξ)

compute the norm of the [`SOTVector`](@ref) `ξ` in the tangent space
$T_x\mathcal M$ at [`SOPoint`](@ref) `x` of the [`Rotations`](@ref) manifold `M`.

$\lVert \xi \rVert_x = \sqrt{\sum_{i,j=0}^n \xi_{ij}^2}$

where $\xi_{ij}$ are the entries of the skew-symmetric matrix `ξ`, i.e. the norm
is given by the Frobenius norm of `ξ`.
"""
norm(M::Rotations, x::SOPoint, ξ::SOTVector) = norm(getValue(ξ))

@doc doc"""
    parallelTransport(M,x,y,ξ)

compute the parallel transport of the [`SOTVector`](@ref) `ξ` from
the tangent space $T_x\mathcal M$ at [`SOPoint`](@ref) `x` to
$T_y\mathcal M$ at [`SOPoint`](@ref) `y` on the [`Rotations`](@ref) `M` along
$g$ provided that the corresponding [`geodesic`](@ref) $g(\cdot;x,y)$ is unique.
Since we have only stored the skew-symmetric matrix as a
[`SOTVector`](@ref) `ξ`, the function returns the the [`SOTVector`](@ref) `ξ`.
"""
parallelTransport(M::Rotations,x::SOPoint,y::SOPoint,ξ::SOTVector) = ξ

@doc doc"""
    randomTVector(M,x[, type=:Gaussian, σ=1.0])

return a random [`SOTVector`](@ref) in the tangent space
$T_x\mathrm{SO}(n)$ of the [`SOPoint`](@ref) `x` on the [`Rotations`](@ref)
manifold `M` by generating a random skew-symmetric matrix. The function
takes the real upper triangular matrix of a (Gaussian) random matrix $A$ with
dimension $n\times n$ and subtracts its transposed matrix.
Finally, the matrix is ​​normalized.
"""
function randomTVector(M::Rotations,x::SOPoint, ::Val{:Gaussian}, σ::Real=1.0)
  if M.dimension==1
    SOTVector(zeros(1,1))
  else
    A = randn(Float64, M.dimension, M.dimension)
    A = triu(A,1) - transpose(triu(A,1))
    A = (1/norm(A))*A
    SOTVector(A)
  end
end

@doc doc"""
    randomMPoint(M[, type=:Gaussian, σ=1.0])

return a random [`SOPoint`](@ref) `x` on the manifold [`Rotations`](@ref) `M`
by generating a (Gaussian) random orthogonal matrix with determinant $+1$. Let

$QR = A$

be the QR decomposition of a random matrix $A$, then the formula reads

$x = QD$

where $D$ is a diagonal matrix with the signs of the diagonal entries of $R$,
i.e.

$D_{ij}=\begin{cases} \operatorname{sgn}(R_{ij}) & \text{if} \; i=j \\ 0 & \, \text{otherwise} \end{cases}.$

It can happen that the matrix gets -1 as a determinant. In this case, the first
and second columns are swapped.
"""
function randomMPoint(M::Rotations, ::Val{:Gaussian}, σ::Real=1.0)
  if M.dimension==1
    SOPoint(ones(1,1))
  else
    A=randn(Float64, M.dimension, M.dimension)
    s=diag(sign.(qr(A).R))
    D=Diagonal(s)
    C = qr(A).Q*D
    if det(C)<0
      C[:,[1,2]] = C[:,[2,1]]
    end
    SOPoint(C)
  end
end

@doc doc"""
    retractionPolar(M,x,ξ [,t=1.0])

move the [`SOPoint`](@ref) `x` in the direction of the [`SOTVector`](@ref) `ξ`
on the  [`Rotations`](@ref) manifold `M`. This SVD-based retraction is a second-order
approximation of the exponential map. Let

$USV = x + txξ$

be the singular value decomposition, then the formula reads

$\operatorname{retr}_x\xi = UV^\mathrm{T}$
"""
function retractionPolar(M::Rotations, x::SOPoint, ξ::SOTVector, t::Float64=1.0)
  y = getValue(x) + t * getValue(x) * getValue(ξ)
  S = svd(y)
  A = S.U * transpose(S.V)
  SOPoint(A)
end

@doc doc"""
    retractionQR(M,x,ξ [,t=1.0])

move the [`SOPoint`](@ref) `x` in the direction of the [`SOTVector`](@ref) `ξ`
on the [`Rotations`](@ref) manifold `M`. This QR-based retraction is a
first-order approximation of the exponential map. Let

$QR = x + txξ$

be the QR decomposition, then the formula reads

$\operatorname{retr}_x\xi = QD$

where the matrix $D$ is given by

```math
D_{ij}=\begin{cases}
\operatorname{sgn}(R_{ij}+0,5) & \text{if} \; i=j \\
0 & \, \text{otherwise.}
\end{cases}
```
"""
function retractionQR(M::Rotations, x::SOPoint, ξ::SOTVector, t::Float64=1.0)
  y = getValue(x) + t * getValue(x)*getValue(ξ)
  QRdecomp = qr(y)
  d = diag(QRdecomp.R)
  D = Diagonal( sign.( sign.(d .+ 0.5) ) )
  A = QRdecomp.Q * D
  SOPoint(A)
end
retraction(M::Rotations, x::SOPoint, ξ::SOTVector, t::Float64=1.0) = retractionQR(M,x,ξ,t)

@doc doc"""
    inverseRetractionPolar(M,x,y)

return a [`SOTVector`](@ref) `ξ` of the tagent space $T_x\mathrm{SO}(n)$
of the [`SOPoint`](@ref) `x` on the [`Rotations`](@ref) manifold `M`
with which the [`SOPoint`](@ref) `y` can be reached by the
[`retractionPolar`](@ref) after time 1. The formula reads

$ξ = -\frac{1}{2}(x^{\mathrm{T}}ys - (x^{\mathrm{T}}ys)^{\mathrm{T}})$

where $s$ is the solution to the Sylvester equation

$x^{\mathrm{T}}ys + s(x^{\mathrm{T}}y)^{\mathrm{T}} + 2\mathrm{I}_n = 0.$
"""
function inverseRetractionPolar(M::Rotations, x::SOPoint, y::SOPoint)
  A = transpose(getValue(x)) * getValue(y)
  H = 2 * one(getValue(x))
  K = convert(Array{Float64,2}, transpose(A))
  B = sylvester(A, K, H)
  C =  A * B
  SOTVector( -0.5 * ( C - transpose(C) ) )
end

@doc doc"""
    inverseRetractionQR(M,x,y)

return a [`SOTVector`](@ref) `ξ` of the tagent space $T_x\mathrm{SO}(n)$
of the [`SOPoint`](@ref) `x` on the [`Rotations`](@ref) manifold `M`
with which the [`SOPoint`](@ref) `y` can be reached by the
[`retractionQR`](@ref) from the [`SOPoint`](@ref) `x` after time 1.
"""
function inverseRetractionQR(M::Rotations, x::SOPoint, y::SOPoint)
  A = transpose(getValue(x)) * getValue(y)
  R = zeros(M.dimension, M.dimension)
  for i = 1:M.dimension
    b = zeros(i)
    b[end] = 1
    b[1:(end-1)] = - transpose(R[1:(i-1), 1:(i-1)]) * A[i, 1:(i-1)]
    R[1:i, i] = A[1:i, 1:i] \ b
  end
  C =  A * R
  SOTVector( 0.5 * ( C - transpose(C) ) )
end
inverseRetraction(M::Rotations, x::SOPoint, y::SOPoint) = inverseRetractionQR(M,x,y)

@doc doc"""
    zeroTVector(M,x)

return a zero [`SOTVector`](@ref) $\xi$ from the tagent space $T_x\mathrm{SO}(n)$
of [`SOPoint`](@ref) `x` on the [`Rotations`](@ref) manifold `M`, i.e. a zero
matrix.
"""
zeroTVector(M::Rotations, x::SOPoint) = SOTVector( zero(getValue(x)) )

function validateMPoint(M::Rotations, x::SOPoint)
  if det(getValue(x)) ≉ 1
    throw( ErrorException("The determinant of $x has to be +1 but it is $(det(getValue(x))).") )
  end
  if transpose(getValue(x))*getValue(x) ≉ one(getValue(x))
    throw( ErrorException("$x has to be orthogonal but it's not.") )
  end
  return true
end

function validateTVector(M::Rotations, x::SOPoint, ξ::SOTVector)
  if transpose(getValue(ξ))+getValue(ξ) ≉ zero(getValue(ξ))
    throw( ErrorException("$ξ has to be skew-symmetric but it's not.") )
  end
  return true
end
