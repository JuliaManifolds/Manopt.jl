#
#      Sn - The manifold of the n-dimensional sphere
#  Point is a Point on the n-dimensional sphere.
#
import LinearAlgebra: norm, dot, nullspace
import Base: exp, log, show, cat
export Sphere, SnPoint, SnTVector,show, getValue
export distance, dot, exp, log, manifoldDimension, norm
export randomMPoint, opposite, parallelTransport, zeroTVector
export validateMPoint, validateTVector
export zeroTVector, typeofMPoint, typeofTVector
#
# Type definitions
#
@doc doc"""
    Sphere <: Manifold

The manifold $\mathcal M = \mathbb S^n$ of unit vectors in $\mathbb R^{n+1}$.
This manifold is a matrix manifold (see [`IsMatrixM`](@ref)) and embedded (see
[`IsEmbeddedM`](@ref)).

# Abbreviation

`Sn`

# Constructor

    Sphere(n)

generate the sphere $\mathbb S^n$

`
Its abbreviation is `Sn`.
"""
struct Sphere <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Sphere(dimension::Int) = new("$dimension-Sphere",dimension,"Sn($(dimension-1))")
end
@doc doc"""
    SnPoint <: MPoint

A point $x$ on the manifold $\mathcal M = \mathbb S^n$ represented by a unit
vector from $\mathbb R^{n+1}$
"""
struct SnPoint{T<:AbstractFloat} <: MPoint
  value::Vector{T}
  SnPoint{T}(value::Vector{T}) where {T <: AbstractFloat} = new(value)
end
SnPoint(value::Vector{T}) where {T <: AbstractFloat} = SnPoint{T}(value)
getValue(x::SnPoint) = x.value;

@doc doc"""
    SnTVector <: TVector

A tangent vector $\xi \in T_x\mathcal M$ on the manifold
$\mathcal M = \mathbb S^n$. For the representation the tangent space can be
given as $T_x\mathbb S^n = \bigl\{\xi \in \mathbb R^{n+1}
\big| \langle x,\xi\rangle = 0\bigr\}$, where $\langle\cdot,\cdot\rangle$
denotes the Euclidean inner product on $\mathbb R^{n+1}$.
"""
struct SnTVector{T <: AbstractFloat} <: TVector
  value::Vector{T}
  SnTVector{T}(value::Vector{T}) where {T <: AbstractFloat} = new(value)
end
SnTVector(value::Vector{T})  where {T <: AbstractFloat}  = SnTVector{T}(value)
getValue(ξ::SnTVector) = ξ.value;
# Traits
# ---
# (a) Sn is a MatrixManifold
@traitimpl IsMatrixM{Sphere}
@traitimpl IsMatrixP{SnPoint}
@traitimpl IsMatrixTV{SnTVector}
# (b) Sn is Embedded
@traitimpl IsEmbeddedM{Sphere}
@traitimpl IsEmbeddedP{SnPoint}
@traitimpl IsEmbeddedV{SnTVector}

# Functions
# ---
@doc doc"""
    distance(M,x,y)

Compute the Riemannian distance on $\mathcal M=\mathbb S^n$ embedded in
$\mathbb R^{n+1}$, which is given by

$ d_{\mathbb S^n}(x,y) = \operatorname{acos} \bigl(\langle x,y\rangle\bigr), $

where $\langle\cdot,\cdot\rangle$ denotes the Euclidean inner product
on $\mathbb R^{n+1}$.
"""
distance(M::Sphere,x::SnPoint,y::SnPoint) = acos(
	min( max(dot(getValue(x), getValue(y) ), -1.), 1. )
)

@doc doc"""
    dot(M,x,ξ,ν)
Compute the Riemannian inner product for two [`SnTVector`](@ref)s `ξ` and `ν`
from $T_x\mathcal M$ of the [`Sphere`](@ref) `M` given by
$\langle \xi, \nu \rangle_x = \langle \xi,\nu \rangle$, i.e. the inner product
in the embedded space $\mathbb R^{n+1}$.
"""
dot(M::Sphere, x::SnPoint, ξ::SnTVector, ν::SnTVector) = dot( getValue(ξ), getValue(ν) )

@doc doc"""
    exp(M,x,ξ[, t=1.0])
Compute the exponential map on the [`Sphere`](@ref) `M`$=\mathbb S^n$ with
respect to the [`SnPoint`](@ref) `x` and the [`SnTVector`](@ref) `ξ`, which can
be shortened with `t` to `tξ`. The formula reads

$\exp_x\xi = \cos(\lVert\xi\rVert_2)x + \sin(\lVert\xi\rVert_2)\frac{\xi}{\lVert\xi\rVert_2}.$
"""
function exp(M::Sphere,x::SnPoint,ξ::SnTVector,t::Float64=1.0)
  len = norm( getValue(ξ) )
  if len < eps(Float64)
    return x
  else
	xV = cos(t*len) * getValue(x)  +  (sin(t*len)/len) * getValue(ξ)
    return SnPoint( xV ./ norm(xV)  )
  end
end
@doc doc"""
    log(M,x,y)
Compute the logarithmic map on the [`Sphere`](@ref)
$\mathcal M=\mathbb S^n$, i.e. the [`SnTVector`](@ref) whose corresponding
[`geodesic`](@ref) starting from [`SnPoint`](@ref) `x` reaches the
[`SnPoint`](@ref)` y` after time 1 on the [`Sphere`](@ref) `M`.
The formula reads for $x\neq -y$

$\log_x y = d_{\mathbb S^n}(x,y)\frac{y-\langle x,y\rangle x}{\lVert y-\langle x,y\rangle x \rVert_2}.$
"""
function log(M::Sphere,x::SnPoint,y::SnPoint)
  scp = dot( getValue(x), getValue(y) )
  ξvalue = getValue(y) - scp*getValue(x)
  ξvnorm = norm(ξvalue)
  if (ξvnorm > eps(Float64))
    return SnTVector( ξvalue * acos(min(max(scp,-1.),1.) )/ξvnorm );
  else
    return zeroTVector(M,x)
  end
end
@doc doc"""
    manifoldDimension(x)
returns the dimension of the [`Sphere`](@ref) `M`$=\mathbb S^n$, the
[`SnPoint`](@ref) `x`, itself embedded in $\mathbb R^{n+1}$, belongs to.
"""
manifoldDimension(x::SnPoint)::Integer = length( getValue(x) )-1
"""
    manifoldDimension(M)
returns the dimension of the [`Sphere`](@ref) `M`.
"""
manifoldDimension(M::Sphere)::Integer = M.dimension
@doc doc"""
    norm(M,x,ξ)
Computes the norm of the [`SnTVector`](@ref) `ξ` in the tangent space
$T_x\mathcal M$ at [`SnPoint`](@ref) `x` of the [`Sphere`](@ref) `M`.
"""
norm(M::Sphere, x::SnPoint, ξ::SnTVector) = norm( getValue(ξ) )
@doc doc"""
    opposite(M,x)
returns the antipodal point of x, i.e. $ y = -x $.
"""
opposite(M::Sphere, x::SnPoint) = SnPoint( -getValue(x) )
@doc doc"""
    parallelTransport(M,x,y,ξ)
Compute the paralllel transport of the [`SnTVector`](@ref) `ξ` from
the tangent space $T_x\mathcal M$ at [`SnPoint`](@ref) `x` to
$T_y\mathcal M$ at [`SnPoint`](@ref)` y` on the [`Sphere`](@ref) `M` provided
that the corresponding [`geodesic`](@ref) $g(\cdot;x,y)$ is unique.
The formula reads

$P_{x\to y}(\xi) = \xi - \frac{\langle \log_xy,\xi\rangle_x}{d^2_{\mathbb S^n}(x,y)}
\bigl(\log_xy + \log_yx \bigr).$
"""
function parallelTransport(M::Sphere, x::SnPoint, y::SnPoint, ξ::SnTVector)
  ν = log(M,x,y)
  νL = norm(M,x,ν)
  if νL > 0
    ν = ν/νL
    return SnTVector( getValue(ξ) - dot(M,x,ν,ξ)*( getValue(ν) + getValue(log(M,y,x))/νL) )
  else # if length of ν is 0, we have p=q and hence ξ is unchanged
    return ξ
  end
end
@doc doc"""
    randomMPoint(M [,:Gaussian, σ=1.0])

return a random point on the Sphere by projecting a normal distirbuted vector
from within the embedding to the sphere.
"""
function randomMPoint(M::Sphere, ::Val{:Gaussian}, σ::Float64=1.0)::SnPoint
	v = σ * randn(manifoldDimension(M)+1);
	return SnPoint(v./norm(v))
end
@doc doc"""
    randomTVector(M,x [,:Gaussian,σ=1.0])

return a random tangent vector in the tangent space of the [`SnPoint`](@ref)
`x` on the [`Sphere`](@ref) `M`.
"""
function randomTVector(M::Sphere, x::SnPoint, ::Val{:Gaussian}, σ::Float64=1.0)
    n = σ * randn( size( getValue(x)) ) # Gaussian in embedding
	  nP = n - dot(n,getValue(x))*getValue(x) #project to TpM (keeps Gaussianness)
	  return SnTVector( nP )
end
tangentONB(M::Sphere, x::SnPoint, y::SnPoint) = tangentONB(M,x,log(M,x,y))
function tangentONB(M::Sphere,x::SnPoint,ξ::SnTVector)
    d = manifoldDimension(M)
    A = zeros(d+1,d+1)
    A[1,:] = transpose(getValue(x))
    A[2,:] = transpose(getValue(ξ))
    V = nullspace(A)
    κ = ones(d)
    if ξ != zeroTVector(M,x)
        # if we have a nonzero direction for the geodesic, add it and it gets curvature zero from the tensor
		ξ = ξ/norm(M,x,ξ)
		V = cat(getValue(ξ),V,dims=2)
        κ[1] = 0.0 # no curvature along the geodesic direction, if x!=y
    end
    Ξ = [ SnTVector(V[:,i]) for i in 1:d ]
    return Ξ,κ
end
typeofTVector(::Type{SnPoint{T}}) where T = SnTVector{T}
typeofMPoint(::Type{SnTVector{T}}) where T = SnPoint{T} 
"""
    typicalDistance(M)

returns the typical distance on the [`Sphere`](@ref)` Sn`: π.
"""
typicalDistance(M::Sphere) = π

@doc doc"""
    validateMPoint(M,x)

validate, whether the [`SnPoint`](@ref) `x` is on the [`Sphere`](@ref) `M`$=\mathbb S^n$,
i.e. that the vector is of the correct dimension $n$ and its norm is $\lVert x \rVert = 1$.
"""
function validateMPoint(M::Sphere, x::SnPoint)
  if length(getValue(x)) ≠ M.dimension+1
    throw( ErrorException(
      "The Point $x is not on the $(M.name), since the vector dimension ($(length(getValue(x)))) is not $(M.dimension+1)."
    ))
  end
  if abs(norm(getValue(x)) - 1) >= 10^(-15)
    throw( ErrorException(
      "The Point $x is not on the $(M.name) since its norm $(norm(getValue(x))) ≠ 1"
    ))
  end
  return true
end

@doc doc"""
    validateTVector(M,x,ξ)

validate, whether the tangent vector [`SnTVector`](@ref) `ξ` is in the tangent
space of [`SnPoint`](@ref) `x` is on the [`Sphere`](@ref) `M`$=\mathbb S^n$,
i.e. that all three lengths are correct and $x^\mathrm{T}\xi = 0$.
"""
function validateTVector(M::Sphere,x::SnPoint,ξ::SnTVector)
  if (length(getValue(x)) ≠ length(getValue(ξ))) || (length(getValue(x)) ≠ M.dimension+1)
    throw( ErrorException(
      "The three dimensions of the $(M.name), the point x ($(length(getValue(x)))), and the tangent vector ($(length(getValue(ξ)))) don't match."
    ))
  end
  if dot(getValue(x),getValue(ξ)) >= 10^(-15)
    throw( ErrorException(
      "The tangent vector $ξ is not a tangent vector to $x on $(M.name) since the inner product is $( dot(getValue(x),getValue(ξ)) ) and not zero."
    ))
  end
  return true
end

@doc doc"""
    ξ = zeroTVector(M,x)

returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`SnPoint`](@ref) $x\in\mathbb S^n$ on the [`Sphere`](@ref)` Sn`.
"""
zeroTVector(M::Sphere, x::SnPoint) = SnTVector(  zero( getValue(x) )  );
# Display
# ---
show(io::IO, M::Sphere) = print(io,"The $(M.name)")
show(io::IO, p::SnPoint) = print(io, "Sn($( getValue(p) ))")
show(io::IO, ξ::SnTVector) = print(io, "SnT($( getValue(ξ) ))") 