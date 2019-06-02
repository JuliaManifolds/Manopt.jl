#
#      Hn - The manifold of the n-dimensional sphere
#  Point is a Point on the n-dimensional sphere.
#
import LinearAlgebra: norm, dot
import Base: exp, log, show
export Hyperbolic, HnPoint, HnTVector, getValue
export distance, dot, exp, log, manifoldDimension, norm, parallelTransport
export typeofMPoint, typeofTVector, MinkowskiDot
export validateMPoint, validateTVector, zeroTVector
#
# Type definitions
#

@doc doc"""
    Hyperbolic <: Manifold
The manifold $\mathbb H^n$ is the set

```math
\mathbb H^n = \Bigl\{x\in\mathbb R^{n+1}
\ \Big|\ \langle x,x \rangle_{\mathrm{M}}= -x_{n+1}^2
+ \displaystyle\sum_{k=1}^n x_k^2 = -1, x_{n+1} > 0\Bigr\},
```

where $\langle\cdot,\cdot\rangle_{\mathrm{M}}$ denotes the [`MinkowskiDot`](@ref)
is Minkowski inner product, and this inner product in the embedded space yields
the Riemannian metric when restricted to the tangent bundle $T\mathbb H^n$.

This manifold is a matrix manifold (see [`IsMatrixM`](@ref)) and embedded (see
[`IsEmbeddedM`](@ref)).

# Abbreviation

`Hn`

# Constructor

    Hyperbolic(n)

generates the `n`-dimensional hyperbolic manifold embedded in $\mathbb R^{n+1}$.
"""
struct Hyperbolic <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Hyperbolic(dimension::Int) = new("$dimension-Hyperbolic Space",dimension,"Hn($(dimension-1))")
end
@doc doc"""
    HnPoint <: MPoint

A point $x$ on the manifold $\mathbb H^n$ represented by a vector
$x\in\mathbb R^{n+1}$ with Minkowski inner product

$\langle x,x\rangle_{\mathrm{M}} = -x_{n+1}^2 + \sum_{k=1}^n x_k^2 = -1$..
"""
struct HnPoint{T<:AbstractFloat} <: MPoint
  value::Vector{T}
  HnPoint{T}( value::Vector{T} ) where {T<:AbstractFloat} = new(value)
  HnPoint{T}( value::T) where {T<:AbstractFloat} = new([value])
end
HnPoint(value::Vector{T}) where {T<:AbstractFloat} = HnPoint{T}(value)
HnPoint(value::T) where {T <: AbstractFloat} = HnPoint{T}(value)

getValue(x::HnPoint) = length(x.value)==1 ? x.value[1] : x.value

@doc doc"""
    HnTVector <: TVector

A tangent vector $\xi \in T_x\mathbb H^n$ to a [`HnPoint`](@ref) $x$ on the
$n$-dimensional [`Hyperbolic`](@ref) space $\mathbb H^n$. To be precise
$\xi\in\mathbb R^{n+1}$ is hyperbocally orthogonal to $x\in\mathbb R^{n+1}$,
i.e. orthogonal with respect to the Minkowski inner product

$\langle \xi, x \rangle_{\mathrm{M}} = -\xi_{n+1}x_{n+1} + \sum_{k=1}^n \xi_k x_k = 0$ 
"""
struct HnTVector{T <: AbstractFloat}  <: TVector
    value::Vector{T}
    HnTVector{T}(value::Vector{T})  where {T <: AbstractFloat}  = new(value)
    HnTVector{T}(value::T) where {T <: AbstractFloat}  = new([value])
end
HnTVector(value::T) where {T <: AbstractFloat} = HnTVector{T}(value)
HnTVector(value::Vector{T})  where {T <: AbstractFloat}  = HnTVector{T}(value)
  
getValue(ξ::HnTVector) = length(ξ.value)==1 ? ξ.value[1] : ξ.value

# Traits
# ---
# (a) Hn is a MatrixManifold
@traitimpl IsMatrixM{Hyperbolic}
@traitimpl IsMatrixP{HnPoint}
@traitimpl IsMatrixTV{HnTVector}
# (b) Hn is a MatrixManifold
@traitimpl IsEmbeddedM{Hyperbolic}
@traitimpl IsEmbeddedP{HnPoint}
@traitimpl IsEmbeddedV{HnTVector}

# Functions
# ---
@doc doc"""
    distance(M,x,y)

compute the Riemannian distance on the [`Hyperbolic`](@ref) space $\mathbb H^n$
embedded in $\mathbb R^{n+1}$ can be computed as

```math
d_{\mathbb H^n}(x,y)
= \operatorname{acosh} \bigl(-\langle x,y\rangle_{\mathrm{M}}\bigr),
```

where $\langle x,y\rangle_{\mathrm{M}} = -x_{n+1}y_{n+1} +
\displaystyle\sum_{k=1}^n x_ky_k$ denotes the [`MinkowskiDot`](@ref) Minkowski
inner product on $\mathbb R^{n+1}$.
"""
distance(M::Hyperbolic,x::HnPoint{T},y::HnPoint{T}) where {T <: AbstractFloat} = acosh(  max(1,-MinkowskiDot(getValue(x), getValue(y)))  )

@doc doc"""
    dot(M,x,ξ,ν)

compute the Riemannian inner product for two [`HnTVector`](@ref)s `ξ` and `ν`
from $T_x\mathcal M$ of the [`Hyperbolic`](@ref) space $\mathbb H^n$ given by
$\langle \xi, \nu \rangle_{\mathrm{M}}$ the [`MinkowskiDot`](@ref) Minkowski
inner product on $\mathbb R^{n+1}$.
"""
dot(M::Hyperbolic, x::HnPoint{T}, ξ::HnTVector{T}, ν::HnTVector{T}) where {T <: AbstractFloat} = MinkowskiDot( getValue(ξ), getValue(ν) )

@doc doc"""
    exp(M,x,ξ,[t=1.0])

computes the exponential map on the [`Hyperbolic`](@ref) space $\mathbb H^n$ with
respect to the [`HnPoint`](@ref) `x` and the [`HnTVector`](@ref) `ξ`, which can
be shortened with `t` to `tξ`. The formula reads

$\exp_x\xi = \cosh(\sqrt{\langle\xi,\xi\rangle_{\mathrm{M}}})x + \operatorname{sinh}(\sqrt{\langle\xi,\xi\rangle_{\mathrm{M}}})\frac{\xi}{\sqrt{\langle\xi,\xi\rangle_{\mathrm{M}}}}.$
"""
function exp(M::Hyperbolic,x::HnPoint{T},ξ::HnTVector{T},t::Float64=1.0)  where {T <: AbstractFloat}
  len = sqrt(MinkowskiDot( getValue(ξ), getValue(ξ) ));
  if len < eps(Float64)
  	return x
  else
  	return HnPoint( cosh(t*len)*getValue(x) + sinh(t*len)/len*getValue(ξ) )
  end
end
@doc doc"""
    log(M,x,y)

computes the logarithmic map on the [`Hyperbolic`](@ref) space $\mathbb H^n$,
i.e., the [`HnTVector`](@ref) whose corresponding
[`geodesic`](@ref) starting from [`HnPoint`](@ref) `x` reaches the
[`HnPoint`](@ref) `y` after time 1 on the [`Hyperbolic`](@ref) space
$\mathbb H^n$.
The formula reads for $x\neq y$

```math
\log_x y = d_{\mathbb H^n}(x,y)\frac{y-\langle x,y\rangle_{\mathrm{M}} x}{\lVert y-\langle x,y\rangle_{\mathrm{M}} x \rVert_2}
```
and is zero otherwise.
"""
function log(M::Hyperbolic,x::HnPoint{T},y::HnPoint{T}) where {T <: AbstractFloat}
  scp = MinkowskiDot( getValue(x), getValue(y) )
  ξvalue = getValue(y) + scp*getValue(x)
  ξvnorm = sqrt(max(scp^2 - 1,0));
  if (ξvnorm > eps(Float64))
    return HnTVector( ξvalue*acosh(max(1.,-scp))/ξvnorm )
  else
    return zeroTVector(M,x)
  end
end
@doc doc"""
    manifoldDimension(x)

returns the dimension of the [`Hyperbolic`](@ref) space $\mathbb H^n$, the
[`HnPoint`](@ref) `x`, itself embedded in $\mathbb R^{n+1}$, belongs to.
"""
manifoldDimension(x::HnPoint)::Integer = length( getValue(x) )-1
@doc doc"""
    manifoldDimension(M)

returns the dimension of the [`Hyperbolic`](@ref) space $\mathbb H^n$.
"""
manifoldDimension(M::Hyperbolic)::Integer = M.dimension
@doc doc"""
    norm(M,x,ξ)
Computes the norm of the [`HnTVector`](@ref) `ξ` in the tangent space
$T_x\mathcal M$ at [`HnPoint`](@ref) `x` of the
[`Hyperbolic`](@ref) space $\mathbb H^n$.
"""
norm(M::Hyperbolic, x::HnPoint{T}, ξ::HnTVector{T})  where {T <: AbstractFloat}= sqrt(dot(M,x,ξ,ξ))
@doc doc"""
    parallelTransport(M,x,y,ξ)

Compute the paralllel transport of the [`HnTVector`](@ref) `ξ` from
the tangent space $T_x\mathcal M$ at [`HnPoint`](@ref) `x` to
$T_y\mathcal M$ at [`HnPoint`](@ref) `y` on the
[`Hyperbolic`](@ref) space $\mathbb H^n$ along the unique [`geodesic`](@ref)
$g(\cdot;x,y)$.
The formula reads

$P_{x\to y}(\xi) = \xi - \frac{\langle \log_xy,\xi\rangle_x}
{d^2_{\mathbb H^n}(x,y)}\bigl(\log_xy + \log_yx \bigr).$
"""
function parallelTransport(M::Hyperbolic, x::HnPoint{T}, y::HnPoint{T}, ξ::HnTVector{T})  where {T <: AbstractFloat}
  ν = log(M,x,y);
  νL = norm(M,x,ν);
  if νL > 0
    ν = ν/νL;
	return HnTVector( getValue(ξ) - dot(M,x,ν,ξ)*( getValue(ν) + getValue(log(M,y,x))/νL) );
  else
    # if length of ν is 0, we have p=q and hence ξ is unchanged
    return ξ;
  end
end

typeofTVector(::Type{HnPoint{T}}) where T = HnTVector{T}
typeofMPoint(::Type{HnTVector{T}}) where T = HnPoint{T} 

@doc doc"""
    typicalDistance(M)
returns the typical distance on the [`Hyperbolic`](@ref) space `M`: $\sqrt{n}$.
"""
typicalDistance(M::Hyperbolic) = sqrt(M.dimension);
@doc doc"""
    validateMPoint(M,x)

validate, that the [`HnPoint`](@ref) `x` is a valid point on the
[`Hyperbolic`](@ref) space `M`, i.e. that the dimension of $x\in\mathbb H^n$ is
correct and that its [`MinkowskiDot`](@ref) inner product is $\langle x,x\rangle_{\mathrm{M}} = -1$.
"""
function validateMPoint(M::Hyperbolic, x::HnPoint)
    if length(getValue(x)) ≠ M.dimension+1
    throw( ErrorException(
      "The Point $x is not on the $(M.name), since the vector dimension ($(length(getValue(x)))) is not $(M.dimension+1)."
    ))
  end
  if (MinkowskiDot(getValue(x),getValue(x))+1) >= 10^(-15)
    throw( ErrorException(
      "The Point $x is not on the $(M.name) since its minkowski inner product <x,x>_MN is $(norm(getValue(x))) is not -1"
    ))
  end
  return true
end
@doc doc"""
    validateTVector(M,x,ξ)

check that the [`HnTVector`](@ref) `ξ` is a valid tangent vector in the tangent
space of the [`HnPoint`](@ref) `x` on the [`Hyperbolic`](@ref) space `M`, i.e. `x`
is a valid point on `M`, the vectors within `ξ` and `x` agree in length and the
Minkowski inner product [`MinkowskiDot`](@ref)`(x,ξ) `is zero.
"""
function validateTVector(M::Hyperbolic, x::HnPoint, ξ::HnTVector)
    if !validateMPoint(M,x)
        return false
     end
    if length(getValue(x)) != length(getValue(ξ))
        throw( ErrorException(
            "The lengths of x ($(length(getValue(x)))) and ξ ($(length(getValue(ξ)))) do not agree, so ξ can not be a tangent vector of x."
        ))
    end
    if abs( MinkowskiDot(getValue(x),getValue(ξ)) ) > 10^(-15)
        throw( ErrorException(
            "The tangent vector should be (hyperbolically) orthogonal to its base, but the (Minkowski) inner product yields $( abs( MinkowskiDot(getValue(x),getValue(ξ))) )."
        ))
    end
    return true
end
@doc doc"""
    ξ = zeroTVector(M,x)

returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`HnPoint`](@ref) $x\in\mathbb H^n$ on the [`Hyperbolic`](@ref) space `M`.
"""
zeroTVector(M::Hyperbolic, x::HnPoint{T}) where {T <: AbstractFloat} = HnTVector(  zero( getValue(x) )  );
# Display
# ---
show(io::IO, M::Hyperbolic) = print(io, "The $(M.name).")
show(io::IO, p::HnPoint) = print(io, "Hn($( getValue(p) ))")
show(io::IO, ξ::HnTVector) = print(io, "HnT($( getValue(ξ) ))")
# Helper
#
@doc doc"""
    MinkowskiDot(a,b)
computes the Minkowski inner product of two Vectors `a` and `b` of same length
`n+1`, i.e.

$\langle a,b\rangle_{\mathrm{M}} = -a_{n+1}b_{n+1} +
\displaystyle\sum_{k=1}^n a_kb_k.$
"""
MinkowskiDot(a::Vector,b::Vector) = -a[end]*b[end] + sum( a[1:end-1].*b[1:end-1] )