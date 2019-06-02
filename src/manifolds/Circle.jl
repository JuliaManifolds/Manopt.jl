#
#      S1 - The manifold of the 1-dimensional sphere represented by angles
#
# Manopt.jl, R. Bergmann, 2018-06-26
import Base: exp, log, show

export Circle, S1Point, S1TVector
export distance, dot, embed, exp, log, manifoldDimension, norm, opposite
export parallelTransport
export parallelTransport, randomMPoint, randomTVector, typeofMPoint, typeofTVector
export zeroTVector
export show, getValue
export validateMPoint, validateTVector

export symRem
# Types
# ---
@doc doc"""
    Circle <: Manifold

The one-dimensional manifold $\mathbb S^1$ represented by angles.
Note that one can also use the $n$-dimensional sphere with $n=1$ to obtain the
same manifold represented by unit vectors in $\mathbb R^2$.

# Abbreviation

`S1`

# Constructor

    Circle()

construct a circle
"""
struct Circle <: Manifold
  name::String
  dimension::Int
  abbreviation::String
  Circle() = new("1-Sphere as angles",1,"S1")
end
@doc doc"""
    S1Point <: MPoint

a point $x\in\mathbb S^1$ represented by an angle `getValue(x)`$\in[-\pi,\pi)$,
usually referred to as “cyclic“ or “phase” data.
"""
struct S1Point <: MPoint
  value::Float64
  S1Point(value::Float64) = new(value)
end
getValue(x::S1Point) = x.value

@doc doc"""
    S1TVector <: TVector

a tangent vector $\xi\in\mathbb S^1$ represented by a real valiue
`getValue(ξ)`$\in\mathbb R$.
"""
struct S1TVector <: TVector
  value::Float64
  S1TVector(value::Float64) = new(value)
end
getValue(ξ::S1TVector) = ξ.value
# Traits
# ---
#(a) S1 is a matrix manifold
@traitimpl IsMatrixM{Circle}
@traitimpl IsMatrixP{S1Point}
@traitimpl IsMatrixTV{S1TVector}

# Functions
# ---

@doc doc"""
    distance(M,x,y)

return the distance two cyclic data items, which is given by
$\lvert (x-y)_{2\pi} \rvert $,
where $(\cdot)_{2\pi}$ is the symmetric remainder modulo $2\pi$,
see [`symRem`](@ref).
"""
distance(M::Circle, x::S1Point,y::S1Point) = abs( symRem(getValue(y) - getValue(x)) )
@doc doc"""
    dot(M,x,ξ,ν)

compute the inner product of two [`S1TVector`](@ref)s in the tangent space $T_x\mathbb S^1$
of the [`S1Point`](@ref) `x`. Since the values are angles, we
obtain $\langle \xi,\nu\rangle_x = \xi\nu$.
"""
dot(M::Circle, x::S1Point, ξ::S1TVector, ν::S1TVector) = getValue(ξ)*getValue(ν)

@doc doc"""
    embed(M,x)

embed the [`Circle`](@ref)` `[`Manifold`](@ref), i.e. turn the [`S1Point`](@ref)
into an [`SnPoint`](@ref) on the manifold [`Sphere`](@ref)`(1)` embedded in
$\mathbb R^2$.
"""
embed(M::Circle,x::S1Point) = SnPoint([cos(getValue(x)),sin(getValue(x))])

@doc doc"""
    exp(M,x,ξ,[t=1.0])

compute the exponential map on the [`Circle`](@ref) $\mathbb S^1$ with
respect to the [`S1Point`](@ref) `x` and the [`S1TVector`](@ref) `ξ`, which can
be shortened with `t` to `tξ`. The formula reads
```math
y = (x+\xi)_{2\pi}
```
where $(\cdot)_{2\pi}$ is the symmetric remainder modulo $2\pi$,
see [`symRem`](@ref).
"""
exp(M::Circle, x::S1Point,ξ::S1TVector,t::Float64=1.0) = S1Point( symRem(getValue(x) + t*getValue(ξ)) )

@doc doc"""
    log(M,x,y)

compute the logarithmic map on the [`Circle`](@ref) $\mathbb S^1$,
i.e., the [`S1TVector`](@ref) `ξ` whose corresponding
[`geodesic`](@ref) starting from [`S1Point`](@ref) `x` reaches the
[`S1Point`](@ref)` y` after time 1. The formula reads

```math
\xi = (y-x)_{2\pi},
```

where $(\cdot)_{2\pi}$ is the symmetric remainder modulo $2\pi$,
see [`symRem`](@ref).
"""
log(M::Circle, x::S1Point,y::S1Point)::S1TVector = S1TVector(symRem( getValue(y) - getValue(x) ))

"""
    manifoldDimension(x)

return the dimension of the manifold the [`S1Point`](@ref) `x` belongs to, i.e.
of the [`Circle`](@ref), which is 1.
"""
manifoldDimension(x::S1Point) = 1

"""
    manifoldDimension(M)

return the dimension of the [`Circle`](@ref) manifold, i.e., 1.
"""
manifoldDimension(M::Circle) = 1

@doc doc"""
    norm(M,x,ξ)

compute the norm of the [`S1TVector`](@ref) `ξ` in the tangent space
$T_x\mathcal M$ at [`S1Point`](@ref) `x` of the
[`Circle`](@ref) $\mathbb S^1$, which is just its absolute value $\lvert\xi\rvert$.
"""
norm(M::Circle, x::S1Point, ξ::S1TVector)::Float64 = abs( getValue(ξ) )

@doc doc"""
    opposite(M,x)

return the antipodal [`S1Point`](@ref) of `x` on the [`Circle`](@ref) `M`,
i.e. $y = (x+\pi)_{2\pi}$.
"""
opposite(M::Circle, x::S1Point) = S1Point( symRem(getValue(x)+π) )
@doc doc"""
    parallelTransport(M,x,y,ξ)

compute the parallel transport of the [`S1TVector`](@ref) `ξ` from the tangent
space $T_x\mathbb S^1$ at the [`S1Point`](@ref) `x` to $T_y\mathbb S^1$
at the [`S1Point`](@ref)` y`.
Since the [`Sphere`](@ref) `M` is represented in angles this is the identity.
"""
parallelTransport(M::Circle, x::S1Point, y::S1Point, ξ::S1TVector) = ξ
@doc doc"""
    randomMPoint(M,:Uniform)

return a random [`S1Point`](@ref) on the [`Circle`](@ref) $\mathbb S^1$ by
picking a random element from $[-\pi,\pi)$ uniformly.
"""
randomMPoint(M::Circle, ::Val{:Uniform}) = S1Point((rand()-0.5)*2*π)
randomMPoint(M::Circle) = randomMPoint(M,Val(:Uniform)) # introduce different default

@doc doc"""
    randomTVector(M,x [,Gaussian,σ=1.0])

returns a random tangent vector from the tangent space of the [`S1Point`](@ref)
 `x` on the [`Circle`](@ref) $\mathbb S^1$ by using a normal distribution with
mean 0 and standard deviation 1.
"""
randomTVector(M::Circle, x::S1Point, ::Val{:Gaussian}, σ::Real=1.0) = S1TVector(σ*randn())

@doc doc"""
    (Ξ,κ) = tangentONB(M,x,ξ)

compute an ONB within the tangent space $T_x\mathcal M$ such that $\xi$ is the
first vector and compute the eigenvalues of the curvature tensor
$R(\Xi,\dot g)\dot g$, where $g=g_{x,\xi}$ is the geodesic with $g(0)=x$,
$\dot g(0) = \xi$, i.e. $\kappa_1$ corresponding to $\Xi_1=\xi$ is zero.

# See also
 [`jacobiField`](@ref), [`adjointJacobiField`](@ref).
"""
tangentONB(M::Circle,x::S1Point,ξ::S1TVector) = [S1TVector(sign(getValue(ξ))==0 ? 1.0 : sign(getValue(ξ)))], [0.]

@doc doc"""
    (Ξ,κ) = tangentONB(M,x,y)

compute an ONB within the tangent space $T_x\mathcal M$ such that $\xi=\log_xy$ is the
first vector and compute the eigenvalues of the curvature tensor
$R(\Xi,\dot g)\dot g$, where $g=g_{x,\xi}$ is the geodesic with $g(0)=x$,
$\dot g(0) = \xi$, i.e. $\kappa_1$ corresponding to $\Xi_1=\xi$ is zero.

# See also
 [`jacobiField`](@ref), [`adjointJacobiField`](@ref).
"""
tangentONB(M::Circle, x::S1Point, y::S1Point) = tangentONB(M,x,log(M,x,y))

typeofTVector(::Type{S1Point}) = S1TVector
typeofMPoint(::Type{S1TVector}) = S1Point 

"""
    typicalDistance(M)

returns the typical distance on the [`Circle`](@ref) `M`: π.
"""
typicalDistance(M::Circle) = π/2;

@doc doc"""
    ξ = zeroTVector(M,x)

returns a zero vector in the tangent space $T_x\mathcal M$ of the
[`S1Point`](@ref) $x\in\mathbb S^1$ on the [`Circle`](@ref)` S1`.
"""
zeroTVector(M::Circle, x::S1Point) = S1TVector(  zero( getValue(x) )  );
# Display
# ---
show(io::IO, M::Circle) = print(io, "The manifold S1 consisting of angles");
show(io::IO, x::S1Point) = print(io, "S1($( getValue(x) ))");
show(io::IO, ξ::S1TVector) = print(io, "S1T($( getValue(ξ) ))");
# little Helpers
# ---
@doc doc"""
    symRem(x,[T=π])

symmetric remainder of `x` with respect to the interall 2*`T`, i.e.
`(x+T)%2T`, where the default for `T` is $\pi$
"""
function symRem(x::Float64, T::Float64=Float64(π))::Float64
  return rem(x, 2*T,RoundNearest)
end
@doc doc"""
    validateMPoint(M,x)

validate, that a [`S1Point`](@ref) `x` has a valid value for a point on the
[`Circle`](@ref) `M`$=\mathbb S^1$, i.e. is within $[-\pi,\pi)$. 
"""
function validateMPoint(M::Circle, x::S1Point)
    if (getValue(x) < - π) || (getValue(x) >= π) # out of range
        throw( ErrorException(
            "The Point $x is out of range for the Circle represented by anfgles in radians [-π,π),"
        ))
    end
    return true
end
@doc doc"""
    validateTVector(M,x,ξ)

validate, that the [`S1TVector`](@ref) `ξ` is a valid tangent vector in the
tangent space of the [`S1Point`](@ref) `x` ont the [`Circle`](@ref) `M`$=\mathbb S^1$,
though this is always the case since all real values are valid.
""" 
function validateTVector(M::Circle,x::S1Point,ξ::S1TVector)
    return true
end