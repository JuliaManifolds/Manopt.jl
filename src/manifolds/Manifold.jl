#
#      Manifold -- a manifold defined via its data types:
#  * A point on the manifold, MPoint
#  * A point in an tangential space TVector
#
# Manopt.jl, R. Bergmann, 2019
import LinearAlgebra: norm, dot
import Base: exp, log, +, -, *, /, ==, show, copy
# introcude new types
export Manifold, MPoint, TVector
# introduce new functions
export geodesic, midPoint, reflection, jacobiField, adjointJacobiField
export +, -, *, /, ==, show, copy
@doc doc"""
An abstract manifold $\mathcal M$ to keep global information on a specific manifold
"""
abstract type Manifold end

@doc doc"""
An abstract point $x$ on a [`Manifold`](@ref) $\mathcal M$.
"""
abstract type MPoint end

@doc doc"""
A tangent vector $\xi \in T_x\mathcal M$ at a [`MPoint`](@ref) point $x$ on a
[`Manifold`](@ref) $\mathcal M$.
"""
abstract type TVector end

# scale tangential vectors
*(ξ::T,s::N) where {T <: TVector, N <: Number} = T(s* getValue(ξ) )
*(s::N, ξ::T) where {T <: TVector, N <: Number} = T(s* getValue(ξ) )
# /
/(ξ::T,s::N) where {T <: TVector, N <: Number} = T( getValue(ξ) ./ s)
/(s::N, ξ::T) where {T <: TVector, N <: Number} = T(s ./ getValue(ξ) )
# + - of TVectors
+(ξ::T,ν::T) where {T <: TVector} = T( getValue(ξ) + getValue(ν) )
-(ξ::T,ν::T) where {T <: TVector} = T( getValue(ξ) - getValue(ν) )
# unary operators
-(ξ::T) where {T <: TVector} = T(- getValue(ξ))
+(ξ::T) where {T <: TVector} = T(getValue(ξ))

# compare Points & vectors
==(x::T, y::T) where {T <: MPoint} = all(getValue(x) == getValue(y) )
==(ξ::T,ν::T) where {T <: TVector} = (  all( getValue(ξ) == getValue(ν) )  )
#
#
# General functions available on manifolds based on exp/log/dist
#
#
@doc doc"""
    addNoise(M,x)

add noise to a [`MPoint`](@ref) `x` on the [`Manifold`](@ref) `M` by using the
[`randomTVector`](@ref) method and doing an exponential step.
Optional parameters, like the type of noise and parameters for the noise
may be given and are just passed on-
"""
addNoise(M::mT, x::P, options...) where {mT <: Manifold, P <: MPoint} = exp(
    M,x,randomTVector(M,x,options...)
)

@doc doc"""
    ζ = adjointJacobiField(M,x,y,t,η,w)

compute the AdjointJacobiField $J$ along the geodesic $g_{x,y}$ on the manifold
$\mathcal M$ with initial conditions (depending on the application)
$\eta\in T_{g(t;x,y)\mathcal M}$ and weights $\beta$. The result is a vector
$\zeta \in T_x\mathcal M$. The main difference to [`jacobiField`](@ref) is the,
that the input $\eta$ and the output $\zeta$ switched tangent spaces.

For detais see [`jacobiField`](@ref)
"""
function adjointJacobiField(M::mT,x::P,y::P,t::Number,η::T,β::Function=βDgx) where {mT<:Manifold, P<:MPoint, T<:TVector}
    z = geodesic(M,x,y,t); # Point the TzM of the resulting vector lies in
    Ξ,κ = tangentONB(M,x,y) # ONB at x
    Θ = parallelTransport.(Ref(M),Ref(x),Ref(z),Ξ) # Frame at z
    # Decompose wrt. Ξ, multiply with the weights from w and recompose with Θ.
    ξ = sum( ( dot.(Ref(M),Ref(z),Ref(η),Θ) ).* ( β.(κ,Ref(t),distance(M,x,y)) ).*Ξ )
end

copy(x::P) where {P <: MPoint} = P(copy(getValue(x)))
copy(ξ::T) where {T <: TVector} = T(copy(getValue(ξ)))

"""
    midPoint(M,x,y,z)

compute the mid point between x and y. If there is more than one mid point
of (not neccessarily miniizing) geodesics (i.e. on the sphere), the one nearest
to z.
"""
function midPoint(M::mT,x::T,y::T,z::T)::T where {mT <: Manifold, T <: MPoint}
    # since this is the fallback, it just uses the non-nearest one
    return midPoint(M,x,y)
end

"""
    midPoint(M,x,y)

compute the (geodesic) mid point of the two [`MPoint`](@ref)s `x` and `y` on the
[`Manifold`](@ref) `M`. If the geodesic is not unique, either a deterministic
choice is returned or an error is raised. For the deteministic choixe, see
[`midPoint(M,x,y,z)`](@ref), the mid point closest to a third [`MPoint`](@ref)
`z`.
"""
function midPoint(M::mT,x::T, y::T)::T where {mT <: Manifold, T <: MPoint}
  return exp(M,x,0.5*log(M,x,y))
end

"""
    geodesic(M,x,y)

return a function to evaluate the geodesic connecting the two [`MPoint`](@ref)s
`x` and `y` on the [`Manifold`](@ref) `M`.
"""
geodesic(M::mT, x::T,y::T) where {mT <: Manifold, T <: MPoint} = t -> exp(M,x,t*log(M,x,y))

"""
    geodesic(M,x,y,n)

return vector containing the equispaced `n` sample-values along the geodesic
connecting the two [`MPoint`](@ref)s `x` and `y` on the [`Manifold`](@ref) `M`."""
geodesic(M::mT, x::T,y::T,n::Integer) where {mT <: Manifold, T <: MPoint} = geodesic(M,x,y,[range(0.,1.,length=n)...])

"""
    geodesic(M,x,y,t)

return the point along the geodesic from [`MPoint`](@ref) `x` to `y` given by
at value `t` (in `[0,1]`) on the [`Manifold`](@ref) `M`
"""
geodesic(M::mT,x::T,y::T,t::N) where {mT <: Manifold, T <: MPoint, N <: Number} = geodesic(M,x,y)(t)

"""
    geodesic(M,x,y,T)

return vector containing the [`MPoint`](@ref) along the geodesic from
[`MPoint`](@ref) `x` to `y` on the [`Manifold`](@ref) `M` specified by the
points from the vector `T` (of numbers between 0 and 1).
"""
geodesic(M::mT, x::P,y::P,T::Vector{S}) where {mT <: Manifold, P <: MPoint, S <: Number} = geodesic(M,x,y).(T)

@doc doc"""
    getValue(x)

get the value representing the [`MPoint`](@ref) `x`.
This function defaults to returning `x.value`; if your representation is 
different, you should implement this function for your type
"""
function getValue(x::P) where {P <: MPoint}
    try
        return x.value
    catch
        sig1 = string( typeof(x) )
        throw( DomainError("getValue not defined/implemented for a $sig1.") );
    end
end
@doc doc"""
    getValue(ξ)

get the value representing the [`TVector`](@ref) `ξ`.
This function defaults to returning `ξ.value`; if your representation is 
different, you should implement this function for your type
"""
function getValue(ξ::T) where {T <: TVector}
    try
        return ξ.value
    catch
        sig1 = string( typeof(ξ) )
        throw( DomainError("getValue – not defined/implemented for tangent vector $sig1.") );
    end
end

@doc doc"""
    ζ = jacobiField(M,x,y,t,η,β)

compute the jacobiField $J$ along the geodesic $g_{x,y}$ on the
[`Manifold`](@ref) `M` $\mathcal M$ with initial conditions (depending on the
application) $\eta\in T_x\mathcal M$ and weights $\beta$. The result is a
[`TVector`](@ref) in $\zeta \in T_{g(t;x,y)}\mathcal M$.

# See also
 [`adjointJacobiField`](@ref)
"""
function jacobiField(M::mT,x::P,y::P,t::Number,η::T,β::Function=βDgx) where {mT<:Manifold, P<:MPoint, T<:TVector}
    z = geodesic(M,x,y,t); # Point the TzM of the resulting vector lies in
    Ξ,κ = tangentONB(M,x,y) # ONB at x
    Θ = parallelTransport.(Ref(M),Ref(x),Ref(z),Ξ) # Frame at z
    # Decompose wrt. Ξ, multiply with the weights from w and recompose with Θ.
    ξ = sum( ( dot.(Ref(M),Ref(x),Ref(η),Ξ) ).* ( β.(κ,Ref(t),Ref(distance(M,x,y))) ).*Θ )
end

@doc doc"""
    manifoldDimension(x)

return the dimension of the manifold `M` the point `x` belongs to.
"""
function manifoldDimension(x::P)::Integer where {P<:MPoint}
  sig1 = string( typeof(x) )
  throw( DomainError("manifoldDimension not defined/implemented for a $sig1." ) )
end
@doc doc"""
    manifoldDimension(M)

returns the dimension of the manifold `M`.
"""
function manifoldDimension(M::mT)::Integer where {mT<:Manifold}
  sig1 = string( typeof(M) )
  throw( DomainError("manifoldDimension not defined/implemented on $sig1." ) )
end

@doc doc"""
    norm(M,x,ξ)

computes the length of a [`TVector`](@ref) `ξ` in the tangent space of the
[`MPoint`](@ref) `x` on the [`Manifold`](@ref) `M` induced by the inner product.
"""
norm(M::mT,x::P,ξ::T) where {mT<:Manifold,P<:MPoint,T<:TVector} = sqrt(dot(M,x,ξ,ξ))

randomMPoint(M::mT,options...) where {mT <: Manifold} = randomMPoint(M, :Gaussian, options...)
@doc doc"""
    randomMPoint(M,x [,:Gaussian,options...])

generate a random [`MPoint`](@ref) on the [`Manifold`](@ref) `M` by falling
back to the default `:Gaussian` noise with the default standard deviation
on the specific manifold.
"""
randomMPoint(M::mT,s::Symbol,options...) where {mT <: Manifold} = randomMPoint(M, Val(s), options...)

randomTVector(M::mT, x::P, options...) where {mT <: Manifold, P <: MPoint} = randomTVector(M,x,:Gaussian,options...)
@doc doc"""
    randomTVector(M,x [,:Gaussian,options...])

generate a random tangent vector at [`MPoint`](@ref) `x`
on the [`Manifold`](@ref) `M` using `:Gaussian` noise where options usually
contain the standard deviation σ on the specific manifold.
"""
randomTVector(M::mT, x::P,s::Symbol,options...) where {mT <: Manifold, P <: MPoint} = randomTVector(M,x,Val(s),options...)

@doc doc"""
    y = reflection(M,p,x)

reflect the `MPoint x` at `MPoint p`, i.e. compute
$y = R_p(x) = \exp_p(-\log_px)$. On Euclidean space this results in the point
reflection $R_p(x) = p - (x-p) = 2p-x$.

# Arguments
* `M` – a [`Manifold`](@ref) $\mathcal M$
* `p` – an [`MPoint`](@ref) $p\in\mathcal M$ to relfect at
* `x` – an [`MPoint`](@ref) $x\in\mathcal M$ that is reflected

# Output
* `y` – the resulting reflection.
"""
reflection(M::mT, p::P, x::P) where {mT <: Manifold, P<: MPoint} = exp(M,p,-log(M,p,x))

"""
    typeofMPoint(ξ)

return the [`MPoint`](@ref) belonging to the [`TVector`](@ref) type of `ξ`.
"""
typeofMPoint(ξ::T) where {T <: TVector} = typeofMPoint(typeof(ξ))
"""
    typeofTVector(x)

return the [`TVector`](@ref) belonging to the [`MPoint`](@ref) type of `x`.
"""
typeofTVector(x::P) where {P <: MPoint} = typeofTVector(typeof(x))
