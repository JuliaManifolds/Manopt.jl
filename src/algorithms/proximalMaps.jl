#
# Manopt.jl – Proximal maps
#
# This file provides several proximal maps on manifolds or on small
# product manifolds, like M^2
#
# ---
# Manopt.jl - R. Bergmann – 2017-07-06

export proxDistance, proxTV, proxDistanceSquared, proxTVSquared

"""
    proxDistance(M,λ,f,p) -
  compute the proximal map with parameter λ of `distance(f,p)` for some
  fixed `MPoint` f on the `Manifold` M.
"""
function proxDistance{mT <: Manifold, T <: MPoint}(M::mT,λ::Number,f::T,p::T)::T
  exp(M,p, min(λ, distance(M,f,p))*log(M,p,f))
end

"""
    proxDistanceSquared(M,λ,f,p)
  computes the proximal map with prameter `λ` of distance^2(f,p) for some fixed
  `MPoint` f on the `Manifold` M.
"""
function proxDistanceSquared{mT <: Manifold, T <: MPoint}(M::mT,λ::Number,f::T,p::T)::T
  return exp(M,p, λ/(1+λ)*log(M,p,f) )
end
"""
    proxTuple = proxTV(M,λ,(p,q))
  Compute the proximal map prox_f(p,q) for f(p,q) = dist(p,q) with
  parameter `λ`
  # Arguments
  * `M` – a manifold
  * `(p,q)` : a tuple of size 2 containing two MPoints p and q
  * `λ` – a real value, parameter of the proximal map
  # Returns
  * `(pp,qp)` – resulting two-MPoint-Tuple of the proximal map
"""
function proxTV{mT <: Manifold, T <: MPoint}(M::mT,λ::Number, pointTuple::Tuple{T,T})::Tuple{T,T}
  step = min(0.5, λ/distance(M,pointTuple[1],pointTuple[2]))
  return (  exp(M,pointTuple[1], step*log(M,pointTuple[1],pointTuple[2])),
            exp(M,pointTuple[2], step*log(M,pointTuple[2],pointTuple[1])) )
end
"""
    proxTuple = proxTVSquared(M,λ,(p,q))
  Compute the proximal map prox_f(p,q) for f(p,q) = dist(p,q)^2 with
  parameter `λ`.
  # Arguments
  *
  * `(p,q)` : a tuple of size 2 containing two MPoints x and y
  * `λ` : a real value, parameter of the proximal map
  # OUTPUT
  * `(pr,qr)` : resulting two-MPoint-Tuple of the proximal map
"""
function proxTVSquared{mT <: Manifold, T <: MPoint}(M::mT,λ::Number, pointTuple::Tuple{T,T})::Tuple{T,T}
  step = λ/(1+2*λ)*distance(M, pointTuple[1],pointTuple[2])
  return (  exp(M, pointTuple[1], step*log(M, pointTuple[1],pointTuple[2])),
            exp(M, pointTuple[2], step*log(M, pointTuple[2],pointTuple[1])) )
end
