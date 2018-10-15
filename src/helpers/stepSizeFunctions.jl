#
# Collects a few step Size Function
#
export constantStepSize, decreasingStepSize, normedStepSize
@doc doc"""
    constantStepSize(c)
returns a function depenting on the iteration `i`, a manifold point `x` (the
current iterate) and a tangent vector (a subgradient) `ξ` to return a step size,
here a constant step size, i.e. `(i,x,ξ) -> c`
"""
constantStepSize(c::Number) = (i,x,ξ) -> c
@doc doc"""
    decreasingStepSize(c[,k=1])
returns a function depenting on the iteration `i`, a manifold point `x` (the
current iterate) and a tangent vector (a subgradient) `ξ` to return a step size,
here a linearly decreasing step size, i.e. `(i,x,ξ) -> c(i^k)`
"""
decreasingStepSize(c::Number,k::Number=1) = (i,x,ξ) -> c/(i^k)
@doc doc"""
    normedStepSize(M,c)
returns a function depenting on the iteration `i`, a manifold point `x` (the
current iterate) and a tangent vector (a subgradient) `ξ` to return a step size,
here a normed step size, i.e. `(i,x,ξ) -> c/norm(M,xξ)`, i.e. normed with respect
to the Riemannian metric on the manifold `M` or more precisely in the tangent
space at `x`.
"""
normedStepSize(M::mT,c::Number) where {mT <: Manifold} = (i,x,ξ) -> c/norm(M,x,ξ)
