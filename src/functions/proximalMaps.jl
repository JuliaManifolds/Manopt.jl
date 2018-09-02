#
# Manopt.jl – Proximal maps
#
# This file provides several proximal maps on manifolds or on small
# product manifolds, like M^2
#
# ---
# Manopt.jl - R. Bergmann – 2017-07-06

export proxDistance, proxTV, proxTV2

@doc doc"""
    y = proxDistance(M,λ,f,x,[p]) -
compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ with
parameter λ of $\varphi(x) = \frac{1}{p}d_{\mathcal M}^p(f,x)$.

# Input
* `M` a manifold $\mathcal M$
* `λ` the prox parameter
* `f` an [`MPoint`](@ref) $f\in\mathcal M$ (the data)
* `x` the argument of the proximal map

# Optional argument
* `p` : (2) exponent of the distance.

# Ouput
* `y` : the proximal map of $\varphi$
"""
function proxDistance(M::mT,λ::Number,f::T,x::T,p::Int=2) where {mT <: Manifold, T <: MPoint}
  d = distance(M,f,x)
  if p==2
    t =  λ/(1+λ);
  elseif p==1
    if λ < d
      t = λ/d;
    else
      t = 1;
    end
  else
      throw(ErrorException(
        "Proximal Map of distance(M,f,x) not implemented for $(p) (requires p=1 or 2)"
      ))
  end
  return exp(M,x,log(M,x,f),t);
end
@doc doc"""
    (y1,y2) = proxTV(M,λ,(x1,x2),[p])
Compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ of
$\varphi(x,y) = d_{\mathcal M}^p(x,y)$ with
parameter `λ`.

# Input
* `M`     : a manifold
* `λ`     : a real value, parameter of the proximal map
* `(x1,x2)` : a tuple of two [`MPoint`](@ref)s,

# Optional
(default is given in brackets)
* `p` : (1) exponent of the distance of the TV term

# Ouput
* (y1,y2) : resulting tuple of [`MPoint`](@ref) of the $\operatorname{prox}_{\lambda\varphi}($ `(x1,x2)` $)$
"""
function proxTV(M::mT,λ::Number, pointTuple::Tuple{P,P},p::Int=1)::Tuple{P,P} where {mT <: Manifold, P <: MPoint}
  x1 = pointTuple[1];
  x2 = pointTuple[2];
  d = distance(M,x1,x2);
  if p==1
    t = min(0.5, λ/d);
  elseif p==2
    t = λ/(1+2*λ);
  else
    throw(ErrorException(
      "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)"
    ))
  end
  return (  exp(M, x1, log(M, x1, x2), t), exp(M, x2, log(M, x2, x1), t)  );
end
@doc doc"""
    ξ = proxTV(M,λ,x,[p])
Compute the proximal maps $\operatorname{prox}_{\lambda\varphi}$ of
all forward differences orrucirng in the power manifold array, i.e.
$\varphi(xi,xj) = d_{\mathcal M}^p(xi,xj)$ with `xi` and `xj` are array
elemets of `x` and `j = i+e_k`, where `e_k` is the $k$th unitvector.
The parameter `λ` is the prox parameter.

# Input
* `M`     : a manifold
* `λ`     : a real value, parameter of the proximal map
* `x`    : a a [`PowPoint`](@ref).

# Optional
(default is given in brackets)
* `p` : (1) exponent of the distance of the TV term

# Ouput
* y : resulting of [`PowPoint`](@ref) with all mentioned proximal
  points evaluated (in a cylic order).
"""
function proxTV(M::Power, λ::Number, x::PowPoint,p::Int=1)::PowPoint
  R = CartesianIndices(M.dims)
  d = length(M.dims)
  maxInd = last(R)
  y = copy(x)
  for k in 1:d # for all directions
    ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
    for l in 0:1
      for i in R # iterate over all pixel
        if (i[k] % 2) == l
          j = i+ek # compute neighbor
          if all( map(<=, j.I, maxInd.I)) # is this neighbor in range?
            (y[i],y[j]) = proxTV( M.manifold,λ,(y[i],y[j]),p) # Compute TV on these
          end
        end
      end # i in R
    end # even odd
  end # directions
  return y
end
@doc doc"""
    (y1,y2) = proxTV2(M,λ,(x1,x2),[p])
Compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ of
$\varphi(x1,x2,x3) = d_{\mathcal M}^p(c(x1,x3),x2)$ with
parameter `λ`>0, where $c(x,z)$ denotes the mid point of a shortest
geodesic from x1 to x3.

# Input
* `M`          : a manifold
* `λ`          : a real value, parameter of the proximal map
* `(x1,x2,x3)` : a tuple of three [`MPoint`](@ref)s

# Optional
(default is given in brackets)
* `p` : (1) exponent of the distance of the TV term

# Ouput
* (y1,y2,y3) : resulting tuple of [`MPoint`](@ref)s of the proximal map
"""
function proxTV2(M::mT,λ,pointTuple::Tuple{P,P,P},p::Int=1)::Tuple{P,P,P} where {mT <: Manifold, P <: MPoint}
  throw(ErrorException(
    "Proximal Map of TV2(M,x1,x2,x3) not (yet) implemented for the manifold $(M)."
  ))
end
function proxTV2(M::Circle,λ,pointTuple::Tuple{S1Point,S1Point,S1Point},p::Int=1)::Tuple{S1Point,S1Point,S1Point}
  w = [1., -2. ,1. ]
  x = getValue.(pointTuple)
  if p==1 # Theorem 3.5 in Bergmann, Laus, Steidl, Weinmann, 2014.
    m = min(   λ, abs(  symRem( sum( x .* w  ) )  )/(dot(w,w))   )
    s = sign.( symRem(sum(x .* w)) )
    return Tuple( S1Point.( symRem.( x  .-  m .* s .* w ) ) )
  elseif p==2 # Theorem 3.6 ibd.
    t = λ * symRem( sum( x .* w ) ) ./ (1 + λ*dot(w,w) )
    return Tuple(  S1Point.( symRem.( x - t.*w ) )  )
  else
    throw(ErrorException(
      "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)"
    ))
  end
end
@doc doc"""
    ξ = proxTV2(M,λ,x,[p])
Compute the proximal maps $\operatorname{prox}_{\lambda\varphi}$ of
all centered second order differences orrucirng in the power manifold array, i.e.
$\varphi(xk,xi,xj) = d_{\mathcal M}^p(xi,xj)$ with `xk` and `xj` are array
elemets of `x` and `j = i+e_k`, `k = i+e_k` where `e_k` is the $k$th unitvector.
The parameter `λ` is the prox parameter.

# Input
* `M`     : a manifold
* `λ`     : a real value, parameter of the proximal map
* `x`     : a [`PowPoint`](@ref).

# Optional
(default is given in brackets)
* `p` : (1) exponent of the distance of the TV term

# Ouput
* y : resulting of [`PowPoint`](@ref) with all mentioned proximal points evaluated (in a cylic order).
"""
function proxTV2(M::Power, λ::Number, x::PowPoint,p::Int=1)::PowPoint
  R = CartesianIndices(M.dims)
  d = length(M.dims)
  minInd, maxInd = first(R), last(R)
  y = copy(x)
  for k in 1:d # for all directions
    ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
    for l in 0:2
      for i in R # iterate over all pixel
        if (i[k] % 3) == l
          jF = i+ek # compute forward neighbor
          jB = i-ek # compute backward neighbor
          if all( map(<=, jF.I, maxInd.I) ) && all( map(>=, jB.I, minInd.I)) # are neighbors in range?
            (y[jB], y[i], y[jF]) = proxTV2( M.manifold, λ, (y[jB], y[i], y[jF]),p) # Compute TV on these
          end
        end # if mod 3
      end # i in R
    end # for mod 3
  end # directions
  return y
end
