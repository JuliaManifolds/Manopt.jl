#
# Manopt.jl – Proximal maps
#
# This file provides several proximal maps on manifolds or on small
# product manifolds, like M^2
#
# ---
# Manopt.jl - R. Bergmann – 2017-07-06

export proxDistance, proxTV, proxDistanceSquared, proxTVSquared

@doc doc"""
    y = proxDistance(M,λ,f,x,[p]) -
compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ with
parameter λ of $\varphi(x) = d_{\mathcal M}^p(f,x)$.

# Input
* `M` a manifold $\mathcal M$
* `λ` the prox parameter
* `f` an `MPoint` $f\in\mathcal M$
* `x` the argument of the proximal map

# Optional argument
* `p` : (2) exponent of the distance.

# Ouput
* `y` : the proximal map of $\varphi$
"""
proxDistance{mT <: Manifold, T <: MPoint}(M::mT,λ::Number,f::T,x::T) = proxDistance(M,λ,f,x,2)
function proxDistance{mT <: Manifold, T <: MPoint}(M::mT,λ::Number,f::T,x::T,p::Int)
  d = distance(M,p,q)
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
  return exp(M,x,f,t);
end
@doc doc"""
    (y1,y2) = proxTV(M,λ,(x1,x2),[p])
Compute the proximal map $\operatorname{prox}_{\lambda\varphi}$ of
$\varphi(x,y) = d_{\mathcal M}^p(x,y)$ with
parameter `λ`.

# Input
* `M`     : a manifold
* `λ`     : a real value, parameter of the proximal map
* `(x,y)` : a tuple of `MPoints`, `x,y`

# Optional
(default is given in brackets)
* `p` : (1) exponent of the distance of the TV term

# Ouput
* (y1,y2) : resulting tuple of `MPoints` of the $\operatorname{prox}_{\lambda\varphi}($ `(x1,x2)` $)$
"""
function proxTV{mT <: Manifold, T <: MPoint}(M::mT,λ::Number, pointTuple::Tuple{T,T},p::Int=1)::Tuple{T,T}
  x1 = pointTuple[1];
  x2 = pointTuple[2];
  d = distance(M,x1,x2);
  if p==1
    t = min(0.5, λ/d);
  elseif p==2
    t = λ/(1+2*λ);
  else
    throw(ErrorException(
      "Proximal Map of TV(M,x1,x2) not implemented for $(p) (requires p=1 or 2)"
    ))
  end
  return (  exp(M, x1, log(M,x1,x2), t), exp(M, x2, log(M, x2, x1), t)  );
end
