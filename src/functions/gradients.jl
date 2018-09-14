#
#
#

export gradTV, gradTV2

@doc doc"""
    gradTV(M,(x,y),[p=1])
computes the (sub) gradient of $\frac{1}{p}d^p_{\mathcal M}(x,y)$ with respect
to both $x$ and $y$.
"""
function gradTV(M::mT where {mT <: Manifold}, xT::Tuple{P,P} where {P <: MPoint}, p::Number=1)
  x = xT[1];
  y = xT[2];
  if p==2
      return (-log(M,x,y), -log(M,y,x))
  else
    d = distance(M,x,y);
    if d==0 # subdifferential containing zero
      return (zeroTVector(M,x),zeroTVector(M,y))
    else
      return (-log(M,x,y)/(d^(2-p)), -log(M,y,x)/(d^(2-p)))
    end
  end
end
@doc doc"""
    ξ = gradTV(M,λ,x,[p])
Compute the (sub)gradient $\partial F$ of all forward differences orrucirng,
in the power manifold array, i.e. of the function

$F(x) = \sum_{i}\sum_{j\in\mathcal N_i} d^p(x_i,x_j)$

where $i$ runs over all indices of the [`Power`] manifold `M` and $\mathcal N_i$
denotes the forward neighbors of $i$.

# Input
* `M`     : a manifold
* `x`    : a a [`PowPoint`](@ref).

# Optional
(default is given in brackets)
* `p` : (1) exponent of the distance of the TV term

# Ouput
* \xi : resulting tangent vector in $T_x\mathcal M$ representing the gradient.
"""
function gradTV(M::Power,x::PowPoint,p::Int=1)::PowTVector
  R = CartesianIndices(M.dims)
  d = length(M.dims)
  maxInd = last(R)
  ξ = zeroTVector(M,x)
  c = costTV(M,x,p,false)
  for i in R # iterate over all pixel
    di = 0.
    for k in 1:d # for all direction combinations (TODO)
      ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
      j = i+ek # compute neighbor
      if all( map(<=, j.I, maxInd.I)) # is this neighbor in range?
        if p != 1
          g = gradTV(M.manifold,(x[i],x[j]),p)/c[i] # Compute TV on these
        else
          g = gradTV(M.manifold,(x[i],x[j]),p) # Compute TV on these
        end
        ξ[i] += g[1]
        ξ[j] += g[2]
      end
    end # directions
  end # i in R
  return ξ
end

@doc doc"""
    gradTV2(M,(x,y,z),p)
computes the (sub) gradient of $\frac{1}{p}d_2^p_{\mathcal M}(x,y,z)$ with respect
    to $x$, $y$, and $z$, where $d_2$ denotes the second order absolute difference
    using the mid point model, i.e. let
    $\mathcal C = \{c | \exists g(\cdot;x,z) : c = g(\frac{1}{2};x,z)\}$ the set of
    mid points between $x$ and $z$ on the manifold $\mathcal M$. Then the
    absolute second order difference is defined as

    $ d_2(x,y,z) = \min_{c\in\mathcal C_{x,z}} d(c,y).$

    While the (sub)gradient with respect to $y$ is easy, the other two require
    the evaluation of an ['AdjointJacobiField'](@ref). See
    Bačák, Bergmann, Steidl, Weinmann, 2016 for the derivation
"""
function gradTV2(M::mT where {mT <: Manifold}, xT::Tuple{P,P,P} where {P <: MPoint}, p::Number=1)
  x = xT[1];
  y = xT[2];
  z = xT[3];
  c = midPoint(M,x,z)
  d = distance(M,y,c)
  innerLog = log(M,c,y)
  if p==2
      return ( AdjDxGeo(M,x,z,1/2,innerLog), -log(M,y,c), AdjDyGeo(M,x,z,1/2,innerLog))
  else
    if d==0 # subdifferential containing zero
      return (zeroTVector(M,x),zeroTVector(M,y),zeroTVector(M,z))
    else
      return ( AdjDxGeo(M,x,z,1/2,innerLog/(d^(2-p))), -log(M,y,c/(d^(2-p))), AdjDyGeo(M,x,z,1/2,innerLog/./(d^(2-p))) )
    end
  end
end
function gradTV2(M::Power,x::PowPoint,p::Int=1)::PowTVector
  R = CartesianIndices(M.dims)
  d = length(M.dims)
  maxInd = last(R)
  ξ = zeroTVector(M,x)
  c = costTV2(M,x,p,false)
  for i in R # iterate over all pixel
    di = 0.
    for k in 1:d # for all direction combinations (TODO)
      ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
      jp = i+ek # compute neighbor
      jm = i-ek
      if all( map(<=, j.I, maxInd.I)) && all( map(<=, j.I, maxInd.I)) # are both neighbors in range?
        if p != 1
          g = gradTV2(M.manifold,(x[jm],x[i],x[jp]),p)/c[i] # Compute TV2 on these
        else
          g = gradTV2(M.manifold,(x[jm],x[i],x[jp]),p) # Compute TV2 on these
        end
        ξ[jm] += g[1]
        ξ[i] += g[2]
        ξ[jp] += g[3]
      end
    end # directions
  end # i in R
  return ξ
end
