#
#
#

export gradTV, gradTV2

@doc doc"""
    gradTV(M,(x,y),[p=1])
computes the (sub) gradient of $\frac{1}{p}d^p_{\mathcal M}(x,y)$ with respect
to both $x$ and $y$.
"""
function gradTV(M::mT where {mT <: Manifold}, x::Tuple{P,P} where {P <: MPoint}, p::Number)
  x1 = pointTuple[1];
  x2 = pointTuple[2];
  if p==2
      return (-log(M,x1,x2), -log(M,x2,x1))
  else
    d = distance(M,x1,x2);
    if d==0 # subdifferential containing zero
      return (zeroTVector(M,x),zeroTVector(M,y))
    else
      return (-log(M,x1,x2)./(d^(2-p)), -log(M,x2,x1)./(d^(2-p)))
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
function gradTV(M::Power,x::PowPoint,p::Int=1)::PowPoint
  R = CartesianIndices(M.dims)
  d = length(M.dims)
  maxInd = last(R)
  ξ = zeroTVector(M,x)
  c = costTV(M,x,p,false)
  for i in R # iterate over all pixel
    ξi = zeroTVector(M.manifold,x[i])
    di = 0.
    for k in 1:d # for all direction combinations (TODO)
      ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
      j = i+ek # compute neighbor
      if all( map(<=, j.I, maxInd.I)) # is this neighbor in range?
        if p != 1
          ξi += gradTV(M.manifold,(x[i],x[j]),p)/c[i] # Compute TV on these
        else
          ξi += gradTV(M.manifold,(x[i],x[j]),p) # Compute TV on these
        end
      end
      j = i-ek # compute neighbor
      if all( map(>=, j.I, maxInd.I)) # is this neighbor in range?
        if p != 1
          ξi += gradTV(M.manifold,(x[i],x[j]),p)/c[j] # Compute TV on these
        else
          ξi += gradTV(M.manifold,(x[i],x[j]),p) # Compute TV on these
        end
      end
    end # directions
      ξ[i] = ξi
  end # i in R
  return ξ
end

@doc doc"""
    gradTV2()
    dummy docu, to be implemented soon
"""
function gradTV2()
end
