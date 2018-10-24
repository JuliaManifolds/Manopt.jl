#
#
#
export gradTV, gradTV2, gradIntrICTV12

@doc doc"""
   ∇u,⁠∇v = gradIntrICTV12(M,f,u,v,α,β)
Computes (sub)gradient of the intrinsic infimal convolution model using the mid point
model of second order differences, see [`costTV2`](@ref), i.e. for some $f\in\mathcal M$
on a [`Power`] manifold $\mathcal M$ this function computes the (sub) gradient of

$E_{\mathrm{IC}}^{\mathrm{int}}(u,v) =
\frac{1}{2}\sum_{i\in\mathcal G} d_{\mathcal M}(g(\frac{1}{2},v_i,w_i),f_i)
+ \alpha
\bigl(
\beta\mathrm{TV}(v) + (1-\beta)\mathrm{TV}_2(w)
\bigr),$
where both total variations refer to the intrinsic ones, [`gradTV`](@ref) and
[`gradTV2`](@ref), respectively.
"""
function gradIntrICTV12(M::mT,f::P,u::P,v::P,α,β) where {mT <: Manifold, P <: MPoint}
  c = midPoint(M,u,v,f)
  iL = log(M,c,f)
  return AdjDxGeo(M,u,v,1/2,iL) + α*β*gradTV(M,u), AdjDyGeo(M,u,v,1/2,iL) + α * (1-β) * gradTV2(M,v)
end
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
* `M`     : a [`Power`](@ref) manifold
* `x`     : a [`PowPoint`](@ref).

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
the evaluation of an [`adjointJacobiField`](@ref). See
Bačák, Bergmann, Steidl, Weinmann, 2016 for the derivation
"""
function gradTV2(M::mT where {mT <: Manifold}, xT::Tuple{P,P,P} where {P <: MPoint}, p::Number=1)
  x = xT[1];
  y = xT[2];
  z = xT[3];
  c = midPoint(M,x,z,y) # nearest mid point of x and z to y
  d = distance(M,y,c)
  innerLog = log(M,c,y)
  if p==2
      return ( AdjDxGeo(M,x,z,1/2,innerLog), log(M,y,c), AdjDyGeo(M,x,z,1/2,innerLog))
  else
    if d==0 # subdifferential containing zero
      return (zeroTVector(M,x),zeroTVector(M,y),zeroTVector(M,z))
    else
      return ( AdjDxGeo(M,x,z,1/2,innerLog/(d^(2-p))), log(M,y,c)/(d^(2-p)), AdjDyGeo(M,x,z,1/2,innerLog/(d^(2-p))) )
    end
  end
end
@doc doc"""
    gradTV2(M,x,p)
computes the (sub) gradient of $\frac{1}{p}d_2^p_{\mathcal M}(x_1,x_2,x_3)$
with respect to all $x_1,x_2,x_3$ occuring along any array dimension in the
[`PowPoint`](@ref)` x`, where `M` is the corresponding [`Power`](@ref) manifold.
"""
function gradTV2(M::Power,x::PowPoint,p::Int=1)::PowTVector
  R = CartesianIndices(M.dims)
  d = length(M.dims)
  minInd, maxInd = first(R), last(R)
  ξ = zeroTVector(M,x)
  c = costTV2(M,x,p,false)
  for i in R # iterate over all pixel
    di = 0.
    for k in 1:d # for all direction combinations (TODO)
      ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
      jF = i+ek # compute forward neighbor
      jB = i-ek # compute backward neighbor
      if all( map(<=, jF.I, maxInd.I) ) && all( map(>=, jB.I, minInd.I)) # are neighbors in range?
        if p != 1
          g = gradTV2(M.manifold,(x[jB],x[i],x[jF]),p)/c[i] # Compute TV2 on these
        else
          g = gradTV2(M.manifold,(x[jB],x[i],x[jF]),p) # Compute TV2 on these
        end
        ξ[jB] += g[1]
        ξ[i] += g[2]
        ξ[jF] += g[3]
      end
    end # directions
  end # i in R
  return ξ
end
