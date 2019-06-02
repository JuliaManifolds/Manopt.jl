#
#
#
export gradTV, gradTV2, gradIntrICTV12, forwardLogs
export gradDistance
@doc doc"""
    gradDistance(M,y,x[, p=2])

compute the (sub)gradient of the distance (squared) 

```math
f(x) = \frac{1}{2} d^p_{\mathcal M}(x,y)
```

to a fixed [`MPoint`](@ref)` y` on the [`Manifold`](@ref) `M` and `p` is an
integer. The gradient reads

```math
  \nabla f(x) = -d_{\mathcal M}^{p-2}(x,y)\log_xy
```

for $p\neq 1$ or $x\neq  y$. Note that for the remaining case $p=1$,
$x=y$ the function is not differentiable. This function returns then the
[`zeroTVector`](@ref)`(M,x)`, since this is an element of the subdifferential.

# Optional

* `p` – (`2`) the exponent of the distance,  i.e. the default is the squared
  distance
"""
gradDistance(M,y,x,p::Int=2) = (p==2) ? -log(M,x,y) : -distance(M,x,y)^(p-2)*log(M,x,y)

@doc doc"""
    ∇u,⁠∇v = gradIntrICTV12(M,f,u,v,α,β)

compute (sub)gradient of the intrinsic infimal convolution model using the mid point
model of second order differences, see [`costTV2`](@ref), i.e. for some $f\in\mathcal M$
on a [`Power`](@ref) manifold $\mathcal M$ this function computes the (sub)gradient of

```math
E(u,v) =
\frac{1}{2}\sum_{i\in\mathcal G} d_{\mathcal M}(g(\frac{1}{2},v_i,w_i),f_i)
+ \alpha
\bigl(
\beta\mathrm{TV}(v) + (1-\beta)\mathrm{TV}_2(w)
\bigr),
```
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

compute the (sub) gradient of $\frac{1}{p}d^p_{\mathcal M}(x,y)$ with respect
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

$F(x) = \sum_{i}\sum_{j\in\mathcal I_i} d^p(x_i,x_j)$

where $i$ runs over all indices of the [`Power`](@ref) manifold `M`
and $\mathcal I_i$ denotes the forward neighbors of $i$.

# Input
* `M` – a [`Power`](@ref) manifold
* `x` – a [`PowPoint`](@ref).

# Ouput
* ξ – resulting tangent vector in $T_x\mathcal M$.
"""
function gradTV(M::Power,x::PowPoint,p::Int=1)::PowTVector
  R = CartesianIndices(M.powerSize)
  d = length(M.powerSize)
  maxInd = last(R)
  ξ = zeroTVector(M,x)
  c = costTV(M,x,p,0)
  for i in R # iterate over all pixel
    di = 0.
    for k in 1:d # for all direction combinations
      ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
      j = i+ek # compute neighbor
      if all( map(<=, j.I, maxInd.I)) # is this neighbor in range?
        if p != 1
          g = (c[i]==0 ? 1 : 1/c[i]) .* gradTV(M.manifold,(x[i],x[j]),p) # Compute TV on these
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
    ξ = forwardLogs(M,x)

compute the forward logs $F$ (generalizing forward differences) orrucirng,
in the power manifold array, the function

```math
$F_i(x) = \sum_{j\in\mathcal I_i} \log_{x_i} x_j,\quad i \in \mathcal G,
```

where $\mathcal G$ is the set of indices of the [`Power`](@ref) manifold `M`
and $\mathcal I_i$ denotes the forward neighbors of $i$.

# Input
* `M` – a [`Power`](@ref) manifold
* `x` – a [`PowPoint`](@ref).

# Ouput
* `ξ` – resulting tangent vector in $T_x\mathcal M$ representing the logs, where
  $\mathcal N$ is thw power manifold with the number of dimensions added to `size(x)`.
"""
function forwardLogs(M::Power, x::PowPoint{P,Nt}) where {P <: MPoint, Nt}
  sX = size(x)
  R = CartesianIndices(sX)
  d = length(sX)
  maxInd = [last(R).I...] # maxInd as Array
  if d > 1
    d2 = fill(1,d+1)
    d2[d+1] = d
  else
    d2 = 1
  end
  N = Power(M.manifold, (prod(sX)*d,) )
  xT = PowPoint(repeat(getValue(x),inner=d2))
  ξ = zeroTVector(N,xT)
  for i in R # iterate over all pixel
    for k in 1:d # for all direction combinations
      I = [i.I...] # array of index
      J = I .+ 1 .* (1:d .== k) #i + e_k is j
      if all( J .<= maxInd ) # is this neighbor in range?
        j = CartesianIndex{d}(J...) # neigbbor index as Cartesian Index
        ξ[i,k] = log(M.manifold,x[i],x[j]) # Compute log and store in kth entry
      end
    end # directions
  end # i in R
  return ξ
end

@doc doc"""
    gradTV2(M,(x,y,z),p)

computes the (sub) gradient of $\frac{1}{p}d_2^p(x,y,z)$ with respect
to $x$, $y$, and $z$, where $d_2$ denotes the second order absolute difference
using the mid point model, i.e. let
```math
  \mathcal C = \bigl\{ c\in \mathcal M \ |\ g(\tfrac{1}{2};x_1,x_3) \text{ for some geodesic }g\bigr\}
```
denote the mid points between $x$ and $z$ on the manifold $\mathcal M$.
Then the absolute second order difference is defined as

```math
d_2(x,y,z) = \min_{c\in\mathcal C_{x,z}} d(c,y).
```

While the (sub)gradient with respect to $y$ is easy, the other two require
the evaluation of an [`adjointJacobiField`](@ref).
See [Illustration of the Gradient of a Second Order Difference](@ref secondOrderDifferenceGrad)
for its derivation.
"""
function gradTV2(M::mT where {mT <: Manifold}, xT::Tuple{P,P,P} where {P <: MPoint}, p::Number=1)
  x = xT[1];
  y = xT[2];
  z = xT[3];
  c = midPoint(M,x,z,y) # nearest mid point of x and z to y
  d = distance(M,y,c)
  innerLog = -log(M,c,y)
  if p==2
      return ( AdjDxGeo(M,x,z,1/2,innerLog), -log(M,y,c), AdjDyGeo(M,x,z,1/2,innerLog))
  else
    if d==0 # subdifferential containing zero
      return (zeroTVector(M,x),zeroTVector(M,y),zeroTVector(M,z))
    else
      return ( AdjDxGeo(M,x,z,1/2,innerLog/(d^(2-p))), -log(M,y,c)/(d^(2-p)), AdjDyGeo(M,x,z,1/2,innerLog/(d^(2-p))) )
    end
  end
end
@doc doc"""
    gradTV2(M,x [,p=1])

computes the (sub) gradient of $\frac{1}{p}d_2^p(x_1,x_2,x_3)$
with respect to all $x_1,x_2,x_3$ occuring along any array dimension in the
[`PowPoint`](@ref) `x`, where `M` is the corresponding [`Power`](@ref) manifold.
"""
function gradTV2(M::Power,x::PowPoint,p::Int=1)::PowTVector
  R = CartesianIndices(M.powerSize)
  d = length(M.powerSize)
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
          g = (c[i] == 0 ? 1 : 1/c[i]) .* gradTV2(M.manifold,(x[jB],x[i],x[jF]),p) # Compute TV2 on these
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
