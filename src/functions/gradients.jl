@doc raw"""
    ∇distance(M,y,x[, p=2])

compute the (sub)gradient of the distance (squared)

```math
f(x) = \frac{1}{2} d^p_{\mathcal M}(x,y)
```

to a fixed point `y` on the manifold `M` and `p` is an
integer. The gradient reads

```math
  \nabla f(x) = -d_{\mathcal M}^{p-2}(x,y)\log_xy
```

for $p\neq 1$ or $x\neq  y$. Note that for the remaining case $p=1$,
$x=y$ the function is not differentiable. This function returns then the
[`zero_tangent_vector`](@ref)`(M,x)`, since this is an element of the subdifferential.

# Optional

* `p` – (`2`) the exponent of the distance,  i.e. the default is the squared
  distance
"""
∇distance(M,y,x,p::Int=2) = (p==2) ? -log(M,x,y) : -distance(M,x,y)^(p-2)*log(M,x,y)

@doc raw"""
    ∇u,⁠∇v = ∇intrinsic_infimal_convolution_TV12(M,f,u,v,α,β)

compute (sub)gradient of the intrinsic infimal convolution model using the mid point
model of second order differences, see [`costTV2`](@ref), i.e. for some $f ∈ \mathcal M$
on a `PowerManifold` manifold $\mathcal M$ this function computes the (sub)gradient of

```math
E(u,v) =
\frac{1}{2}\sum_{i ∈ \mathcal G} d_{\mathcal M}(g(\frac{1}{2},v_i,w_i),f_i)
+ \alpha
\bigl(
\beta\mathrm{TV}(v) + (1-\beta)\mathrm{TV}_2(w)
\bigr),
```
where both total variations refer to the intrinsic ones, [`∇TV`](@ref) and
[`∇TV2`](@ref), respectively.
"""
function ∇intrinsic_infimal_convolution_TV12(M::mT,f,u,v,α,β) where {mT <: Manifold}
  c = mid_point(M,u,v,f)
  iL = log(M,c,f)
  return AdjDpGeo(M,u,v,1/2,iL) + α*β*∇TV(M,u), AdjDqGeo(M,u,v,1/2,iL) + α * (1-β) * ∇TV2(M,v)
end
@doc raw"""
    ∇TV(M,(x,y),[p=1])

compute the (sub) gradient of $\frac{1}{p}d^p_{\mathcal M}(x,y)$ with respect
to both $x$ and $y$.
"""
function ∇TV(M::MT, xT::Tuple{T,T}, p=1)where {MT <: Manifold, T}
  x = xT[1];
  y = xT[2];
  if p==2
      return (-log(M,x,y), -log(M,y,x))
  else
    d = distance(M,x,y);
    if d==0 # subdifferential containing zero
      return (zero_tangent_vector(M,x),zero_tangent_vector(M,y))
    else
      return (-log(M,x,y)/(d^(2-p)), -log(M,y,x)/(d^(2-p)))
    end
  end
end
@doc raw"""
    ξ = ∇TV(M,λ,x,[p])
Compute the (sub)gradient $\partial F$ of all forward differences orrucirng,
in the power manifold array, i.e. of the function

$F(x) = \sum_{i}\sum_{j ∈ \mathcal I_i} d^p(x_i,x_j)$

where $i$ runs over all indices of the `PowerManifold` manifold `M`
and $\mathcal I_i$ denotes the forward neighbors of $i$.

# Input
* `M` – a `PowerManifold` manifold
* `x` – a point.

# Ouput
* ξ – resulting tangent vector in $T_x\mathcal M$.
"""
function ∇TV(M::PowerManifold,x,p::Int=1)
    power_size = power_dimensions(M)
    rep_size = representation_size(M.manifold)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R)
    X = zero_tangent_vector(M,x)
    c = costTV(M,x,p,0)
    for i in R # iterate over all pixel
        di = 0.
        for k in 1:d # for all direction combinations
            ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
            j = i+ek # compute neighbor
            if all( map(<=, j.I, maxInd.I)) # is this neighbor in range?
                if p != 1
                    g = (c[i]==0 ? 1 : 1/c[i]) .* ∇TV(M.manifold,(x[i],x[j]),p) # Compute TV on these
                else
                    g = ∇TV(M.manifold,(x[i],x[j]),p) # Compute TV on these
                end
                X[i] += g[1]
                X[j] += g[2]
            end
        end # directions
    end # i in R
    return X
end

@doc raw"""
    ξ = forward_logs(M,x)

compute the forward logs $F$ (generalizing forward differences) orrucirng,
in the power manifold array, the function

```math
$F_i(x) = \sum_{j ∈ \mathcal I_i} \log_{x_i} x_j,\quad i  ∈  \mathcal G,
```

where $\mathcal G$ is the set of indices of the `PowerManifold` manifold `M`
and $\mathcal I_i$ denotes the forward neighbors of $i$.

# Input
* `M` – a `PowerManifold` manifold
* `x` – a point.

# Ouput
* `ξ` – resulting tangent vector in $T_x\mathcal M$ representing the logs, where
  $\mathcal N$ is thw power manifold with the number of dimensions added to `size(x)`.
"""
function forward_logs(M::PowerManifold, x)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    sX = size(x)
    maxInd = [last(R).I...] # maxInd as Array
    if d > 1
        d2 = fill(1,d+1)
        d2[d+1] = d
    else
        d2 = 1
    end
    N = PowerManifold(M.manifold, prod(sX)*d)
    xT = repeat(x,inner=d2)
    ξ = zero_tangent_vector(N,xT)
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

@doc raw"""
    ∇TV2(M,(x,y,z),p)

computes the (sub) gradient of $\frac{1}{p}d_2^p(x,y,z)$ with respect
to $x$, $y$, and $z$, where $d_2$ denotes the second order absolute difference
using the mid point model, i.e. let
```math
  \mathcal C = \bigl\{ c ∈  \mathcal M \ |\ g(\tfrac{1}{2};x_1,x_3) \text{ for some geodesic }g\bigr\}
```
denote the mid points between $x$ and $z$ on the manifold $\mathcal M$.
Then the absolute second order difference is defined as

```math
d_2(x,y,z) = \min_{c ∈ \mathcal C_{x,z}} d(c,y).
```

While the (sub)gradient with respect to $y$ is easy, the other two require
the evaluation of an [`adjoint_Jacobi_field`](@ref).
See [Illustration of the Gradient of a Second Order Difference](@ref secondOrderDifferenceGrad)
for its derivation.
"""
function ∇TV2(M::MT, xT, p::Number=1) where {MT <: Manifold}
  x = xT[1];
  y = xT[2];
  z = xT[3];
  c = mid_point(M,x,z,y) # nearest mid point of x and z to y
  d = distance(M,y,c)
  innerLog = -log(M,c,y)
  if p==2
      return ( AdjDpGeo(M,x,z,1/2,innerLog), -log(M,y,c), AdjDqGeo(M,x,z,1/2,innerLog))
  else
    if d==0 # subdifferential containing zero
      return (zero_tangent_vector(M,x),zero_tangent_vector(M,y),zero_tangent_vector(M,z))
    else
      return ( AdjDpGeo(M,x,z,1/2,innerLog/(d^(2-p))), -log(M,y,c)/(d^(2-p)), AdjDqGeo(M,x,z,1/2,innerLog/(d^(2-p))) )
    end
  end
end
@doc raw"""
    ∇TV2(M,x [,p=1])

computes the (sub) gradient of $\frac{1}{p}d_2^p(x_1,x_2,x_3)$
with respect to all $x_1,x_2,x_3$ occuring along any array dimension in the
point `x`, where `M` is the corresponding `PowerManifold`.
"""
function ∇TV2(M::PowerManifold, x, p::Int=1)
    power_size = power_dimensions(M)
    rep_size = representation_size(M.manifold)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    minInd, maxInd = first(R), last(R)
    ξ = zero_tangent_vector(M,x)
    c = costTV2(M,x,p,false)
    for i in R # iterate over all pixel
        di = 0.
        for k in 1:d # for all direction combinations (TODO)
            ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
            jF = i+ek # compute forward neighbor
            jB = i-ek # compute backward neighbor
            if all( map(<=, jF.I, maxInd.I) ) && all( map(>=, jB.I, minInd.I)) # are neighbors in range?
                if p != 1
                    g = (c[i] == 0 ? 1 : 1/c[i]) .* ∇TV2(M.manifold,(x[jB],x[i],x[jF]),p) # Compute TV2 on these
                else
                    g = ∇TV2(M.manifold,(x[jB],x[i],x[jF]),p) # Compute TV2 on these
                end
                ξ[jB] += g[1]
                ξ[i]  += g[2]
                ξ[jF] += g[3]
            end
        end # directions
    end # i in R
    return ξ
end
