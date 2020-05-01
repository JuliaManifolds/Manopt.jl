
@doc raw"""
    cost_acceleration_bezier(M::Manifold, B::Array{Array{P,1},1}, pts::Array{Float64,1})

compute the value of the discrete Acceleration of the composite Bezier curve

$\sum_{i=1}^{N-1}\frac{d^2_2 [ B(t_{i-1}), B(t_{i}), B(t_{i+1})]}{\Delta_t^3}$

where for this formula the `pts` along the curve are equispaced and denoted by
$t_i$ and $d_2$ refers to the second order absolute difference [`costTV2`](@ref)
(squared).
"""
function cost_acceleration_bezier(
    M::Manifold,
    B::Array{P,1},1},
    pts::Array{Float64,1},
) where {P}
    p = de_casteljau(M.manifold,B,pts)
    n = length(p)
    f = p[ [1,3:n...,n] ]
    b = p[ [1,1:(n-2)...,n] ]
    d = distance.(Ref(M.manifold), p, geodesic.(Ref(M.manifold),f,b,Ref(0.5))).^2
    samplingFactor = 1/(( ( max(pts...) - min(pts...) )/(n-1) )^3)
    return samplingFactor*sum(d)
end
@doc raw"""
    cost_L2_acceleration_bezier(M,B,pts,λ,d)

compute the value of the discrete Acceleration of the composite Bezier curve
together with a data term, i.e.

````math
\frac{\lambda}{2}\sum_{i=0}^{m} d_{\mathcal M}(d_i, c_B(i))^2+
\sum_{i=1}^{m-1}\frac{d^2_2 [ B(t_{i-1}), B(t_{i}), B(t_{i+1})]}{\Delta_t^3}
````

where for this formula the `pts` along the curve are equispaced and denoted by
$t_i$ and $d_2$ refers to the second order absolute difference [`costTV2`](@ref)
(squared), the junction points are denoted by $p_i$, and to each $p_i$ corresponds
one data item in the manifold points given in `d`.
"""
function cost_L2_acceleration_bezier(
    M::Manifold,
    B::Array{Array{P,1},1},
    pts::Array{Float64,1},
    λ::Float64,
    d::Array{P,1}
) where {P}
    p = get_bezier_junctions(M.manifold,B)
    return costAccelerationBezier(M,B,pts) + λ/2*sum(distance.(Ref(M.manifold),p,d).^2)
end

@doc raw"""
    costIntrICTV12(M, f, u, v, α, β)

Compute the intrinsic infimal convolution model, where the addition is replaced
by a mid point approach and the two functions involved are [`costTV2`](@ref)
and [`costTV`](@ref). The model reads

```math
E(u,v) =
  \frac{1}{2}\sum_{i ∈ \mathcal G}
    d_{\mathcal M}\bigl(g(\frac{1}{2},v_i,w_i),f_i\bigr)
  +\alpha\bigl( \beta\mathrm{TV}(v) + (1-\beta)\mathrm{TV}_2(w) \bigr).
```
"""
function costIntrICTV12(M::Manifold, f, u, v, α, β)
    IC = 1/2*distance(M, shortest_geodesic(M, u, v, 0.5), f)^2
    TV12 = β * costTV(M, u) + (1-β) * costTV2(M, v)
    return  IC + α*TV12
end

@doc raw"""
    costL2TV(M, f, α, x)

compute the $\ell^2$-TV functional on the `PowerManifold manifold `M` for given
(fixed) data `f` (on `M`), a nonnegative weight `α`, and evaluated at `x` (on `M`),
i.e.

```math
E(x) = d_{\mathcal M}^2(f,x) + \alpha \operatorname{TV}(x)
```

# See also
[`costTV`](@ref)
"""
costL2TV(M, f, α, x) = 1/2 * distance(M, f, x)^2  +  α*costTV(M, x)

@doc raw"""
    costL2TVTV2(M, f, α, β, x)

compute the $\ell^2$-TV-TV2 functional on the `PowerManifold` manifold `M` for
given (fixed) data `f` (on `M`), nonnegative weight `α`, `β`, and evaluated
at `x` (on `M`), i.e.

```math
E(x) = d_{\mathcal M}^2(f,x) + \alpha\operatorname{TV}(x)
  + \beta\operatorname{TV}_2(x)
```

# See also
[`costTV`](@ref), [`costTV2`](@ref)
"""
costL2TVTV2(M::PowerManifold, f, α, β, x) = 1/2*distance(M,f,x)^2 + α*costTV(M,x) + β*costTV2(M,x)

@doc raw"""
    costL2TV2(M, f, β, x)

compute the $\ell^2$-TV2 functional on the `PowerManifold` manifold `M`
for given data `f`, nonnegative parameter `β`, and evaluated at `x`, i.e.

```math
E(x) = d_{\mathcal M}^2(f,x) + \beta\operatorname{TV}_2(x)
```

# See also
[`costTV2`](@ref)
"""
function costL2TV2(M::PowerManifold, f, β, x)
    return 1/2*distance(M,f,x)^2 + β*costTV2(M,x)
end

@doc raw"""
    costTV(M, x, p)

Compute the $\operatorname{TV}^p$ functional for a tuple `pT` of pointss
on a [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) `M`, i.e.

```math
E(x_1,x_2) = d_{\mathcal M}^p(x_1,x_2), \quad x_1,x_2 ∈ \mathcal M
```

# See also

[`∇TV`](@ref), [`prox_TV`](@ref)
"""
function costTV(M::Manifold, x::Tuple{T,T}, p::Int=1) where {T}
  return distance(M,x[1],x[2])^p
end
@doc raw"""
    costTV(M,x [,p=2,q=1])

Compute the $\operatorname{TV}^p$ functional for data `x`on the `PowerManifold`
manifold `M`, i.e. $\mathcal M = \mathcal N^n$, where $n ∈ \mathbb N^k$ denotes
the dimensions of the data `x`.
Let $\mathcal I_i$ denote the forward neighbors, i.e. with $\mathcal G$ as all
indices from $\mathbf{1} ∈ \mathbb N^k$ to $n$ we have
$\mathcal I_i = \{i+e_j, j=1,\ldots,k\}\cap \mathcal G$.
The formula reads

```math
E^q(x) = \sum_{i ∈ \mathcal G}
  \bigl( \sum_{j ∈  \mathcal I_i} d^p_{\mathcal M}(x_i,x_j) \bigr)^{q/p}.
```

# See also
[`∇TV`](@ref), [`prox_TV`](@ref)
"""
function costTV(M::PowerManifold, x, p=1, q=1)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R)
    cost = fill(0.,Tuple(power_size))
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
        for i in R # iterate over all pixel
            j = i+ek # compute neighbor
            if all( map(<=, j.I, maxInd.I)) # is this neighbor in range?
                cost[i] += costTV(M.manifold, (x[M,Tuple(i)...], x[M,Tuple(j)...]), p)
            end
        end
    end
    cost = (cost).^(1/p)
    if q > 0
        return sum(cost.^q)^(1/q)
    else
        return cost
    end
end
@doc raw"""
    costTV2(M,(x1,x2,x3) [,p=1])

Compute the $\operatorname{TV}_2^p$ functional for the 3-tuple of points
`(x1,x2,x3)`on the [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) `M`. Denote by

```math
  \mathcal C = \bigl\{ c ∈  \mathcal M \ |\ g(\tfrac{1}{2};x_1,x_3) \text{ for some geodesic }g\bigr\}
```

the set of mid points between $x_1$ and $x_3$. Then the functionr reads

$d_2^p(x_1,x_2,x_3) = \min_{c ∈ \mathcal C} d_{\mathcal M}(c,x_2).$

# See also
[`∇TV2`](@ref), [`prox_TV2`](@ref)
"""
function costTV2(M::MT, x::Tuple{T,T,T}, p=1) where {MT <: Manifold, T}
    # note that here mid_point returns the closest to x2 from the e midpoints between x1 x3
    return 1/p*distance(M,mid_point(M,x[1],x[3]),x[2])^p
end
@doc raw"""
    costTV2(M,x [,p=1])

compute the $\operatorname{TV}_2^p$ functional for data `x` on the
`PowerManifold` manifoldmanifold `M`, i.e. $\mathcal M = \mathcal N^n$,
where $n ∈ \mathbb N^k$ denotes the dimensions of the data `x`.
Let $\mathcal I_i^{\pm}$ denote the forward and backward neighbors, respectively,
i.e. with $\mathcal G$ as all indices from $\mathbf{1} ∈ \mathbb N^k$ to $n$ we
have $\mathcal I^\pm_i = \{i\pm e_j, j=1,\ldots,k\}\cap \mathcal I$.
The formula then reads

```math
E(x) = \sum_{i ∈ \mathcal I,\ j_1 ∈  \mathcal I^+_i,\ j_2 ∈  \mathcal I^-_i}
d^p_{\mathcal M}(c_i(x_{j_1},x_{j_2}), x_i),
```

where $c_i(\cdot,\cdot)$ denotes the mid point between its two arguments that is
nearest to $x_i$.

# See also
[`∇TV2`](@ref), [`prox_TV2`](@ref)
"""
function costTV2(M::PowerManifold, x, p::Int=1, Sum::Bool=true)
    Tt = Tuple( power_dimensions(M) )
    R = CartesianIndices( Tt )
    d = length(Tt)
    minInd, maxInd = first(R), last(R)
    cost = fill(0., Tt)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
        for i in R # iterate over all pixel
            jF = i+ek # compute forward neighbor
            jB = i-ek # compute backward neighbor
            if all( map(<=, jF.I, maxInd.I) ) && all( map(>=, jB.I, minInd.I)) # are neighbors in range?
                cost[i] += costTV2(
                    M.manifold,
                    (x[M,Tuple(jB)...], x[M,Tuple(i)...], x[M,Tuple(jF)...]),
                    p,
                )
            end
        end # i in R
  end # directions
  if p != 1
    cost = (cost).^(1/p)
  end
  if Sum
    return sum(cost)
  else
    return cost
  end
end
