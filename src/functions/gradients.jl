@doc raw"""
    ∇acceleration_bezier(M::Manifold, B::Array{NTuple{N,P},1},T)

compute the gradient of the discretized acceleration of a (composite) Bézier curve $c_B(t)$
on the `Manifold` `M` with respect to its control points `B`. The curve is
evaluated at the points given in `T` (elementwise in $[0,N]$, where $N$ is the
number of segments of the Bézier curve).

See [`de_casteljau`](@ref) for more details on the curve.
"""
function ∇acceleration_bezier(
    M::Manifold,
    B::Array{P,1},
    T::Array{Float64,1};
    transport_method::AbstractVectorTransportMethod = ParallelTransport()
) where {P}
    gradB = _∇acceleration_bezier(M,B,T; transport_method = transport_method)
    for k=1:length(B) # we interpolate so we do not move end points
        m = length(B[k])
        gradB[k][1] = zero_tangent_vector(M, B[k][1])
        gradB[k][m] = zero_tangent_vector(M, B[k][m])
    end
    return gradB
end
function ∇acceleration_bezier(
    M::Manifold,
    b::NTuple{N,P},
    T::Array{Float64,1},
) where {P,N}
    gradb = _∇acceleration_bezier(M,[b],T)[1]
    gradb[1] = zero_tangent_vector(M,b[1])
    gradb[end] = zero_tangent_vector(M,b[end])
    return gradb
end

@doc raw"""
    ∇L2_acceleration_bezier(M,B,pts,λ,d)

compute the gradient of the discretized acceleration of a composite Bézier curve
on the `Manifold` `M` with respect to its control points `B` together with a
data term that relates the junction points `p_i` to the data `d` with a weigth
$\lambda$ comapred to the acceleration. The curve is evaluated at the points
given in `pts` (elementwise in $[0,N]$), where $N$ is the number of segments of
the Bézier curve.

See [`de_casteljau`](@ref) for more details on the curve.
"""
function ∇L2_acceleration_bezier(
    M::Manifold,
    B::Array{P,1},
    T::Array{Float64,1},
    λ::Float64,
    d::Array{Q,1};
    transport_method::AbstractVectorTransportMethod = ParallelTransport(),
) where {P,Q}
    gradB = _∇acceleration_bezier(M, B, T; transport_method = transport_method)
    # add start and end data grad
    gradB[1][1] = gradB[1][1] + λ*∇distance(M,B[1][1],first(d))
    # include data term
    for k=2:length(B)
        m = length(B[k])
        η = λ*∇distance(M, B[k][1], d[k])
        # copy to second entry
        gradB[k-1][m] = gradB[k-1][m] + η
        gradB[k][1] = gradB[k][1] + η
    end
    # add start and end data grad
    gradB[end][end] = gradB[end][end] + λ*∇distance(M,B[end][end],last(d))
    return gradB
end

# common helper for the two acceleration grads
function _∇acceleration_bezier(
    M::Manifold,
    B::Array{P,1},
    T::Array{Float64,1};
    transport_method::AbstractVectorTransportMethod = ParallelTransport()
) where {P}
    n = length(T)
    p = de_casteljau(M,B,T)
    center = p
    forward = p[ [1, 3:n..., n] ]
    backward = p[ [1,1:(n-2)..., n] ]
    mid = mid_point.(Ref(M), forward,backward)
    samplingFactor = 1/(( ( max(T...) - min(T...) )/(n-1) )^3)
    # where the point of interest appears...
    inner = -2 .* samplingFactor .* log.(Ref(M),mid,center)
    asForward = adjoint_differential_geodesic_startpoint.(Ref(M),forward,backward, Ref(0.5), inner)
    asCenter = - 2*samplingFactor*log.(Ref(M),center,mid)
    asBackward = adjoint_differential_geodesic_endpoint.(Ref(M),forward,backward, Ref(0.5), inner )
    # effect of these to the centrol points is the preliminary gradient
    ∇B =  adjoint_differential_bezier_control(
        M,
        B,
        T[ [1,3:n...,n]],asForward) .+ adjoint_differential_bezier_control(M,B,T,asCenter) .+ adjoint_differential_bezier_control(M,B,T[ [1,1:(n-2)...,n] ],asBackward)
    # include c0 & C1 condition
    for k=length(B):-1:2
        m = length(B[k])
        # updates b-, b+, p
        X1 = ∇B[k-1][m-1] + adjoint_differential_geodesic_startpoint(M, B[k-1][m-1],B[k][1],2.,∇B[k][2])
        X2 = ∇B[k][2] + adjoint_differential_geodesic_startpoint(M, B[k][2],B[k][1],2., -∇B[k-1][m-1])
        X3 = ∇B[k-1][m] + ∇B[k][1] + adjoint_differential_geodesic_endpoint(M, B[k-1][m-1], B[k][1], 2., ∇B[k][2])
        ∇B[k-1][m-1] .= X1
        ∇B[k][2] .= X2
        ∇B[k][1] .= X3
        ∇B[k-1][m] .= - vector_transport(M, B[k][1], X3, B[k-1][m], transport_method)
    end
    return ∇B
end

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
$x=y$ the function is not differentiable. In this case, the function returns the
corresponding zero tangent vector, since this is an element of the subdifferential.

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
  return adjoint_differential_geodesic_startpoint(M,u,v,1/2,iL) + α*β*∇TV(M,u), adjoint_differential_geodesic_endpoint(M,u,v,1/2,iL) + α * (1-β) * ∇TV2(M,v)
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
function forward_logs(M::PowerManifold{𝔽,TM,TSize,TPR}, p) where {𝔽,TM,TSize,TPR}
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    sX = size(p)
    maxInd = [last(R).I...] # maxInd as Array
    if d > 1
        d2 = fill(1,d+1)
        d2[d+1] = d
    else
        d2 = 1
    end
    N = PowerManifold(M.manifold, TPR(), power_size..., d)
    xT = repeat(p,inner=d2)
    X = zero_tangent_vector(N,xT)
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = [i.I...] # array of index
            J = I .+ 1 .* (1:d .== k) #i + e_k is j
            if all( J .<= maxInd ) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neigbbor index as Cartesian Index
                X[N, Tuple(i)..., k] = log(M.manifold,p[M, Tuple(i)...], p[M, Tuple(j)...])
            end
        end # directions
    end # i in R
    return X
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
      return ( adjoint_differential_geodesic_startpoint(M,x,z,1/2,innerLog), -log(M,y,c), adjoint_differential_geodesic_endpoint(M,x,z,1/2,innerLog))
  else
    if d==0 # subdifferential containing zero
      return (zero_tangent_vector(M,x),zero_tangent_vector(M,y),zero_tangent_vector(M,z))
    else
      return ( adjoint_differential_geodesic_startpoint(M,x,z,1/2,innerLog/(d^(2-p))), -log(M,y,c)/(d^(2-p)), adjoint_differential_geodesic_endpoint(M,x,z,1/2,innerLog/(d^(2-p))) )
    end
  end
end
@doc raw"""
    ∇TV2(M,q [,p=1])

computes the (sub) gradient of $\frac{1}{p}d_2^p(x_1,x_2,x_3)$
with respect to all $x_1,x_2,x_3$ occuring along any array dimension in the
point `x`, where `M` is the corresponding `PowerManifold`.
"""
function ∇TV2(M::PowerManifold, q, p::Int=1)
    power_size = power_dimensions(M)
    rep_size = representation_size(M.manifold)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    minInd, maxInd = first(R), last(R)
    X = zero_tangent_vector(M,q)
    c = costTV2(M,q,p,false)
    for i in R # iterate over all pixel
        di = 0.
        for k in 1:d # for all direction combinations (TODO)
            ek = CartesianIndex(ntuple(i  ->  (i==k) ? 1 : 0, d) ) #k th unit vector
            jF = i+ek # compute forward neighbor
            jB = i-ek # compute backward neighbor
            if all( map(<=, jF.I, maxInd.I) ) && all( map(>=, jB.I, minInd.I)) # are neighbors in range?
                if p != 1
                    g = (c[i] == 0 ? 1 : 1/c[i]) .* ∇TV2(M.manifold,(q[jB],q[i],q[jF]),p) # Compute TV2 on these
                else
                    g = ∇TV2(M.manifold,(q[jB],q[i],q[jF]),p) # Compute TV2 on these
                end
                X[M,Tuple(jB)...] = g[1]
                X[M,Tuple(i)...] = g[2]
                X[M,Tuple(jF)...] = g[3]
            end
        end # directions
    end # i in R
    return X
end
