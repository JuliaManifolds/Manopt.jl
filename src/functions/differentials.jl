@doc raw"""
    DpGeo(M, p, q, t, X)
computes $D_p g(t;x,y)[\eta]$.

# See also
 [`DqGeo`](@ref), [`jacobiField`](@ref)
"""
DpGeo(M::mT,x,y,t,η) where {mT <: Manifold} = jacobiField(M,x,y,t,η,βDxg)
@doc raw"""
    DqGeo(M,x,y,t,η)
computes $D_yg(t;x,y)[\eta]$.

# See also
 [`DpGeo`](@ref), [`jacobiField`](@ref)
"""
DqGeo(M::mT, x, y, t, η) where {mT <: Manifold} = jacobiField(M,y,x,1-t,η,βDxg)
@doc raw"""
    DpExp(M, p, X, Y)

Compute $D_p\exp_p X[Y]$.

# See also
[`DXExp`](@ref), [`jacobiField`](@ref)
"""
DpExp(M::MT,p,X,Y) where {MT <: Manifold} = jacobiField(M, p, exp(M,p,X), 1.0, Y, βDpExp)
@doc raw"""
    DXExp(M,x,ξ,η)
computes $D_\xi\exp_x\xi[\eta]$.
Note that $\xi ∈  T_\xi(T_x\mathcal M) = T_x\mathcal M$ is still a tangent vector.

# See also
 [`DpExp`](@ref), [`jacobiField`](@ref)
"""
DξExp(M::MT, x, ξ, η) where {MT <: Manifold} = jacobiField(M,x,exp(M,x,ξ),1.0,η,βDXExp)
@doc raw"""
    DqLog(M,x,y,η)
computes $D_xlog_xy[\eta]$.

# See also
 [`DyLog`](@ref), [`jacobiField`](@ref)
"""
DqLog(M::mT, x, y, η) where {mT <: Manifold} = jacobiField(M,x,y,0.0,η,βDpLog)
@doc raw"""
    DyLog(M,x,y,η)
computes $D_ylog_xy[\eta]$.

# See also
 [`DqLog`](@ref), [`jacobiField`](@ref)
"""
DyLog(M::MT, x, y, η) where {MT <: Manifold} = jacobiField(M,y,x,1.0,η,βDqLog)

@doc raw"""
    ν = DforwardLogs(M,x,ξ)

compute the differenital of [`forwardLogs`](@ref) $F$ on the `PowerManifold` manifold
`M` at `x` and direction `ξ` ,
in the power manifold array, the differential of the function

```math
F_i(x) = \sum_{j ∈ \mathcal I_i} \log_{x_i} x_j$, \quad i  ∈  \mathcal G,
```

where $\mathcal G$ is the set of indices of the `PowerManifold` manifold `M`
and $\mathcal I_i$ denotes the forward neighbors of $i$.

# Input
* `M`     – a `PowerManifold` manifold
* `x`     – a point.
* `ξ`     – a tangent vector.

# Ouput
* `ν` – resulting tangent vector in $T_x\mathcal N$ representing the differentials of the logs, where
  $\mathcal N$ is thw power manifold with the number of dimensions added to `size(x)`.
"""
function DforwardLogs(M::PowerManifold{𝔽,MT,T,TPR}, p, X) where {𝔽,MT <: Manifold, T, TPR}
    power_size = [T.parameters...]
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = [last(R).I...]
    d2 = (d>1) ? ones(Int,d+1) + (d-1)*(1:(d+1) .== d+1 ) : 1
    if d > 1
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size...,d)
    else
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size...)
    end
    Y = zero_tangent_vector(N, repeat(p,inner=d2) )
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = [i.I...] # array of index
            J = I .+ 1 .* (1:d .== k) #i + e_k is j
            if all( J .<= maxInd )
                # this is neighbor in range,
                j = CartesianIndex{d}(J...) # neigbbor index as Cartesian Index
                # collects two, namely in kth direction since xi appears as base and arg
                Y[i,k] = DqLog(M.manifold,p[i],p[j],X[i]) + DyLog(M.manifold,p[i],p[j],X[j])
            end
        end # directions
    end # i in R
    return Y
end
