@doc raw"""
    differential_geodesic_startpoint(M, p, q, t, X)
computes $D_p g(t;p,q)[\eta]$.

# See also
 [`differential_geodesic_endpoint`](@ref), [`jacobi_field`](@ref)
"""
differential_geodesic_startpoint(M::mT,x,y,t,η) where {mT <: Manifold} = jacobi_field(M,x,y,t,η,βDxg)
@doc raw"""
    differential_geodesic_endpoint(M,x,y,t,η)
computes $D_qg(t;p,q)[\eta]$.

# See also
 [`differential_geodesic_startpoint`](@ref), [`jacobi_field`](@ref)
"""
differential_geodesic_endpoint(M::mT, x, y, t, η) where {mT <: Manifold} = jacobi_field(M,y,x,1-t,η,βDxg)
@doc raw"""
    differential_exp_basepoint(M, p, X, Y)

Compute $D_p\exp_p X[Y]$.

# See also
[`differential_exp_argument`](@ref), [`jacobi_field`](@ref)
"""
function differential_exp_basepoint(M::Manifold,p,X,Y)
    return jacobi_field(M, p, exp(M,p,X), 1.0, Y, βdifferential_exp_basepoint)
end
@doc raw"""
    differential_exp_argument(M, p, X, Y)
computes $D_X\exp_pX[Y]$.
Note that $X ∈  T_X(T_p\mathcal M) = T_p\mathcal M$ is still a tangent vector.

# See also
 [`differential_exp_basepoint`](@ref), [`jacobi_field`](@ref)
"""
function differential_exp_argument(M::Manifold, p, X, Y)
    return jacobi_field(M,p,exp(M,p,X),1.0,Y,βdifferential_exp_argument)
end

@doc raw"""
    differential_log_basepoint(M, p, q, X)
computes $D_p\log_pq[X]$.

# See also
 [`differential_log_argument`](@ref), [`jacobi_field`](@ref)
"""
function differential_log_basepoint(M::Manifold, p, q, X)
    return jacobi_field(M,p,q,0.0,X,βdifferential_log_basepoint)
end

@doc raw"""
    differential_log_argument(M,p,q,X)
computes $D_q\log_p,q[X]$.

# See also
 [`differential_log_argument`](@ref), [`jacobi_field`](@ref)
"""
function differential_log_argument(M::Manifold, p, q, X)
    return jacobi_field(M,q,p,1.0,X,βdifferential_log_argument)
end

@doc raw"""
    Y = differential_forward_logs(M, p, X)

compute the differenital of [`forward_logs`](@ref) $F$ on the `PowerManifold` manifold
`M` at `p` and direction `X` ,
in the power manifold array, the differential of the function

```math
F_i(x) = \sum_{j ∈ \mathcal I_i} \log_{p_i} p_j$, \quad i  ∈  \mathcal G,
```

where $\mathcal G$ is the set of indices of the `PowerManifold` manifold `M`
and $\mathcal I_i$ denotes the forward neighbors of $i$.

# Input
* `M`     – a `PowerManifold` manifold
* `p`     – a point.
* `X`     – a tangent vector.

# Ouput
* `Y` – resulting tangent vector in $T_x\mathcal N$ representing the differentials of the logs, where
  $\mathcal N$ is thw power manifold with the number of dimensions added to `size(x)`.
"""
function differential_forward_logs(M::PowerManifold, p, X)
    power_size = power_dimensions(M)
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
                Y[i,k] = differential_log_argument(M.manifold,p[i],p[j],X[i]) + differential_log_argument(M.manifold,p[i],p[j],X[j])
            end
        end # directions
    end # i in R
    return Y
end
