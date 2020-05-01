@doc raw"""
    adjoint_differential_bezier_control(M,b,t,η)

evaluate the adjoint of the differential of a Bézier curve on the manifold `M`
with respect to its control points `b` based on a point `t` $\in[0,1]$ on the
curve and a tangent vector $\eta\in T_{\beta(t)}\mathcal M$.

See [`de_casteljau`](@ref) for more details on the curve.
"""
function adjoint_differential_bezier_control(
    M::Manifold,
    b::Array{P,1},
    t::Float64,
    η::Q) where {P,Q}
  if length(b) == 2
    return [
        adjoint_differential_geodesic_startpoint(M,b[1],b[2],t,η),
        adjoint_differential_geodesic_endpointl(M,b[1],b[2],t,η),
    ]
  else
    bInner = shortest_geodesic.(Ref(M), b[1:end-1], b[2:end],Ref(t))
    ξInner = adjoint_differential_bezier_control(M,bInner,t,η)
    startEffects = [
        adjoint_differential_geodesic_startpoint.(Ref(M),b[1:end-1], b[2:end],Ref(t),ξInner)...,
        zeroTVector(M,b[end]),
    ]
    endEffects = [
        zeroTVector(M,b[1]),
        adjoint_differential_geodesic_endpointl.(Ref(M),b[1:end-1], b[2:end],Ref(t),ξInner)...
    ]
    return startEffects .+ endEffects
  end
end
@doc raw"""
    adjoint_differential_bezier_control(M,b,t,X)

evaluate the adjoint of the differential of a Bézier curve on the manifold `M`
with respect to its control points `b` based on a points `T`$=(t_i)_{i=1}^n that
are pointwise in $ t_i\in[0,1]$ on the curve and given corresponding tangential
vectors $X = (\eta_i)_{i=1}^n$, $\eta_i\in T_{\beta(t_i)}\mathcal M$

See [`de_casteljau`](@ref) for more details on the curve.
"""
function adjoint_differential_bezier_control(
    M::Manifold,
    b::Array{P,1},
    t::Array{Float64,1},
    X::Array{Q,1}
) where {P,Q}
    return sum(adjoint_differential_bezier_control.(Ref(M),Ref(b),t,X))
end

@doc raw"""
    adjoint_differential_bezier_control(M,B,t,X)

evaluate the adjoint of the differential of a composite Bézier curve on the
manifold `M` with respect to its control points `b` based on a points `T`$=(t_i)_{i=1}^n$
that are pointwise in $t_i\in[0,1]$ on the curve and given corresponding tangential
vectors $X = (\eta_i)_{i=1}^n$, $\eta_i\in T_{\beta(t_i)}\mathcal M$

See [`de_casteljau`](@ref) for more details on the curve.
"""
function adjoint_differential_bezier_control(
    M::Manifold,
    B::Array{Array{P,1},1},
    t::Float64,
    X::Q
) where {P,Q}
  # doubly nested broadbast on the Array(Array) of CPs (note broadcast _and_ .)
  if (0 > t) || ( t > length(B) )
    error("The parameter ",t," to evaluate the composite Bézier curve at is outside the interval [0,",length(B),"].")
  end
  Y = broadcast( b -> zeroTVector.(Ref(M),b) , B) # Double broadcast
  seg = max(ceil(Int,t),1)
  localT = ceil(Int,t) == 0 ? 0. : t - seg+1
  Y[seg] = adjoint_differential_bezier_control(M,B[seg], localT, X)
  return Y
end
@doc raw"""
    adjoint_differential_bezier_control(M,B,t,η)
evaluate the adjoint of the differential with respect to the controlpoints.

See [`de_casteljau`](@ref) for more details on the curve.
"""
function adjoint_differential_bezier_control(
    M::Manifold,
    B::Array{Array{P,1},1},
    T::Array{Float64,1},
    X::Array{Q,1}
) where {P,Q}
  return broadcast(+, adjoint_differential_bezier_control.(Ref(M),Ref(B), T, X)... )
end

@doc raw"""
    adjoint_differential_geodesic_startpoint(M,p, q, t, X)

Compute the adjoint of $D_p γ(t; p, q)[X]$.

# See also

[`differential_geodesic_startpoint`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
adjoint_differential_geodesic_startpoint(M::MT, p, q, t, X) where {MT <: Manifold} = adjoint_Jacobi_field(M, p, q, t, X, βdifferential_geodesic_startpoint)

@doc raw"""
    adjoint_differential_geodesic_endpoint(M, p, q, t, X)

Compute the adjoint of $D_q γ(t; p, q)[X]$.

# See also

[`differential_geodesic_endpoint`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_geodesic_endpoint(M::MT, p, q, t, X) where {MT <: Manifold}
    return adjoint_Jacobi_field(M, q, p, 1-t, X, βdifferential_geodesic_startpoint)
end

@doc raw"""
    adjoint_differential_exp_basepoint(M, p, X, Y)

Computes the adjoint of $D_p \exp_p X[Y]$.

# See also

[`differential_exp_basepoint`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_exp_basepoint(M::MT, p, X, Y) where {MT <: Manifold}
    return adjoint_Jacobi_field(M, p, exp(M, p, X), 1., Y, βdifferential_exp_basepoint)
end

@doc raw"""
    adjoint_differential_exp_argument(M, p, X, Y)

Compute the adjoint of $D_X\exp_p X[Y]$.
Note that $X ∈  T_p(T_p\mathcal M) = T_p\mathcal M$ is still a tangent vector.

# See also

[`differential_exp_argument`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_exp_argument(M::mT, p, X, Y) where {mT <: Manifold}
    return adjoint_Jacobi_field(M, p, exp(M, p, X), 1.0, Y, βdifferential_exp_argument)
end

@doc raw"""
    adjoint_differential_log_basepoint(M, p, q, X)

computes the adjoint of $D_p log_p q[X]$.

# See also
[`differential_log_basepoint`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_log_basepoint(M::Manifold, p, q, X)
    return adjoint_Jacobi_field(M, p, q, 0., X, βdifferential_log_basepoint)
end

@doc raw"""
    adjoint_differential_log_argument(M, p, q, X)

Compute the adjoint of $D_q log_p q[X]$.

# See also
[`differential_log_argument`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_log_argument(M::Manifold, p, q, X)
    return adjoint_Jacobi_field(M, p, q, 1., X, βdifferential_log_argument)
end

@doc raw"""
    Y = adjoint_differential_forward_logs(M, p, X)

Compute the adjoint differential of [`forward_logs`](@ref) $F$ orrucirng,
in the power manifold array `p`, the differential of the function

$F_i(p) = \sum_{j ∈ \mathcal I_i} \log_{p_i} p_j$

where $i$ runs over all indices of the `PowerManifold` manifold `M` and $\mathcal I_i$
denotes the forward neighbors of $i$
Let $n$ be the number dimensions of the `PowerManifold` manifold (i.e. `length(size(x)`)).
Then the input tangent vector lies on the manifold $\mathcal M' = \mathcal M^n$.

# Input

* `M`     – a `PowerManifold` manifold
* `p`     – an array of points on a manifold
* `X`     – a tangent vector to from the n-fold power of `p`, where n is the `ndims` of `p`

# Ouput

`Y` – resulting tangent vector in $T_p\mathcal M$ representing the adjoint
  differentials of the logs.
"""
function adjoint_differential_forward_logs(M::PowerManifold{𝔽,TM,TSize,TPR}, p, X) where {𝔽,TM,TSize,TPR}
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = [last(R).I...] # maxInd as Array
    N = PowerManifold(M.manifold, TPR(), power_size..., d)
    Y = zero_tangent_vector(M,p)
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = [i.I...] # array of index
            J = I .+ 1 .* (1:d .== k) #i + e_k is j
            if all( J .<= maxInd ) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neigbbor index as Cartesian Index
                Y[M,I...] = Y[M,I...] + adjoint_differential_log_basepoint(
                    M.manifold,
                    p[M,I...],
                    p[M,J...],
                    X[N,I...,k],
                )
                Y[M,J...] = Y[M,J...] + adjoint_differential_log_argument(
                    M.manifold,
                    p[M,I...],
                    p[M,J...],
                    X[N,I...,k],
                )
            end
        end # directions
    end # i in R
    return Y
end
