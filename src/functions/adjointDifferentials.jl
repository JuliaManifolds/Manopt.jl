@doc raw"""
    AdjDpGeo(M,p, q, t, X)

Compute the adjoint of $D_p γ(t; p, q)[X]$.

# See also

[`DpGeo`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDpGeo(M::MT, p, q, t, X) where {MT <: Manifold} = adjointJacobiField(M, p, q, t, X, βDxg)

@doc raw"""
    AdjDqGeo(M, p, q, t, X)

Compute the adjoint of $D_q γ(t; p, q)[X]$.

# See also

[`DqGeo`](@ref), [`adjointJacobiField`](@ref)
"""
function AdjDqGeo(M::MT, p, q, t, X) where {MT <: Manifold}
    return adjointJacobiField(M, q, p, 1-t, X, βDxg)
end

@doc raw"""
    AdjDpExp(M, p, X, Y)

Computes the adjoint of $D_p \exp_p X[Y]$.

# See also

[`DpExp`](@ref), [`adjointJacobiField`](@ref)
"""
function AdjDpExp(M::MT, p, X, Y) where {MT <: Manifold}
    return adjointJacobiField(M, p, exp(M, p, X), 1., Y, βDpExp)
end

@doc raw"""
    AdjDXExp(M, p, X, Y)

Compute the adjoint of $D_X\exp_p X[Y]$.
Note that $X ∈  T_p(T_p\mathcal M) = T_p\mathcal M$ is still a tangent vector.

# See also

[`DξExp`](@ref), [`adjointJacobiField`](@ref)
"""
function AdjDXExp(M::mT, p, X, Y) where {mT <: Manifold}
    return adjointJacobiField(M, p, exp(M, p, X), 1.0, Y, βDXExp)
end

@doc raw"""
    AdjDpLog(M, p, q, X)

computes the adjoint of $D_p log_p q[X]$.

# See also
[`DqLog`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDpLog(M::mT, p, q, X) where {mT <: Manifold} = adjointJacobiField(M, p, q, 0., X, βDpLog)

@doc raw"""
    AdjDqLog(M, p, q, X)

Compute the adjoint of $D_q log_p q[X]$.

# See also
[`DqLog`](@ref), [`adjointJacobiField`](@ref)
"""
AdjDqLog(M::MT, p, q, X) where {MT <: Manifold} = adjointJacobiField(M, p, q, 1., X, βDqLog)

@doc raw"""
    Y = AdjDforwardLogs(M, p, X)

Compute the adjoint differential of [`forwardLogs`](@ref) $F$ orrucirng,
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
function AdjDforwardLogs(M::PowerManifold{ℝ,MT,T}, p, X) where {MT <: Manifold, T}
    power_size = [T.parameters...]
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = [last(R).I...] # maxInd as Array
    N = M.manifold^(power_size...,d)
    Y = zero_tangent_vector(M,p)
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = [i.I...] # array of index
            J = I .+ 1 .* (1:d .== k) #i + e_k is j
            if all( J .<= maxInd ) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neigbbor index as Cartesian Index
                Y[i] += AdjDpLog(M.manifold,p[i],p[j],X[i,k])
                Y[j] += AdjDqLog(M.manifold,p[i],p[j],X[i,k])
            end
        end # directions
    end # i in R
    return Y
end