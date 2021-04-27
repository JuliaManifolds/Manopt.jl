@doc raw"""
    βdifferential_geodesic_startpoint(κ,t,d)

weights for the [`jacobi_field`](@ref) corresponding to the differential of the geodesic
with respect to its start point ``D_x g(t;p,q)[X]``.
They are

```math
β(κ) = \begin{cases}
\frac{\sinh(d(1-t)\sqrt{-κ})}{\sinh(d\sqrt{-κ})}
&\text{ if }κ < 0,\\
1-t & \text{ if } κ = 0,\\
\frac{\sin((1-t)d\sqrt{κ})}{\sinh(d\sqrt{κ})}
&\text{ if }κ > 0.
\end{cases}
```

Due to a symmetry argument, these are also used to compute ``D_q g(t; p,q)[η]``

# See also

[`differential_geodesic_endpoint`](@ref), [`differential_geodesic_startpoint`](@ref), [`jacobi_field`](@ref)
"""
function βdifferential_geodesic_startpoint(κ, t, d)
    (d == 0) && return 1.0 - t
    (κ < 0) && return sinh(sqrt(-κ) * (1.0 - t) * d) / sinh(sqrt(-κ) * d)
    (κ > 0) && return sin(sqrt(κ) * (1.0 - t) * d) / sin(sqrt(κ) * d)
    return 1.0 - t # curvature zero
end
@doc raw"""
    βdifferential_exp_basepoint(κ,t,d)

weights for the [`jacobi_field`](@ref) corresponding to the differential of the geodesic
with respect to its start point ``D_p \exp_p X [Y]``. They are

```math
β(κ) = \begin{cases}
\cosh(\sqrt{-κ})&\text{ if }κ < 0,\\
1 & \text{ if } κ = 0,\\
\cos(\sqrt{κ}) &\text{ if }κ > 0.
\end{cases}
```

# See also

[`differential_exp_basepoint`](@ref), [`jacobi_field`](@ref)
"""
function βdifferential_exp_basepoint(κ, ::Number, d)
    (κ < 0) && return cosh(sqrt(-κ) * d)
    (κ > 0) && return cos(sqrt(κ) * d)
    return 1.0
end
@doc raw"""
    βdifferential_exp_argument(κ,t,d)

weights for the [`jacobi_field`](@ref) corresponding to the differential of the geodesic
with respect to its start point ``D_X \exp_p X[Y]``. They are

```math
β(κ) = \begin{cases}
\frac{\sinh(d\sqrt{-κ})}{d\sqrt{-κ}}&\text{ if }κ < 0,\\
1 & \text{ if } κ = 0,\\
\frac{\sin(d\sqrt{κ})}{d\sqrt{κ}}&\text{ if }κ > 0.
\end{cases}
```

# See also

[`differential_exp_argument`](@ref), [`jacobi_field`](@ref)
"""
function βdifferential_exp_argument(κ, ::Number, d)
    (d == 0) && return 1.0
    (κ < 0) && return sinh(sqrt(-κ) * d) / (d * sqrt((-κ)))
    (κ > 0) && return sin(sqrt(κ) * d) / (d * sqrt(κ))
    return 1.0 # cuvature zero.
end
@doc raw"""
    βdifferential_log_basepoint(κ,t,d)

weights for the [`jacobi_field`](@ref) corresponding to the differential of the geodesic
with respect to its start point ``D_p \log_p q[X]``. They are

```math
β(κ) = \begin{cases}
-\sqrt{-κ}d\frac{\cosh(d\sqrt{-κ})}{\sinh(d\sqrt{-κ})}&\text{ if }κ < 0,\\
-1 & \text{ if } κ = 0,\\
-\sqrt{κ}d\frac{\cos(d\sqrt{κ})}{\sin(d\sqrt{κ})}&\text{ if }κ > 0.
\end{cases}
```

# See also

[`differential_log_argument`](@ref), [`differential_log_argument`](@ref), [`jacobi_field`](@ref)
"""
function βdifferential_log_basepoint(κ, ::Number, d)
    (d == 0) && return -1.0
    (κ < 0) && return -sqrt(-κ) * d * cosh(sqrt(-κ) * d) / sinh(sqrt(-κ) * d)
    (κ > 0) && return -sqrt(κ) * d * cos(sqrt(κ) * d) / sin(sqrt(κ) * d)
    return -1.0 # cuvature zero.
end
@doc raw"""
    βdifferential_log_argument(κ,t,d)

weights for the JacobiField corresponding to the differential of the logarithmic
map with respect to its argument ``D_q \log_p q[X]``. They are

```math
β(κ) = \begin{cases}
\frac{ d\sqrt{-κ} }{\sinh(d\sqrt{-κ})}&\text{ if }κ < 0,\\
1 & \text{ if } κ = 0,\\
\frac{ d\sqrt{κ} }{\sin(d\sqrt{κ})}&\text{ if }κ > 0.
\end{cases}
```

# See also

[`differential_log_basepoint`](@ref), [`jacobi_field`](@ref)
"""
function βdifferential_log_argument(κ, ::Number, d)
    (d == 0) && return 1.0
    (κ < 0) && return sqrt(-κ) * d / sinh(sqrt(-κ) * d)
    (κ > 0) && return sqrt(κ) * d / sin(sqrt(κ) * d)
    return 1.0 # curvature zero
end

@doc raw"""
    Y = adjoint_Jacobi_field(M, p, q, t, X, β)
    adjoint_Jacobi_field!(M, Y, p, q, t, X, β)

Compute the AdjointJacobiField ``J`` along the geodesic ``γ_{p,q}`` on the manifold
``\mathcal M`` with initial conditions (depending on the application)
``X ∈ T_{γ_{p,q}(t)}\mathcal M`` and weights ``β``. The result is a vector
``Y ∈ T_p\mathcal M``. The main difference to [`jacobi_field`](@ref) is the,
that the input `X` and the output `Y` switched tangent spaces.
The computation can be done in place of `Y`.

For details see [`jacobi_field`](@ref)
"""
function adjoint_Jacobi_field(M::Manifold, p, q, t, X, β=βdifferential_geodesic_startpoint)
    x = shortest_geodesic(M, p, q, t)
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M, p, q)))
    V = get_vectors(M, p, B)
    Θ = vector_transport_to.(Ref(M), Ref(p), V, Ref(x), Ref(ParallelTransport()))
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    return sum(
        (inner.(Ref(M), Ref(x), Ref(X), Θ)) .*
        (β.(B.data.eigenvalues, Ref(t), Ref(distance(M, p, q)))) .* V,
    )
end
function adjoint_Jacobi_field!(
    M::Manifold, Y, p, q, t, X, β=βdifferential_geodesic_startpoint
)
    x = shortest_geodesic(M, p, q, t)
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M, p, q)))
    V = get_vectors(M, p, B)
    Θ = vector_transport_to.(Ref(M), Ref(p), V, Ref(x), Ref(ParallelTransport()))
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    Y .= sum(
        (inner.(Ref(M), Ref(x), Ref(X), Θ)) .*
        (β.(B.data.eigenvalues, Ref(t), Ref(distance(M, p, q)))) .* V,
    )
    return Y
end
function adjoint_Jacobi_field(
    M::AbstractPowerManifold, p, q, t, X, β=βdifferential_geodesic_startpoint
)
    Y = allocate_result(M, adjoint_Jacobi_field, p, X)
    for i in get_iterator(M)
        Y[M, i] = adjoint_Jacobi_field(M.manifold, p[M, i], q[M, i], t, X[M, i], β)
    end
    return Y
end
function adjoint_Jacobi_field!(
    M::AbstractPowerManifold, Y, p, q, t, X, β=βdifferential_geodesic_startpoint
)
    Z = deepcopy(first(Y))
    for i in get_iterator(M)
        adjoint_Jacobi_field!(M.manifold, Z, p[M, i], q[M, i], t, X[M, i], β)
        Y[M, i] = Z
    end
    return Y
end

@doc raw"""
    Y = jacobi_field(M, p, q, t, X, β)
    jacobi_field!(M, Y, p, q, t, X, β)

compute the Jacobi field ``J`` along the geodesic ``γ_{p,q}`` on the manifold ``\mathcal M`` with
initial conditions (depending on the application) ``X ∈ T_p\mathcal M`` and weights ``β``. The
result is a tangent vector `Y` from ``T_{γ_{p,q}(t)}\mathcal M``.
The computation can be done in place of `Y`.

# See also

[`adjoint_Jacobi_field`](@ref)
"""
function jacobi_field(M::Manifold, p, q, t, X, β=βdifferential_geodesic_startpoint)
    x = shortest_geodesic(M, p, q, t)
    Y = zero_tangent_vector(M, x)
    return jacobi_field!(M, Y, p, q, t, X, β)
end
function jacobi_field!(M::Manifold, Y, p, q, t, X, β=βdifferential_geodesic_startpoint)
    x = shortest_geodesic(M, p, q, t)
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M, p, q)))
    V = get_vectors(M, p, B)
    Θ = vector_transport_to.(Ref(M), Ref(p), V, Ref(x), Ref(ParallelTransport()))
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    Y .= sum(
        (inner.(Ref(M), Ref(p), Ref(X), V)) .*
        (β.(B.data.eigenvalues, Ref(t), Ref(distance(M, p, q)))) .* Θ,
    )
    return Y
end
function jacobi_field(
    M::AbstractPowerManifold, p, q, t, X, β=βdifferential_geodesic_startpoint
)
    Y = allocate_result(M, jacobi_field, p, X)
    for i in get_iterator(M)
        Y[M, i] = jacobi_field(M.manifold, p[M, i], q[M, i], t, X[M, i], β)
    end
    return Y
end
function jacobi_field!(
    M::AbstractPowerManifold, Y, p, q, t, X, β=βdifferential_geodesic_startpoint
)
    Z = deepcopy(first(Y))
    for i in get_iterator(M)
        jacobi_field!(M.manifold, Z, p[M, i], q[M, i], t, X[M, i], β)
        Y[M, i] = Z
    end
    return Y
end
