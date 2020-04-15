@doc raw"""
    βdifferential_geodesic_startpoint(κ,t,d)

weights for the [`jacobi_field`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_x g(t;p,q)[X]$.
They are

```math
\beta(\kappa) = \begin{cases}
\frac{\sinh(d(1-t)\sqrt{-\kappa})}{\sinh(d\sqrt{-\kappa})}
&\text{ if }\kappa < 0,\\
1-t & \text{ if } \kappa = 0,\\
\frac{\sin((1-t)d\sqrt{\kappa})}{\sinh(d\sqrt{\kappa})}
&\text{ if }\kappa > 0.
\end{cases}
```

Due to a symmetry agrument, these are also used to compute $D_q g(t; p,q)[\eta]$

# See also

[`differential_geodesic_endpoint`](@ref), [`differential_geodesic_startpoint`](@ref), [`jacobi_field`](@ref)
"""
function βdifferential_geodesic_startpoint(κ,t,d)
    if (d==0) || (κ==0)
        return (1-t)
    else
        if κ < 0
            return sinh(sqrt(-κ)*(1-t)*d)/sinh(sqrt(-κ)*d)
        elseif κ > 0
            return sin(sqrt(κ)*(1-t)*d)/sin(sqrt(κ)*d)
        end
    end
end
@doc raw"""
    βdifferential_exp_basepoint(κ,t,d)

weights for the [`jacobi_field`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_p \exp_p X [Y]$. They are

```math
\beta(\kappa) = \begin{cases}
\cosh(\sqrt{-\kappa})&\text{ if }\kappa < 0,\\
1 & \text{ if } \kappa = 0,\\
\cos(\sqrt{\kappa}) &\text{ if }\kappa > 0.
\end{cases}
```

# See also

[`differential_exp_basepoint`](@ref), [`jacobi_field`](@ref)
"""
function βdifferential_exp_basepoint(κ,t,d)
    if κ < 0
        return cosh(sqrt(-κ)*d)
    elseif κ > 0
        return cos(sqrt(κ)*d)
    else
        return 1.0;
    end
end
@doc raw"""
    βdifferential_exp_argument(κ,t,d)

weights for the [`jacobi_field`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_X \exp_p X[Y]$. They are

```math
$\beta(\kappa) = \begin{cases}
\frac{\sinh(d\sqrt{-\kappa})}{d\sqrt{-\kappa}}&\text{ if }\kappa < 0,\\
1 & \text{ if } \kappa = 0,\\
\frac{\sin(d\sqrt{\kappa})}{\sqrt{d\kappa}}&\text{ if }\kappa > 0.
\end{cases}
```

# See also

[`differential_exp_argument`](@ref), [`jacobi_field`](@ref)
"""
function βdifferential_exp_argument(κ,t,d)
    if (d==0) || (κ==0)
        return 1.0
    else
        if κ < 0
            return sinh(sqrt(-κ)*d)/( d*sqrt((-κ)) )
        elseif κ > 0
            return sin( sqrt(κ)*d )/( d*sqrt(κ) )
        end
    end
end
@doc raw"""
    βdifferential_log_basepoint(κ,t,d)

weights for the [`jacobi_field`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_p \log_p q[X]$. They are

```math
\beta(\kappa) = \begin{cases}
-\sqrt{-\kappa}d\frac{\cosh(d\sqrt{-\kappa})}{\sinh(d\sqrt{-\kappa})}&\text{ if }\kappa < 0,\\
-1 & \text{ if } \kappa = 0,\\
-\sqrt{\kappa}d\frac{\cos(d\sqrt{\kappa})}{\sin(d\sqrt{\kappa})}&\text{ if }\kappa > 0.
\end{cases}
```

# See also

[`differential_log_argument`](@ref), [`differential_log_argument`](@ref), [`jacobi_field`](@ref)
"""
function βdifferential_log_basepoint(κ::Number,t::Number,d::Number)
    if (d==0) || (κ==0)
        return -1.0
    else
        if κ < 0
            return - sqrt(-κ)*d*cosh(sqrt(-κ)*d)/sinh(sqrt(-κ)*d)
        else #if κ > 0
            return - sqrt(κ)*d*cos(sqrt(κ)*d)/sin(sqrt(κ)*d)
        end
    end
end
@doc raw"""
    βdifferential_log_argument(κ,t,d)

weights for the JacobiField corresponding to the differential of the logarithmic
map with respect to its argument $D_q \log_p q[X]$. They are

```math
\beta(\kappa) = \begin{cases}
\frac{ d\sqrt{-\kappa} }{\sinh(d\sqrt{-\kappa})}&\text{ if }\kappa < 0,\\
1 & \text{ if } \kappa = 0,\\
\frac{ d\sqrt{\kappa} }{\sin(d\sqrt{\kappa})}&\text{ if }\kappa > 0.
\end{cases}
```

# See also

[`differential_log_basepoint`](@ref), [`jacobi_field`](@ref)
"""
function βdifferential_log_argument(κ::Number,t::Number,d::Number)
    if (d==0) || (κ==0)
        return 1.0
    else
        if κ < 0
            return sqrt(-κ)*d/sinh(sqrt(-κ)*d)
        else #if κ > 0
            return sqrt(κ)*d/sin(sqrt(κ)*d)
        end
    end
end

@doc raw"""
    Y = adjoint_Jacobi_field(M, p, q, t, X, β)

Compute the AdjointJacobiField $J$ along the geodesic $γ_{p,q}$ on the manifold
$\mathcal M$ with initial conditions (depending on the application)
$X ∈ T_{γ_{p,q}(t)}\mathcal M$ and weights $β$. The result is a vector
$Y ∈ T_p\mathcal M$. The main difference to [`jacobi_field`](@ref) is the,
that the input `X` and the output `Y` switched tangent spaces.
For detais see [`jacobi_field`](@ref)
"""
function adjoint_Jacobi_field(M::Manifold, p, q, t, X, β::Function=βdifferential_geodesic_startpoint)
    x = shortest_geodesic(M, p, q, t)
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M,p,q)))
    V = get_vectors(M, p, B)
    Θ = vector_transport_to.(Ref(M),Ref(p), V, Ref(x), Ref(ParallelTransport()))
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    Y = sum(
        (
            inner.(Ref(M),Ref(x),Ref(X),Θ)
        ) .* (
            β.(B.data.eigenvalues,Ref(t),Ref(distance(M,p,q)))
        ) .* V
    )
    return Y
end
function adjoint_Jacobi_field(M::Circle{ℝ}, p::Real, q::Real, t::Real, X::Real, β::Function=βdifferential_geodesic_startpoint)
    x = shortest_geodesic(M, p, q, t)
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M,p,q)))
    V = get_vectors(M, p, B)
    Θ = vector_transport_to.(Ref(M),Ref(p), V, Ref(x), Ref(ParallelTransport()))
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    Y = sum(
        (
            inner.(Ref(M),Ref(x),Ref(X),Θ)
        ) .* (
            β.(B.data.eigenvalues,Ref(t),Ref(distance(M,p,q)))
        ) .* V
    )[1]
    return Y
end
function adjoint_Jacobi_field(M::AbstractPowerManifold, p, q, t, X, β::Function=βdifferential_geodesic_startpoint)
    rep_size = representation_size(M.manifold)
    Y = allocate_result(M, adjoint_Jacobi_field, p, X)
    for i in get_iterator(M)
        lY = adjoint_Jacobi_field(M.manifold, p[i], q[i], t, X[i], β)
        if size(lY) == ()
            Y[i] = lY
        else
            Y[i] .= lY
        end
    end
    return Y
end

@doc doc"""
    Y = jacobi_field(M, p, q, t, X, β)
compute the Jacobi jield $J$ along the geodesic $γ_{p,q}$ on the manifold $\mathcal M$ with
initial conditions (depending on the application) $X ∈ T_p\mathcal M$ and weights $β$. The
result is a tangent vector `Y` from $T_{γ_{p,q}(t)}\mathcal M$.

# See also

[`adjoint_Jacobi_field`](@ref)
"""
function jacobi_field(M::Manifold, p, q, t, X, β::Function=βdifferential_geodesic_startpoint)
    x = shortest_geodesic(M, p, q, t);
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M,p,q)))
    V = get_vectors(M, p, B)
    Θ = vector_transport_to.( Ref(M), Ref(p), V, Ref(x), Ref(ParallelTransport()))
    Y = zero_tangent_vector(M,p)
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    Y .= sum(
        (
            inner.(Ref(M), Ref(p), Ref(X), V)
        ) .* (
            β.(B.data.eigenvalues,Ref(t),Ref(distance(M,p,q)))
        ) .* Θ
    )
    return Y
end
function jacobi_field(M::Circle{ℝ}, p::Real, q::Real, t::Real, X::Real, β::Function=βdifferential_geodesic_startpoint)
    x = shortest_geodesic(M, p, q, t);
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M,p,q)))
    V = get_vectors(M, p, B)
    Θ = vector_transport_to.( Ref(M), Ref(p), V, Ref(x), Ref(ParallelTransport()))
    Y = zero_tangent_vector(M,p)
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    Y .= sum(
        (
            inner.(Ref(M), Ref(p), Ref(X), V)
        ) .* (
            β.(B.data.eigenvalues,Ref(t),Ref(distance(M,p,q)))
        ) .* Θ
    )[1]
    return Y
end
function jacobi_field(M::AbstractPowerManifold, p, q, t, X, β::Function=βdifferential_geodesic_startpoint)
    rep_size = representation_size(M.manifold)
    Y = allocate_result(M, adjoint_Jacobi_field, p, X)
    for i in get_iterator(M)
        lY = jacobi_field(M.manifold, p[i], q[i], t, X[i], β)
        if size(lY) == ()
            Y[i] = lY
        else
            Y[i] .= lY
        end
    end
    return Y
end
