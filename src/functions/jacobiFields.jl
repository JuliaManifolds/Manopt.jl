#
#
# JacobiFields
#
#
export βDxg, βDpExp, βDXExp, βDpLog, βDqLog

@doc raw"""
    βDxg(κ,t,d)

weights for the [`jacobiField`](@ref) corresponding to the differential of the geodesic
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

[`DqGeo`](@ref), [`DpGeo`](@ref), [`jacobiField`](@ref)
"""
function βDxg(κ,t,d)
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
    βDpExp(κ,t,d)

weights for the [`jacobiField`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_p \exp_p X [Y]$. They are

```math
\beta(\kappa) = \begin{cases}
\cosh(\sqrt{-\kappa})&\text{ if }\kappa < 0,\\
1 & \text{ if } \kappa = 0,\\
\cos(\sqrt{\kappa}) &\text{ if }\kappa > 0.
\end{cases}
```

# See also

[`DpExp`](@ref), [`jacobiField`](@ref)
"""
function βDpExp(κ,t,d)
    if κ < 0
        return cosh(sqrt(-κ)*d)
    elseif κ > 0
        return cos(sqrt(κ)*d)
    else
        return 1.0;
    end
end
@doc raw"""
    βDXExp(κ,t,d)

weights for the [`jacobiField`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_X \exp_p X[Y]$. They are

```math
$\beta(\kappa) = \begin{cases}
\frac{\sinh(d\sqrt{-\kappa})}{d\sqrt{-\kappa}}&\text{ if }\kappa < 0,\\
1 & \text{ if } \kappa = 0,\\
\frac{\sin(d\sqrt{\kappa})}{\sqrt{d\kappa}}&\text{ if }\kappa > 0.
\end{cases}
```

# See also

[`DξExp`](@ref), [`jacobiField`](@ref)
"""
function βDXExp(κ,t,d)
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
    βDpLog(κ,t,d)

weights for the [`jacobiField`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_p \log_p q[X]$. They are

```math
\beta(\kappa) = \begin{cases}
-\sqrt{-\kappa}d\frac{\cosh(d\sqrt{-\kappa})}{\sinh(d\sqrt{-\kappa})}&\text{ if }\kappa < 0,\\
-1 & \text{ if } \kappa = 0,\\
-\sqrt{\kappa}d\frac{\cos(d\sqrt{\kappa})}{\sin(d\sqrt{\kappa})}&\text{ if }\kappa > 0.
\end{cases}
```

# See also

[`DqLog`](@ref), [`DyLog`](@ref), [`jacobiField`](@ref)
"""
function βDpLog(κ::Number,t::Number,d::Number)
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
    βDqLog(κ,t,d)

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

[`DyLog`](@ref), [`jacobiField`](@ref)
"""
function βDqLog(κ::Number,t::Number,d::Number)
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
    Y = adjointJacobiField(M, p, q, t, X, β)

Compute the AdjointJacobiField $J$ along the geodesic $γ_{p,q}$ on the manifold
$\mathcal M$ with initial conditions (depending on the application)
$X ∈ T_{γ_{p,q}(t)}\mathcal M$ and weights $β$. The result is a vector
$Y ∈ T_p\mathcal M$. The main difference to [`jacobiField`](@ref) is the,
that the input `X` and the output `Y` switched tangent spaces.
For detais see [`jacobiField`](@ref)
"""
function adjointJacobiField(M::AbstractPowerManifold,p,q,t,X,β::Function=βDpGeo)
    rep_size = representation_size(M.manifold)
    for i in get_iterator(M)
        _write(M, rep_size, adjointJacobiField(M.manifold,
            _read(M, rep_size, p, i),
            _read(M, rep_size, q, i),
            t,
            _read(M, rep_size, X, i),
            β
            )
        )
    end
end
function adjointJacobiField(M::MT, p, q, t, X, β::Function=βDpGeo) where {MT<:Manifold}
    x = geodesic(M,p,q,t)
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M,p,q)))
    Θ = vector_transport_to.(Ref(M),Ref(p),B.vectors, Ref(x), Ref(ParallelTransport()))
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    ξ = sum(
        (inner.(Ref(M),Ref(x),Ref(X),Θ)) .* (β.(B.kappas,Ref(t),Ref(distance(M,p,q)))) .* B.vectors
    )
end

@doc doc"""
    Y = jacobiField(M, p, q, t, X, β)
compute the jacobiField $J$ along the geodesic $γ_{p,q}$ on the manifold $\mathcal M$ with
initial conditions (depending on the application) $X ∈ T_p\mathcal M$ and weights $β$. The
result is a tangent vector `Y` from $T_{γ_{p,q}(t)}\mathcal M$.

# See also

[`adjointJacobiField`](@ref)
"""
function jacobiField(M::MT, p, q, t, X, β::Function=βDgx) where {MT <: Manifold}
    x = geodesic(M, p, q, t);
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M,p,q)))
    Θ = vector_transport_to.( Ref(M), Ref(p), Ref(x), B.vectors, Ref(ParallelTransport()))
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    ξ = sum(
        (inner.(Ref(M),Ref(p),Ref(X),B.vectors)) .* (β.(B.kappas,Ref(t),Ref(distance(M,p,q)))) .* Θ
    )
end