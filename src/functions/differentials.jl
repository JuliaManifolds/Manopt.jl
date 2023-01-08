@doc raw"""
    differential_bezier_control(M::AbstractManifold, b::BezierSegment, t::Float, X::BezierSegment)
    differential_bezier_control!(
        M::AbstractManifold,
        Y,
        b::BezierSegment,
        t,
        X::BezierSegment
    )

evaluate the differential of the Bézier curve with respect to its control points
`b` and tangent vectors `X` given in the tangent spaces of the control points. The result
is the “change” of the curve at `t```∈[0,1]``. The computation can be done in place of `Y`.

See [`de_casteljau`](@ref) for more details on the curve.
"""
function differential_bezier_control(
    M::AbstractManifold, b::BezierSegment, t, X::BezierSegment
)
    # iterative, because recursively would be too many Casteljau evals
    Y = similar(first(X.pts))
    return differential_bezier_control!(M, Y, b, t, X)
end
function differential_bezier_control!(
    M::AbstractManifold, Y, b::BezierSegment, t, X::BezierSegment
)
    # iterative, because recursively would be too many Casteljau evals
    Z = similar(X.pts)
    c = deepcopy(b.pts)
    for l in length(c):-1:2
        Z[1:(l - 1)] .=
            differential_shortest_geodesic_startpoint.(
                Ref(M), c[1:(l - 1)], c[2:l], Ref(t), X.pts[1:(l - 1)]
            ) .+
            differential_shortest_geodesic_endpoint.(
                Ref(M), c[1:(l - 1)], c[2:l], Ref(t), X.pts[2:l]
            )
        c[1:(l - 1)] = shortest_geodesic.(Ref(M), c[1:(l - 1)], c[2:l], Ref(t))
    end
    return copyto!(M, Y, Z[1])
end
@doc raw"""
    differential_bezier_control(
        M::AbstractManifold,
        b::BezierSegment,
        T::AbstractVector,
        X::BezierSegment,
    )
    differential_bezier_control!(
        M::AbstractManifold,
        Y,
        b::BezierSegment,
        T::AbstractVector,
        X::BezierSegment,
    )

evaluate the differential of the Bézier curve with respect to its control points
`b` and tangent vectors `X` in the tangent spaces of the control points. The result
is the “change” of the curve at the points `T`, elementwise in ``t∈[0,1]``.
The computation can be done in place of `Y`.

See [`de_casteljau`](@ref) for more details on the curve.
"""
function differential_bezier_control(
    M::AbstractManifold, b::BezierSegment, T::AbstractVector, X::BezierSegment
)
    return differential_bezier_control.(Ref(M), Ref(b), T, Ref(X))
end
function differential_bezier_control!(
    M::AbstractManifold, Y, b::BezierSegment, T::AbstractVector, X::BezierSegment
)
    return differential_bezier_control!.(Ref(M), Y, Ref(b), T, Ref(X))
end
@doc raw"""
    differential_bezier_control(
        M::AbstractManifold,
        B::AbstractVector{<:BezierSegment},
        t,
        X::AbstractVector{<:BezierSegment}
    )
    differential_bezier_control!(
        M::AbstractManifold,
        Y::AbstractVector{<:BezierSegment}
        B::AbstractVector{<:BezierSegment},
        t,
        X::AbstractVector{<:BezierSegment}
    )

evaluate the differential of the composite Bézier curve with respect to its
control points `B` and tangent vectors `Ξ` in the tangent spaces of the control
points. The result is the “change” of the curve at `t```∈[0,N]``, which depends
only on the corresponding segment. Here, ``N`` is the length of `B`.
The computation can be done in place of `Y`.

See [`de_casteljau`](@ref) for more details on the curve.
"""
function differential_bezier_control(
    M::AbstractManifold,
    B::AbstractVector{<:BezierSegment},
    t,
    X::AbstractVector{<:BezierSegment},
)
    if (0 > t) || (t > length(B))
        return throw(
            DomainError(
                t,
                "The parameter $(t) to evaluate the composite Bézier curve at is outside the interval [0,$(length(B))].",
            ),
        )
    end
    seg = max(ceil(Int, t), 1)
    localT = ceil(Int, t) == 0 ? 0.0 : t - seg + 1
    Y = differential_bezier_control(M, B[seg], localT, X[seg])
    if (Integer(t) == seg) && (seg < length(B)) # boundary case, -> seg-1 is also affecting the boundary differential
        Y .+= differential_bezier_control(M, B[seg + 1], localT - 1, X[seg + 1])
    end
    return Y
end
function differential_bezier_control!(
    M::AbstractManifold,
    Y,
    B::AbstractVector{<:BezierSegment},
    t,
    X::AbstractVector{<:BezierSegment},
)
    if (0 > t) || (t > length(B))
        return throw(
            DomainError(
                t,
                "The parameter $(t) to evaluate the composite Bézier curve at is outside the interval [0,$(length(B))].",
            ),
        )
    end
    seg = max(ceil(Int, t), 1)
    localT = ceil(Int, t) == 0 ? 0.0 : t - seg + 1
    differential_bezier_control!(M, Y, B[seg], localT, X[seg])
    if (Integer(t) == seg) && (seg < length(B)) # boundary case, -> seg-1 is also affecting the boundary differential
        Y .+= differential_bezier_control(M, B[seg + 1], localT - 1, X[seg + 1])
    end
    return Y
end

@doc raw"""
    differential_bezier_control(
        M::AbstractManifold,
        B::AbstractVector{<:BezierSegment},
        T::AbstractVector
        Ξ::AbstractVector{<:BezierSegment}
    )
    differential_bezier_control!(
        M::AbstractManifold,
        Θ::AbstractVector{<:BezierSegment}
        B::AbstractVector{<:BezierSegment},
        T::AbstractVector
        Ξ::AbstractVector{<:BezierSegment}
    )

evaluate the differential of the composite Bézier curve with respect to its
control points `B` and tangent vectors `Ξ` in the tangent spaces of the control
points. The result is the “change” of the curve at the points in `T`, which are elementwise
in ``[0,N]``, and each depending the corresponding segment(s). Here, ``N`` is the
length of `B`. For the mutating variant the result is computed in `Θ`.

See [`de_casteljau`](@ref) for more details on the curve and [^BergmannGousenbourger2018].

[^BergmannGousenbourger2018]:
    > Bergmann, R. and Gousenbourger, P.-Y.: _A variational model for data fitting on manifolds
    > by minimizing the acceleration of a Bézier curve_.
    > Frontiers in Applied Mathematics and Statistics, 2018.
    > doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059),
    > arXiv: [1807.10090](https://arxiv.org/abs/1807.10090)
"""
function differential_bezier_control(
    M::AbstractManifold,
    B::AbstractVector{<:BezierSegment},
    T::AbstractVector,
    Ξ::AbstractVector{<:BezierSegment},
)
    return differential_bezier_control.(Ref(M), Ref(B), T, Ref(Ξ))
end
function differential_bezier_control!(
    M::AbstractManifold,
    Y,
    B::AbstractVector{<:BezierSegment},
    T::AbstractVector,
    Ξ::AbstractVector{<:BezierSegment},
)
    return differential_bezier_control!.(Ref(M), Y, Ref(B), T, Ref(Ξ))
end

@doc raw"""
    Y = differential_forward_logs(M, p, X)
    differential_forward_logs!(M, Y, p, X)

compute the differential of [`forward_logs`](@ref) ``F`` on the `PowerManifold` manifold
`M` at `p` and direction `X` , in the power manifold array, the differential of the function

```math
F_i(x) = \sum_{j ∈ \mathcal I_i} \log_{p_i} p_j, \quad i ∈ \mathcal G,
```

where ``\mathcal G`` is the set of indices of the `PowerManifold` manifold `M`
and ``\mathcal I_i`` denotes the forward neighbors of ``i``.

# Input
* `M`     – a `PowerManifold` manifold
* `p`     – a point.
* `X`     – a tangent vector.

# Ouput
* `Y` – resulting tangent vector in ``T_x\mathcal N`` representing the differentials of the
    logs, where ``\mathcal N`` is the power manifold with the number of dimensions added
    to `size(x)`. The computation can also be done in place.
"""
function differential_forward_logs(M::PowerManifold, p, X)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R).I
    d2 = (d > 1) ? ones(Int, d + 1) + (d - 1) * (1:(d + 1) .== d + 1) : 1
    if d > 1
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size..., d)
    else
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size...)
    end
    Y = zero_vector(N, repeat(p; inner=d2))
    return differential_forward_logs!(M, Y, p, X)
end
function differential_forward_logs!(M::PowerManifold, Y, p, X)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R).I
    e_k_vals = [1 * (1:d .== k) for k in 1:d]
    if d > 1
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size..., d)
    else
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size...)
    end
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = i.I # array of index
            J = I .+ e_k_vals[k] #i + e_k is j
            if all(J .<= maxInd)
                # this is neighbor in range,
                # collects two, namely in kth direction since xi appears as base and arg
                Y[N, I..., k] =
                    differential_log_basepoint(
                        M.manifold, p[M, I...], p[M, J...], X[M, I...]
                    ) .+ differential_log_argument(
                        M.manifold, p[M, I...], p[M, J...], X[M, J...]
                    )
            else
                Y[N, I..., k] = zero_vector(M.manifold, p[M, I...])
            end
        end # directions
    end # i in R
    return Y
end
