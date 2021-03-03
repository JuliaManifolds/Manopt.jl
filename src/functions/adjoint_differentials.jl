@doc raw"""
    adjoint_differential_bezier_control(
        M::Manifold,
        b::BezierSegment,
        t::Float64,
        η::Q
    )
    adjoint_differential_bezier_control!(
        M::Manifold,
        Y::BezierSegment,
        b::BezierSegment,
        t::Float64,
        η::Q
    )

evaluate the adjoint of the differential of a Bézier curve on the manifold `M`
with respect to its control points `b` based on a point `t```∈[0,1]`` on the
curve and a tangent vector ``η∈T_{β(t)}\mathcal M``.
This can be computed in place of `Y`.

See [`de_casteljau`](@ref) for more details on the curve.
"""
function adjoint_differential_bezier_control(
    M::Manifold, b::BezierSegment, t::Float64, η::Q
) where {Q}
    n = length(b.pts)
    if n == 2
        return BezierSegment([
            adjoint_differential_geodesic_startpoint(M, b.pts[1], b.pts[2], t, η),
            adjoint_differential_geodesic_endpoint(M, b.pts[1], b.pts[2], t, η),
        ])
    end
    c = [b.pts, [similar.(b.pts[1:l]) for l in (n - 1):-1:2]...]
    for i in 2:(n - 1) # casteljau on the tree -> forward with interims storage
        c[i] .= shortest_geodesic.(Ref(M), c[i - 1][1:(end - 1)], c[i - 1][2:end], Ref(t))
    end
    Y = [η, [similar(η) for i in 1:(n - 1)]...]
    for i in (n - 1):-1:1 # propagate adjoints -> backward without interims storage
        Y[1:(n - i + 1)] .=
            [ # take previous results and add start&end point effects
                adjoint_differential_geodesic_startpoint.(
                    Ref(M), c[i][1:(end - 1)], c[i][2:end], Ref(t), Y[1:(n - i)]
                )...,
                zero_tangent_vector(M, c[i][end]),
            ] .+ [
                zero_tangent_vector(M, c[i][1]),
                adjoint_differential_geodesic_endpoint.(
                    Ref(M), c[i][1:(end - 1)], c[i][2:end], Ref(t), Y[1:(n - i)]
                )...,
            ]
    end
    return BezierSegment(Y)
end
function adjoint_differential_bezier_control!(
    M::Manifold, Y::BezierSegment, b::BezierSegment, t::Float64, η::Q
) where {Q}
    n = length(b.pts)
    if n == 2
        adjoint_differential_geodesic_startpoint!(M, Y.pts[1], b.pts[1], b.pts[2], t, η)
        adjoint_differential_geodesic_endpoint!(M, Y.pts[2], b.pts[1], b.pts[2], t, η)
        return Y
    end
    c = [b.pts, [similar.(b.pts[1:l]) for l in (n - 1):-1:2]...]
    for i in 2:(n - 1) # casteljau on the tree -> forward with interims storage
        c[i] .= shortest_geodesic.(Ref(M), c[i - 1][1:(end - 1)], c[i - 1][2:end], Ref(t))
    end
    Y.pts[1] = η
    for i in (n - 1):-1:1 # propagate adjoints -> backward without interims storage
        Y.pts[1:(n - i + 1)] .=
            [ # take previous results and add start&end point effects
                adjoint_differential_geodesic_startpoint.(
                    Ref(M), c[i][1:(end - 1)], c[i][2:end], Ref(t), Y.pts[1:(n - i)]
                )...,
                zero_tangent_vector(M, c[i][end]),
            ] .+ [
                zero_tangent_vector(M, c[i][1]),
                adjoint_differential_geodesic_endpoint.(
                    Ref(M), c[i][1:(end - 1)], c[i][2:end], Ref(t), Y.pts[1:(n - i)]
                )...,
            ]
    end
    return Y
end

@doc raw"""
    adjoint_differential_bezier_control(
        M::Manifold,
        b::BezierSegment,
        t::Array{Float64,1},
        X::Array{Q,1}
    )
    adjoint_differential_bezier_control!(
        M::Manifold,
        Y::BezierSegment,
        b::BezierSegment,
        t::Array{Float64,1},
        X::Array{Q,1}
    )
evaluate the adjoint of the differential of a Bézier curve on the manifold `M`
with respect to its control points `b` based on a points `T```=(t_i)_{i=1}^n`` that
are pointwise in `` t_i\in[0,1]`` on the curve and given corresponding tangential
vectors ``X = (η_i)_{i=1}^n``, ``η_i∈T_{β(t_i)}\mathcal M``
This can be computed in place of `Y`.

See [`de_casteljau`](@ref) for more details on the curve and[^BergmannGousenbourger2018].

[^BergmannGousenbourger2018]:
    > Bergmann, R. and Gousenbourger, P.-Y.: _A variational model for data fitting on manifolds
    > by minimizing the acceleration of a Bézier curve_.
    > Frontiers in Applied Mathematics and Statistics, 2018.
    > doi: [10.3389/fams.2018.00059](https://dx.doi.org/10.3389/fams.2018.00059),
    > arXiv: [1807.10090](https://arxiv.org/abs/1807.10090)
"""
function adjoint_differential_bezier_control(
    M::Manifold, b::BezierSegment, t::AbstractVector{Float64}, X::AbstractVector{Q}
) where {Q}
    effects = [bt.pts for bt in adjoint_differential_bezier_control.(Ref(M), Ref(b), t, X)]
    return BezierSegment(sum(effects))
end
function adjoint_differential_bezier_control!(
    M::Manifold,
    Y::BezierSegment,
    b::BezierSegment,
    t::AbstractVector{Float64},
    X::AbstractVector{Q},
) where {Q}
    Z = BezierSegment(similar.(Y.pts))
    fill!.(Y.pts, zero(eltype(first(Y.pts))))
    for i in 1:length(t)
        adjoint_differential_bezier_control!(M, Z, b, t[i], X[i])
        Y.pts .+= Z.pts
    end
    return Y
end

@doc raw"""
    adjoint_differential_bezier_control(
        M::MAnifold,
        B::AbstractVector{<:BezierSegment},
        t::Float64,
        X
    )
    adjoint_differential_bezier_control!(
        M::MAnifold,
        Y::AbstractVector{<:BezierSegment},
        B::AbstractVector{<:BezierSegment},
        t::Float64,
        X
    )

evaluate the adjoint of the differential of a composite Bézier curve on the
manifold `M` with respect to its control points `b` based on a points `T```=(t_i)_{i=1}^n``
that are pointwise in ``t_i\in[0,1]`` on the curve and given corresponding tangential
vectors ``X = (η_i)_{i=1}^n``, ``η_i∈T_{β(t_i)}\mathcal M``
This can be computed in place of `Y`.

See [`de_casteljau`](@ref) for more details on the curve.
"""
function adjoint_differential_bezier_control(
    M::Manifold, B::AbstractVector{<:BezierSegment}, t::Float64, X::Q
) where {Q}
    Y = broadcast(b -> BezierSegment(zero_tangent_vector.(Ref(M), b.pts)), B) # Double broadcast
    return adjoint_differential_bezier_control!(M, Y, B, t, X)
end
function adjoint_differential_bezier_control!(
    M::Manifold,
    Y::AbstractVector{<:BezierSegment},
    B::AbstractVector{<:BezierSegment},
    t::Float64,
    X::Q,
) where {Q}
    # doubly nested broadbast on the Array(Array) of CPs (note broadcast _and_ .)
    if (0 > t) || (t > length(B))
        error(
            "The parameter ",
            t,
            " to evaluate the composite Bézier curve at is outside the interval [0,",
            length(B),
            "].",
        )
    end
    for y in Y
        fill!.(y.pts, zero(eltype(first(y.pts))))
    end
    seg = max(ceil(Int, t), 1)
    localT = ceil(Int, t) == 0 ? 0.0 : t - seg + 1
    adjoint_differential_bezier_control!(M, Y[seg], B[seg], localT, X)
    return Y
end
@doc raw"""
    adjoint_differential_bezier_control(
        M::MAnifold,
        T::AbstractVector{<:Number},
        X::AbstractVector{Q},
    )
    adjoint_differential_bezier_control!(
        M::MAnifold,
        Y::AbstractVector{<:BezierSegment},
        T::AbstractVector{<:Number},
        X::AbstractVector{Q},
    )

Evaluate the adjoint of the differential with respect to the controlpoints at several times `T`.
This can be computed in place of `Y`.

See [`de_casteljau`](@ref) for more details on the curve.
"""
function adjoint_differential_bezier_control(
    M::Manifold,
    B::AbstractVector{<:BezierSegment},
    T::AbstractVector{<:Number},
    X::AbstractVector{Q},
) where {Q}
    Y = broadcast(b -> BezierSegment(zero_tangent_vector.(Ref(M), b.pts)), B) # Double broadcast
    return adjoint_differential_bezier_control!(M, Y, B, T, X)
end
function adjoint_differential_bezier_control!(
    M::Manifold,
    Y::AbstractVector{<:BezierSegment},
    B::AbstractVector{<:BezierSegment},
    T::AbstractVector{Float64},
    X::AbstractVector{Q},
) where {Q}
    Z = [BezierSegment(similar.(y.pts)) for y in Y]
    for j in 1:length(T) # for all times
        adjoint_differential_bezier_control!(M, Z, B, T[j], X[j])
        for i in 1:length(Z)
            Y[i].pts .+= Z[i].pts
        end
    end
    return Y
end
@doc raw"""
    adjoint_differential_geodesic_startpoint(M,p, q, t, X)
    adjoint_differential_geodesic_startpoint!(M, Y, p, q, t, X)

Compute the adjoint of ``D_p γ(t; p, q)[X]`` (in place of `Y`).

# See also

[`differential_geodesic_startpoint`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_geodesic_startpoint(M::Manifold, p, q, t, X)
    return adjoint_Jacobi_field(M, p, q, t, X, βdifferential_geodesic_startpoint)
end
function adjoint_differential_geodesic_startpoint!(M::Manifold, Y, p, q, t, X)
    return adjoint_Jacobi_field!(M, Y, p, q, t, X, βdifferential_geodesic_startpoint)
end

@doc raw"""
    adjoint_differential_geodesic_endpoint(M, p, q, t, X)
    adjoint_differential_geodesic_endpoint!(M, Y, p, q, t, X)

Compute the adjoint of ``D_q γ(t; p, q)[X]`` (in place of `Y`).

# See also

[`differential_geodesic_endpoint`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_geodesic_endpoint(M::Manifold, p, q, t, X)
    return adjoint_Jacobi_field(M, q, p, 1 - t, X, βdifferential_geodesic_startpoint)
end
function adjoint_differential_geodesic_endpoint!(M::Manifold, Y, p, q, t, X)
    return adjoint_Jacobi_field!(M, Y, q, p, 1 - t, X, βdifferential_geodesic_startpoint)
end

@doc raw"""
    adjoint_differential_exp_basepoint(M, p, X, Y)
    adjoint_differential_exp_basepoint!(M, Z, p, X, Y)

Computes the adjoint of ``D_p \exp_p X[Y]`` (in place of `Z`).

# See also

[`differential_exp_basepoint`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_exp_basepoint(M::Manifold, p, X, Y)
    return adjoint_Jacobi_field(M, p, exp(M, p, X), 1.0, Y, βdifferential_exp_basepoint)
end
function adjoint_differential_exp_basepoint!(M::Manifold, Z, p, X, Y)
    return adjoint_Jacobi_field!(M, Z, p, exp(M, p, X), 1.0, Y, βdifferential_exp_basepoint)
end

@doc raw"""
    adjoint_differential_exp_argument(M, p, X, Y)
    adjoint_differential_exp_argument!(M, Z, p, X, Y)

Compute the adjoint of ``D_X\exp_p X[Y]`` (in place of `Z`).
Note that ``X ∈  T_p(T_p\mathcal M) = T_p\mathcal M`` is still a tangent vector.

# See also

[`differential_exp_argument`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_exp_argument(M::Manifold, p, X, Y)
    return adjoint_Jacobi_field(M, p, exp(M, p, X), 1.0, Y, βdifferential_exp_argument)
end
function adjoint_differential_exp_argument!(M::Manifold, Z, p, X, Y)
    return adjoint_Jacobi_field!(M, Z, p, exp(M, p, X), 1.0, Y, βdifferential_exp_argument)
end

@doc raw"""
    adjoint_differential_log_basepoint(M, p, q, X)
    adjoint_differential_log_basepoint!(M, Y, p, q, X)

computes the adjoint of ``D_p log_p q[X]`` (in place of `Y`).

# See also
[`differential_log_basepoint`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_log_basepoint(M::Manifold, p, q, X)
    return adjoint_Jacobi_field(M, p, q, 0.0, X, βdifferential_log_basepoint)
end
function adjoint_differential_log_basepoint!(M::Manifold, Y, p, q, X)
    return adjoint_Jacobi_field!(M, Y, p, q, 0.0, X, βdifferential_log_basepoint)
end

@doc raw"""
    adjoint_differential_log_argument(M, p, q, X)
    adjoint_differential_log_argument!(M, Y, p, q, X)

Compute the adjoint of ``D_q log_p q[X]`` (in place of `Y`).

# See also
[`differential_log_argument`](@ref), [`adjoint_Jacobi_field`](@ref)
"""
function adjoint_differential_log_argument(M::Manifold, p, q, X)
    # order of p and q has to be reversed in this call, cf. Persch, 2018 Lemma 2.3
    return adjoint_Jacobi_field(M, q, p, 1.0, X, βdifferential_log_argument)
end
function adjoint_differential_log_argument!(M::Manifold, Y, p, q, X)
    return adjoint_Jacobi_field!(M, Y, q, p, 1.0, X, βdifferential_log_argument)
end

@doc raw"""
    Y = adjoint_differential_forward_logs(M, p, X)
    adjoint_differential_forward_logs!(M, Y, p, X)

Compute the adjoint differential of [`forward_logs`](@ref) ``F`` orrucirng,
in the power manifold array `p`, the differential of the function

``F_i(p) = \sum_{j ∈ \mathcal I_i} \log_{p_i} p_j``

where ``i`` runs over all indices of the `PowerManifold` manifold `M` and ``\mathcal I_i``
denotes the forward neighbors of ``i``
Let ``n`` be the number dimensions of the `PowerManifold` manifold (i.e. `length(size(x)`)).
Then the input tangent vector lies on the manifold ``\mathcal M' = \mathcal M^n``.
The adjoint differential can be computed in place of `Y`.

# Input

* `M`     – a `PowerManifold` manifold
* `p`     – an array of points on a manifold
* `X`     – a tangent vector to from the n-fold power of `p`, where n is the `ndims` of `p`

# Ouput

`Y` – resulting tangent vector in ``T_p\mathcal M`` representing the adjoint
  differentials of the logs.
"""
function adjoint_differential_forward_logs(
    M::PowerManifold{𝔽,TM,TSize,TPR}, p, X
) where {𝔽,TM,TSize,TPR}
    Y = zero_tangent_vector(M, p)
    return adjoint_differential_forward_logs!(M, Y, p, X)
end
function adjoint_differential_forward_logs!(
    M::PowerManifold{𝔽,TM,TSize,TPR}, Y, p, X
) where {𝔽,TM,TSize,TPR}
    power_size = power_dimensions(M)
    d = length(power_size)
    N = PowerManifold(M.manifold, TPR(), power_size..., d)
    R = CartesianIndices(Tuple(power_size))
    maxInd = last(R).I
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = [i.I...] # array of index
            J = I .+ 1 .* (1:d .== k) #i + e_k is j
            if all(J .<= maxInd) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neigbbor index as Cartesian Index
                Y[M, I...] =
                    Y[M, I...] + adjoint_differential_log_basepoint(
                        M.manifold, p[M, I...], p[M, J...], X[N, I..., k]
                    )
                Y[M, J...] =
                    Y[M, J...] + adjoint_differential_log_argument(
                        M.manifold, p[M, J...], p[M, I...], X[N, I..., k]
                    )
            end
        end # directions
    end # i in R
    return Y
end
