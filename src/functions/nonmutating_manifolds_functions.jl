function adjoint_Jacobi_field(
    M::NONMUTATINGMANIFOLDS, p::Real, q::Real, t::Real, X::Real, β=βdifferential_geodesic_startpoint
)
    x = shortest_geodesic(M, p, q, t)
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M, p, q)))
    V = get_vectors(M, p, B)
    Θ = vector_transport_to.(Ref(M), Ref(p), V, Ref(x), Ref(ParallelTransport()))
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    Y = sum(
        (inner.(Ref(M), Ref(x), Ref(X), Θ)) .*
        (β.(B.data.eigenvalues, Ref(t), Ref(distance(M, p, q)))) .* V,
    )[1]
    return Y
end
function jacobi_field(
    M::NONMUTATINGMANIFOLDS, p::Real, q::Real, t::Real, X::Real, β=βdifferential_geodesic_startpoint
)
    x = shortest_geodesic(M, p, q, t)
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(log(M, p, q)))
    V = get_vectors(M, p, B)
    Θ = vector_transport_to.(Ref(M), Ref(p), V, Ref(x), Ref(ParallelTransport()))
    Y = zero_tangent_vector(M, p)
    # Decompose wrt. frame, multiply with the weights from w and recompose with Θ.
    Y = sum(
        (inner.(Ref(M), Ref(p), Ref(X), V)) .*
        (β.(B.data.eigenvalues, Ref(t), Ref(distance(M, p, q)))) .* Θ,
    )[1]
    return Y
end
function grad_TV2(M::NONMUTATINGMANIFOLDS, q, p::Number=1)
    c = mid_point(M, q[1], q[3], q[2]) # nearest mid point of x and z to y
    d = distance(M, q[2], c)
    innerLog = -log(M, c, q[2])
    X = [zero_tangent_vector(M, q[i]) for i in 1:3]
    if p == 2
        X[1] = adjoint_differential_geodesic_startpoint(M, q[1], q[3], 1 / 2, innerLog)
        X[2] = -log(M, q[2], c)
        X[3] = adjoint_differential_geodesic_endpoint(M, q[1], q[3], 1 / 2, innerLog)
    else
        if d > 0 # gradient case (subdifferential contains zero, see above)
            X[1] = adjoint_differential_geodesic_startpoint(
                M, q[1], q[3], 1 / 2, innerLog / (d^(2 - p))
            )
            X[2] = -log(M, q[2], c) / (d^(2 - p))
            X[3] = adjoint_differential_geodesic_endpoint(
                M, q[1], q[3], 1 / 2, innerLog / (d^(2 - p))
            )
        end
    end
    return X
end
function prox_TV2(::Circle, λ, pointTuple::Tuple{T,T,T}, p::Int=1) where {T}
    w = @SVector [1.0, -2.0, 1.0]
    x = SVector(pointTuple)
    if p == 1 # Theorem 3.5 in Bergmann, Laus, Steidl, Weinmann, 2014.
        sr_dot_xw = sym_rem(sum(x .* w))
        m = min(λ, abs(sr_dot_xw) / (dot(w, w)))
        s = sign(sr_dot_xw)
        return sym_rem.(x .- m .* s .* w)
    elseif p == 2 # Theorem 3.6 ibd.
        t = λ * sym_rem(sum(x .* w)) / (1 + λ * dot(w, w))
        return sym_rem.(x - t .* w)
    else
        throw(
            ErrorException(
                "Proximal Map of TV2(Circle,λ,pT,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
end