function grad_TV2(M::NONMUTATINGMANIFOLDS, q, p::Int=1)
    c = mid_point(M, q[1], q[3], q[2]) # nearest mid point of x and z to y
    d = distance(M, q[2], c)
    innerLog = -log(M, c, q[2])
    X = [zero_vector(M, q[i]) for i in 1:3]
    if p == 2
        X[1] = adjoint_differential_shortest_geodesic_startpoint(
            M, q[1], q[3], 1 / 2, innerLog
        )
        X[2] = -log(M, q[2], c)
        X[3] = adjoint_differential_shortest_geodesic_endpoint(
            M, q[1], q[3], 1 / 2, innerLog
        )
    else
        if d > 0 # gradient case (subdifferential contains zero, see above)
            X[1] = adjoint_differential_shortest_geodesic_startpoint(
                M, q[1], q[3], 1 / 2, innerLog / (d^(2 - p))
            )
            X[2] = -log(M, q[2], c) / (d^(2 - p))
            X[3] = adjoint_differential_shortest_geodesic_endpoint(
                M, q[1], q[3], 1 / 2, innerLog / (d^(2 - p))
            )
        end
    end
    return X
end
function prox_TV2(::NONMUTATINGMANIFOLDS, λ, pointTuple::Tuple{T,T,T}, p::Int=1) where {T}
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
function prox_TV2(M::PowerManifold{𝔽,N}, λ, x, p::Int=1) where {𝔽,N<:NONMUTATINGMANIFOLDS}
    power_size = power_dimensions(M)
    R = CartesianIndices(power_size)
    d = length(size(x))
    minInd = first(R).I
    maxInd = last(R).I
    y = deepcopy(x)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:2
            for i in R # iterate over all pixel
                if (i[k] % 3) == l
                    JForward = i.I .+ ek.I #i + e_k
                    JBackward = i.I .- ek.I # i - e_k
                    if all(JForward .<= maxInd) && all(JBackward .>= minInd)
                        (y[M, JBackward...], y[M, i.I...], y[M, JForward...]) = prox_TV2(
                            M.manifold,
                            λ,
                            (y[M, JBackward...], y[M, i.I...], y[M, JForward...]),
                            p,
                        )
                    end
                end # if mod 3
            end # i in R
        end # for mod 3
    end # directions
    return y
end
