@doc raw"""
    grad_acceleration_bezier(
        M::AbstractManifold,
        B::AbstractVector,
        degrees::AbstractVector{<:Integer}
        T::AbstractVector
    )

compute the gradient of the discretized acceleration of a (composite) Bézier curve ``c_B(t)``
on the `Manifold` `M` with respect to its control points `B` given as a point on the
`PowerManifold` assuming C1 conditions and known `degrees`. The curve is
evaluated at the points given in `T` (elementwise in ``[0,N]``, where ``N`` is the
number of segments of the Bézier curve). The [`get_bezier_junctions`](@ref) are fixed for
this gradient (interpolation constraint). For the unconstrained gradient,
see [`grad_L2_acceleration_bezier`](@ref) and set ``λ=0`` therein. This gradient is computed using
[`adjoint_Jacobi_field`](@ref)s. For details, see [^BergmannGousenbourger2018].
See [`de_casteljau`](@ref) for more details on the curve.

# See also

[`cost_acceleration_bezier`](@ref),  [`grad_L2_acceleration_bezier`](@ref), [`cost_L2_acceleration_bezier`](@ref).

[^BergmannGousenbourger2018]:
    > Bergmann, R. and Gousenbourger, P.-Y.: A variational model for data fitting on
    > manifolds by minimizing the acceleration of a Bézier curve.
    > Frontiers in Applied Mathematics and Statistics (2018).
    > doi [10.3389/fams.2018.00059](http://dx.doi.org/10.3389/fams.2018.00059),
    > arXiv: [1807.10090](https://arxiv.org/abs/1807.10090)
"""
function grad_acceleration_bezier(
    M::AbstractManifold,
    B::AbstractVector,
    degrees::AbstractVector{<:Integer},
    T::AbstractVector,
)
    gradB = _grad_acceleration_bezier(M, B, degrees, T)
    Bt = get_bezier_segments(M, B, degrees, :differentiable)
    for k in 1:length(Bt) # we interpolate so we do not move end points
        zero_vector!(M, gradB[k].pts[end], Bt[k].pts[end])
        zero_vector!(M, gradB[k].pts[1], Bt[k].pts[1])
    end
    zero_vector!(M, gradB[end].pts[end], Bt[end].pts[end])
    return get_bezier_points(M, gradB, :differentiable)
end
function grad_acceleration_bezier(M::AbstractManifold, b::BezierSegment, T::AbstractVector)
    gradb = _grad_acceleration_bezier(M, b.pts, [get_bezier_degree(M, b)], T)[1]
    zero_vector!(M, gradb.pts[1], b.pts[1])
    zero_vector!(M, gradb.pts[end], b.pts[end])
    return gradb
end

@doc raw"""
    grad_L2_acceleration_bezier(
        M::AbstractManifold,
        B::AbstractVector{P},
        degrees::AbstractVector{<:Integer},
        T::AbstractVector,
        λ,
        d::AbstractVector{P}
    ) where {P}

compute the gradient of the discretized acceleration of a composite Bézier curve
on the `Manifold` `M` with respect to its control points `B` together with a
data term that relates the junction points `p_i` to the data `d` with a weight
``λ`` compared to the acceleration. The curve is evaluated at the points
given in `pts` (elementwise in ``[0,N]``), where ``N`` is the number of segments of
the Bézier curve. The summands are [`grad_distance`](@ref) for the data term
and [`grad_acceleration_bezier`](@ref) for the acceleration with interpolation constrains.
Here the [`get_bezier_junctions`](@ref) are included in the optimization, i.e. setting ``λ=0``
yields the unconstrained acceleration minimization. Note that this is ill-posed, since
any Bézier curve identical to a geodesic is a minimizer.

Note that the Beziér-curve is given in reduces form as a point on a `PowerManifold`,
together with the `degrees` of the segments and assuming a differentiable curve, the segments
can internally be reconstructed.

# See also

[`grad_acceleration_bezier`](@ref), [`cost_L2_acceleration_bezier`](@ref), [`cost_acceleration_bezier`](@ref).
"""
function grad_L2_acceleration_bezier(
    M::AbstractManifold,
    B::AbstractVector{P},
    degrees::AbstractVector{<:Integer},
    T::AbstractVector,
    λ,
    d::AbstractVector{P},
) where {P}
    gradB = _grad_acceleration_bezier(M, B, degrees, T)
    Bt = get_bezier_segments(M, B, degrees, :differentiable)
    # add start and end data grad
    # include data term
    for k in 1:length(Bt)
        gradB[k].pts[1] .+= λ * grad_distance(M, d[k], Bt[k].pts[1])
        if k > 1
            gradB[k - 1].pts[end] .+= λ * grad_distance(M, d[k], Bt[k].pts[1])
        end
    end
    gradB[end].pts[end] .+= λ * grad_distance(M, d[end], Bt[end].pts[end])
    return get_bezier_points(M, gradB, :differentiable)
end

# common helper for the two acceleration grads
function _grad_acceleration_bezier(
    M::AbstractManifold,
    B::AbstractVector,
    degrees::AbstractVector{<:Integer},
    T::AbstractVector,
)
    Bt = get_bezier_segments(M, B, degrees, :differentiable)
    n = length(T)
    m = length(Bt)
    p = de_casteljau(M, Bt, T)
    center = p
    forward = p[[1, 3:n..., n]]
    backward = p[[1, 1:(n - 2)..., n]]
    mid = mid_point.(Ref(M), backward, forward)
    # where the point of interest appears...
    dt = (max(T...) - min(T...)) / (n - 1)
    inner = -2 / ((dt)^3) .* log.(Ref(M), mid, center)
    asForward =
        adjoint_differential_shortest_geodesic_startpoint.(
            Ref(M), forward, backward, Ref(0.5), inner
        )
    asCenter = -2 / ((dt)^3) * log.(Ref(M), center, mid)
    asBackward =
        adjoint_differential_shortest_geodesic_endpoint.(
            Ref(M), forward, backward, Ref(0.5), inner
        )
    # effect of these to the centrol points is the preliminary gradient
    grad_B = [
        BezierSegment(a.pts .+ b.pts .+ c.pts) for (a, b, c) in zip(
            adjoint_differential_bezier_control(M, Bt, T[[1, 3:n..., n]], asForward),
            adjoint_differential_bezier_control(M, Bt, T, asCenter),
            adjoint_differential_bezier_control(M, Bt, T[[1, 1:(n - 2)..., n]], asBackward),
        )
    ]
    for k in 1:(length(Bt) - 1) # add both effects of left and right segments
        X = grad_B[k + 1].pts[1] + grad_B[k].pts[end]
        grad_B[k].pts[end] .= X
        grad_B[k + 1].pts[1] .= X
    end
    # include c0 & C1 condition
    for k in length(Bt):-1:2
        m = length(Bt[k].pts)
        # updates b-
        X1 =
            grad_B[k - 1].pts[end - 1] .+ adjoint_differential_shortest_geodesic_startpoint(
                M, Bt[k - 1].pts[end - 1], Bt[k].pts[1], 2.0, grad_B[k].pts[2]
            )
        # update b+ - though removed in reduced form
        X2 =
            grad_B[k].pts[2] .+ adjoint_differential_shortest_geodesic_startpoint(
                M, Bt[k].pts[2], Bt[k].pts[1], 2.0, grad_B[k - 1].pts[end - 1]
            )
        # update p - effect from left and right segment as well as from c1 cond
        X3 =
            grad_B[k].pts[1] .+ adjoint_differential_shortest_geodesic_endpoint(
                M, Bt[k - 1].pts[m - 1], Bt[k].pts[1], 2.0, grad_B[k].pts[2]
            )
        # store
        grad_B[k - 1].pts[end - 1] .= X1
        grad_B[k].pts[2] .= X2
        grad_B[k].pts[1] .= X3
        grad_B[k - 1].pts[end] .= X3
    end
    return grad_B
end

@doc raw"""
    grad_distance(M,y,x[, p=2])
    grad_distance!(M,X,y,x[, p=2])

compute the (sub)gradient of the distance (squared), in place of `X`.

```math
f(x) = \frac{1}{p} d^p_{\mathcal M}(x,y)
```

to a fixed point `y` on the manifold `M` and `p` is an
integer. The gradient reads

```math
  \operatorname{grad}f(x) = -d_{\mathcal M}^{p-2}(x,y)\log_xy
```

for ``p\neq 1`` or ``x\neq  y``. Note that for the remaining case ``p=1``,
``x=y`` the function is not differentiable. In this case, the function returns the
corresponding zero tangent vector, since this is an element of the subdifferential.

# Optional

* `p` – (`2`) the exponent of the distance,  i.e. the default is the squared
  distance
"""
function grad_distance(M, y, x, p::Int=2)
    return (p == 2) ? -log(M, x, y) : -distance(M, x, y)^(p - 2) * log(M, x, y)
end
function grad_distance!(M, X, y, x, p::Int=2)
    log!(M, X, x, y)
    X .*= (p == 2) ? -one(eltype(X)) : -distance(M, x, y)(p - 2)
    return X
end

@doc raw"""
    grad_u,⁠ grad_v = grad_intrinsic_infimal_convolution_TV12(M, f, u, v, α, β)

compute (sub)gradient of the intrinsic infimal convolution model using the mid point
model of second order differences, see [`costTV2`](@ref), i.e. for some ``f ∈ \mathcal M``
on a `PowerManifold` manifold ``\mathcal M`` this function computes the (sub)gradient of

```math
E(u,v) =
\frac{1}{2}\sum_{i ∈ \mathcal G} d_{\mathcal M}(g(\frac{1}{2},v_i,w_i),f_i)
+ \alpha
\bigl(
β\mathrm{TV}(v) + (1-β)\mathrm{TV}_2(w)
\bigr),
```
where both total variations refer to the intrinsic ones, [`grad_TV`](@ref) and
[`grad_TV2`](@ref), respectively.
"""
function grad_intrinsic_infimal_convolution_TV12(M::AbstractManifold, f, u, v, α, β)
    c = mid_point(M, u, v, f)
    iL = log(M, c, f)
    return adjoint_differential_shortest_geodesic_startpoint(M, u, v, 1 / 2, iL) +
           α * β * grad_TV(M, u),
    adjoint_differential_shortest_geodesic_endpoint(M, u, v, 1 / 2, iL) +
    α * (1 - β) * grad_TV2(M, v)
end
@doc raw"""
    X = grad_TV(M, (x,y)[, p=1])
    grad_TV!(M, X, (x,y)[, p=1])

compute the (sub) gradient of ``\frac{1}{p}d^p_{\mathcal M}(x,y)`` with respect
to both ``x`` and ``y`` (in place of `X` and `Y`).
"""
function grad_TV(M::AbstractManifold, q::Tuple{T,T}, p=1) where {T}
    if p == 2
        return (-log(M, q[1], q[2]), -log(M, q[2], q[1]))
    else
        d = distance(M, q[1], q[2])
        if d == 0 # subdifferential containing zero
            return (zero_vector(M, q[1]), zero_vector(M, q[2]))
        else
            return (-log(M, q[1], q[2]) / (d^(2 - p)), -log(M, q[2], q[1]) / (d^(2 - p)))
        end
    end
end
function grad_TV!(M::AbstractManifold, X, q::Tuple{T,T}, p=1) where {T}
    d = distance(M, q[1], q[2])
    if d == 0 # subdifferential containing zero
        zero_vector!(M, X[1], q[1])
        zero_vector!(M, X[2], q[2])
        return X
    end
    log!(M, X[1], q[1], q[2])
    log!(M, X[2], q[2], q[1])
    if p == 2
        X[1] .*= -1
        X[2] .*= -1
    else
        X[1] .*= -1 / (d^(2 - p))
        X[2] .*= -1 / (d^(2 - p))
    end
    return X
end
@doc raw"""
    X = grad_TV(M, λ, x[, p=1])
    grad_TV!(M, X, λ, x[, p=1])

Compute the (sub)gradient ``\partial F`` of all forward differences occurring,
in the power manifold array, i.e. of the function

```math
F(x) = \sum_{i}\sum_{j ∈ \mathcal I_i} d^p(x_i,x_j)
```

where ``i`` runs over all indices of the `PowerManifold` manifold `M`
and ``\mathcal I_i`` denotes the forward neighbors of ``i``.

# Input
* `M` – a `PowerManifold` manifold
* `x` – a point.

# Ouput
* X – resulting tangent vector in ``T_x\mathcal M``. The computation can also be done in place.
"""
function grad_TV(M::PowerManifold, x, p::Int=1)
    power_size = power_dimensions(M)
    rep_size = representation_size(M.manifold)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R)
    X = zero_vector(M, x)
    c = costTV(M, x, p, 0)
    for i in R # iterate over all pixel
        di = 0.0
        for k in 1:d # for all direction combinations
            ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
            j = i + ek # compute neighbor
            if all(map(<=, j.I, maxInd.I)) # is this neighbor in range?
                if p != 1
                    g = (c[i] == 0 ? 1 : 1 / c[i]) .* grad_TV(M.manifold, (x[i], x[j]), p) # Compute TV on these
                else
                    g = grad_TV(M.manifold, (x[i], x[j]), p) # Compute TV on these
                end
                X[i] += g[1]
                X[j] += g[2]
            end
        end # directions
    end # i in R
    return X
end
function grad_TV!(M::PowerManifold, X, x, p::Int=1)
    power_size = power_dimensions(M)
    rep_size = representation_size(M.manifold)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R)
    c = costTV(M, x, p, 0)
    g = [zero_vector(M.manifold, x[first(R)]), zero_vector(M.manifold, x[first(R)])]
    for i in R # iterate over all pixel
        di = 0.0
        for k in 1:d # for all direction combinations
            ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
            j = i + ek # compute neighbor
            if all(map(<=, j.I, maxInd.I)) # is this neighbor in range?
                grad_TV!(M.manifold, g, (x[i], x[j]), p) # Compute TV on these
                if p != 1
                    (c[i] != 0) && (g[1] .*= 1 / c[i])
                    (c[i] != 0) && (g[2] .*= 1 / c[i])
                end
                X[i] += g[1]
                X[j] += g[2]
            end
        end # directions
    end # i in R
    return X
end

@doc raw"""
    Y = forward_logs(M,x)
    forward_logs!(M, Y, x)

compute the forward logs ``F`` (generalizing forward differences) occurring,
in the power manifold array, the function

```math
F_i(x) = \sum_{j ∈ \mathcal I_i} \log_{x_i} x_j,\quad i  ∈  \mathcal G,
```

where ``\mathcal G`` is the set of indices of the `PowerManifold` manifold `M` and
``\mathcal I_i`` denotes the forward neighbors of ``i``. This can also be done in place of `ξ`.

# Input
* `M` – a `PowerManifold` manifold
* `x` – a point.

# Ouput
* `Y` – resulting tangent vector in ``T_x\mathcal M`` representing the logs, where
  ``\mathcal N`` is thw power manifold with the number of dimensions added to `size(x)`.
  The computation can be done in place of `Y`.
"""
function forward_logs(M::PowerManifold{𝔽,TM,TSize,TPR}, p) where {𝔽,TM,TSize,TPR}
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    sX = size(p)
    maxInd = last(R).I
    if d > 1
        d2 = fill(1, d + 1)
        d2[d + 1] = d
    else
        d2 = 1
    end
    sN = d > 1 ? [power_size..., d] : [power_size...]
    N = PowerManifold(M.manifold, TPR(), sN...)
    xT = repeat(p; inner=d2)
    X = zero_vector(N, xT)
    e_k_vals = [1 * (1:d .== k) for k in 1:d]
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = i.I
            J = I .+ 1 .* e_k_vals[k] #i + e_k is j
            if all(J .<= maxInd) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neigbbor index as Cartesian Index
                X[N, i.I..., k] = log(M.manifold, p[M, i.I...], p[M, j.I...])
            end
        end # directions
    end # i in R
    return X
end
function forward_logs!(M::PowerManifold{𝔽,TM,TSize,TPR}, X, p) where {𝔽,TM,TSize,TPR}
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    sX = size(p)
    maxInd = last(R).I
    if d > 1
        d2 = fill(1, d + 1)
        d2[d + 1] = d
    else
        d2 = 1
    end
    sN = d > 1 ? [power_size..., d] : [power_size...]
    N = PowerManifold(M.manifold, TPR(), sN...)
    e_k_vals = [1 * (1:d .== k) for k in 1:d]
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = i.I
            J = I .+ 1 .* e_k_vals[k] #i + e_k is j
            if all(J .<= maxInd) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neigbbor index as Cartesian Index
                X[N, i.I..., k] = log(M.manifold, p[M, i.I...], p[M, j.I...])
            end
        end # directions
    end # i in R
    return X
end

@doc raw"""
    Y = grad_TV2(M, q[, p=1])
    grad_TV2!(M, Y, q[, p=1])

computes the (sub) gradient of ``\frac{1}{p}d_2^p(q_1, q_2, q_3)`` with respect
to all three components of ``q∈\mathcal M^3``, where ``d_2`` denotes the second order
absolute difference using the mid point model, i.e. let

```math
\mathcal C = \bigl\{ c ∈ \mathcal M \ |\ g(\tfrac{1}{2};q_1,q_3) \text{ for some geodesic }g\bigr\}
```
denote the mid points between ``q_1`` and ``q_3`` on the manifold ``\mathcal M``.
Then the absolute second order difference is defined as

```math
d_2(q_1,q_2,q_3) = \min_{c ∈ \mathcal C_{q_1,q_3}} d(c, q_2).
```

While the (sub)gradient with respect to ``q_2`` is easy, the other two require
the evaluation of an [`adjoint_Jacobi_field`](@ref).
"""
function grad_TV2(M::AbstractManifold, q, p::Int=1)
    X = [zero_vector(M, x) for x in q]
    return grad_TV2!(M, X, q, p)
end
function grad_TV2!(M::AbstractManifold, X, q, p::Int=1)
    c = mid_point(M, q[1], q[3], q[2]) # nearest mid point of x and z to y
    d = distance(M, q[2], c)
    innerLog = -log(M, c, q[2])
    if p == 2
        X[1] .= adjoint_differential_shortest_geodesic_startpoint(
            M, q[1], q[3], 1 / 2, innerLog
        )
        log!(M, X[2], q[2], c)
        X[2] .*= -1
        X[3] .= adjoint_differential_shortest_geodesic_endpoint(
            M, q[1], q[3], 1 / 2, innerLog
        )
    else
        if d == 0 # subdifferential containing zero
            for i in 1:3
                zero_vector!(M, X[i], q[i])
            end
        else
            X[1] .= adjoint_differential_shortest_geodesic_startpoint(
                M, q[1], q[3], 1 / 2, innerLog / (d^(2 - p))
            )
            log!(M, X[2], q[2], c)
            X[2] .*= -1 / (d^(2 - p))
            X[3] .= adjoint_differential_shortest_geodesic_endpoint(
                M, q[1], q[3], 1 / 2, innerLog / (d^(2 - p))
            )
        end
    end
    return X
end
@doc raw"""
    grad_TV2(M::PowerManifold, q[, p=1])

computes the (sub) gradient of ``\frac{1}{p}d_2^p(q_1,q_2,q_3)``
with respect to all ``q_1,q_2,q_3`` occurring along any array dimension in the
point `q`, where `M` is the corresponding `PowerManifold`.
"""
function grad_TV2(M::PowerManifold, q, p::Int=1)
    X = zero_vector(M, q)
    return grad_TV2!(M, X, q, p)
end
function grad_TV2!(M::PowerManifold, X, q, p::Int=1)
    power_size = power_dimensions(M)
    rep_size = representation_size(M.manifold)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    minInd, maxInd = first(R), last(R)
    c = costTV2(M, q, p, false)
    for i in R # iterate over all pixel
        di = 0.0
        for k in 1:d # for all direction combinations (TODO)
            ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
            jF = i + ek # compute forward neighbor
            jB = i - ek # compute backward neighbor
            if all(map(<=, jF.I, maxInd.I)) && all(map(>=, jB.I, minInd.I)) # are neighbors in range?
                if p != 1
                    g =
                        (c[i] == 0 ? 1 : 1 / c[i]) .*
                        grad_TV2(M.manifold, (q[jB], q[i], q[jF]), p) # Compute TV2 on these
                else
                    g = grad_TV2(M.manifold, (q[jB], q[i], q[jF]), p) # Compute TV2 on these
                end
                X[M, jB.I...] = g[1]
                X[M, i.I...] = g[2]
                X[M, jF.I...] = g[3]
            end
        end # directions
    end # i in R
    return X
end
