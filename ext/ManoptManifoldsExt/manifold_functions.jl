
"""
    max_stepsize(M::TangentBundle, p)

Tangent bundle has injectivity radius of either infinity (for flat manifolds) or 0
(for non-flat manifolds). This makes a guess of what a reasonable maximum stepsize
on a tangent bundle might be.
"""
function max_stepsize(M::TangentBundle, p)
    return max_stepsize(M.manifold, p[M, :point])
end

"""
    max_stepsize(M::FixedRankMatrices, p)

Return a reasonable guess of maximum step size on `FixedRankMatrices` following
the choice of typical distance in Matlab Manopt, the dimension of `M`. See
[this note](https://github.com/NicolasBoumal/manopt/blob/97b6eb6b185334ab7b3991585ed2c044d69ee905/manopt/manifolds/fixedrank/fixedrankembeddedfactory.m#L76-L78)
"""
function max_stepsize(M::FixedRankMatrices, p)
    return manifold_dimension(M)
end

"""
    mid_point(M, p, q, x)
    mid_point!(M, y, p, q, x)

Compute the mid point between `p` and `q`. If there is more than one mid point
of (not necessarily minimizing) geodesics (for example on the sphere), the one nearest
to `x` is returned (in place of `y`).
"""
mid_point(M::AbstractManifold, p, q, ::Any) = mid_point(M, p, q)
mid_point!(M::AbstractManifold, y, p, q, ::Any) = mid_point!(M, y, p, q)

function mid_point(M::Circle, p, q, x)
    if distance(M, p, q) ≈ π
        X = 0.5 * log(M, p, q)
        Y = log(M, p, x)
        return exp(M, p, (sign(X) == sign(Y) ? 1 : -1) * X)
    end
    return mid_point(M, p, q)
end

function mid_point(M::Sphere, p, q, x)
    if isapprox(M, p, -q)
        X = log(M, p, x) / distance(M, p, x) * π
    else
        X = log(M, p, q)
    end
    return exp(M, p, 0.5 * X)
end
function mid_point!(M::Sphere, y, p, q, x)
    if isapprox(M, p, -q)
        X = log(M, p, x) / distance(M, p, x) * π
    else
        X = log(M, p, q)
    end
    y .= exp(M, p, 0.5 * X)
    return y
end
