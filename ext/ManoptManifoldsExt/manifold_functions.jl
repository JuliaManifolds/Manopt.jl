
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
the choice of typical distance in Matlab Manopt, i.e. dimension of `M`. See
[this note](https://github.com/NicolasBoumal/manopt/blob/97b6eb6b185334ab7b3991585ed2c044d69ee905/manopt/manifolds/fixedrank/fixedrankembeddedfactory.m#L76-L78)
"""
function max_stepsize(M::FixedRankMatrices, p)
    return manifold_dimension(M)
end

"""
    mid_point(M, p, q, x)
    mid_point!(M, y, p, q, x)

Compute the mid point between `p` and `q`. If there is more than one mid point
of (not neccessarily minimizing) geodesics (e.g. on the sphere), the one nearest
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

function prox_TV2(::Euclidean, λ, pointTuple::Tuple{T,T,T}, p::Int=1) where {T}
    w = [1.0, -2.0, 1.0]
    x = [pointTuple...]
    if p == 1 # Example 3.2 in Bergmann, Laus, Steidl, Weinmann, 2014.
        m = min.(Ref(λ), abs.(x .* w) / (dot(w, w)))
        s = sign.(sum(x .* w))
        return x .- m .* s .* w
    elseif p == 2 # Theorem 3.6 ibd.
        t = λ * sum(x .* w) / (1 + λ * dot(w, w))
        return x .- t .* w
    else
        throw(
            ErrorException(
                "Proximal Map of TV2(Euclidean,λ,pT,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
end
