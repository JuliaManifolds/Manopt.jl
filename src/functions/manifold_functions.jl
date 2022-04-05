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
    w = @SVector [1.0, -2.0, 1.0]
    x = SVector(pointTuple)
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
@doc raw"""
    reflect(M, f, x)
    reflect!(M, q, f, x)

reflect the point `x` from the manifold `M` at the point `f(x)` of the
function ``f: \mathcal M → \mathcal M``, i.e.,

````math
    \operatorname{refl}_f(x) = \operatorname{refl}_{f(x)}(x),
````

Compute the result in `q`.

see also [`reflect`](@ref reflect(M::AbstractManifold, p, x))`(M,p,x)`.
"""
reflect(M::AbstractManifold, pr::Function, x) = reflect(M, pr(x), x)
reflect!(M::AbstractManifold, q, pr::Function, x) = reflect!(M, q, pr(x), x)

@doc raw"""
    reflect(M, p, x)
    reflect!(M, q, p, x)
reflect the point `x` from the manifold `M` at point `p`, i.e.

````math
    \operatorname{refl}_p(x) = \exp_p(-\log_p x).
````

where exp and log denote the exponential and logarithmic map on `M`.
This can also be done in place of `q`.
"""
reflect(M::AbstractManifold, p, x) = exp(M, p, -log(M, p, x))
reflect!(M::AbstractManifold, q, p, x) = exp!(M, q, p, -log(M, p, x))
