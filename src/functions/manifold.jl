"""
    mid_point(M, p, q, x)

Compute the mid point between p and q. If there is more than one mid point
of (not neccessarily minimizing) geodesics (i.e. on the sphere), the one nearest
to x is returned.
"""
mid_point(M::Manifold, p, q, ::Any) = mid_point(M, p, q)
mid_point!(M::Manifold, y, p, q, ::Any) = mid_point!(M, y, p, q)

function mid_point(M::Circle, p, q)
    return exp(M,p,0.5*log(M,p,q))
end
function mid_point(M::Circle, p, q, x)
    if distance(M,p,q) ≈ π
        X = 0.5*log(M,p,q)
        Y = log(p,x)
        return exp(M, p, (sign(X) == sign(Y) ? 1 : -1)*X)
    end
    return mid_point(M,p,q)
end

function mid_point(M::Sphere, p, q, x)
    if isapprox(M,p,-q)
        X = log(M,p,x)/distance(M,p,x)*π
    else
        X = log(M,p,q)
    end
    return exp(M,p,0.5*X)
end
function mid_point!(M::Sphere, y, p, q, x)
    if isapprox(M,p,-q)
        X = log(M,p,x)/distance(M,p,x)*π
    else
        X = log(M,p,q)
    end
    y .= exp(M,p,0.5*X)
    return y
end

@doc raw"""
    reflect(M, f, x)

reflect the point `x` from the manifold `M` at the point `f(x)` of the
function $f\colon \mathcal M \to \mathcal M$, i.e.,

````math
    \operatorname{refl}_f(x) = \operatorname{refl}_{f(x)}(x),
````
see also [`reflect`](@ref reflect(M::Manifold, p, x))`(M,p,x)`.
"""

reflect(M::Manifold, pr::Function, x) = reflect(M::Manifold, pr(x), x)

@doc raw"""
    reflect(M, p, x)

reflect the point `x` from the manifold `M` at point `x`, i.e.

````math
    \operatorname{refl}_p(x) = \exp_p(-\log_p x).
````
where exp and log denote the exponential and logarithmic map on `M`.
"""
reflect(M::Manifold, p, x) = exp(M, p, -log(M, p, x))

@doc raw"""
    sym_rem(x,[T=π])

Compute symmetric remainder of `x` with respect to the interall 2*`T`, i.e.
`(x+T)%2T`, where the default for `T` is $π$
"""
function sym_rem(x::N, T = π) where {N<:Number}
    return (x ≈ T ? convert(N, -T) : rem(x, convert(N, 2 * T), RoundNearest))
end
