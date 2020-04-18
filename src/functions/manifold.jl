    """
        mid_point(M, p, q, x)

    Compute the mid point between p and q. If there is more than one mid point
    of (not neccessarily miniizing) geodesics (i.e. on the sphere), the one nearest
    to x is returned.
    """
    mid_point(M::Manifold, p, q, x) = mid_point(M, p, q)
    mid_point!(M::Manifold, y, p, q, x) = mid_point!(M, y, p, q)

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

    """
        mid_point(M, p, q)

    Compute the (geodesic) mid point of the two points `p` and `q` on the
    manfold `M`. If the geodesic is not unique, either a deterministic choice is taken or
    an error is raised depending on the manifold. For the deteministic choixe, see
    [`mid_point(M, p, q, x)`](@ref), the mid point closest to a third point
    `x`.
    """
    mid_point(M::Manifold, p, q) = exp(M, p, log(M, p, q), 0.5)
    mid_point!(M::Manifold, y, p, q) = exp!(M, y, p, log(M, p, q), 0.5)

    @doc raw"""
        reflect(M, f, x)

    reflect the point `x` from the manifold `M` at the point `f(x)` of the
    function $f\colon \mathcal M \to \mathcal M$.
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