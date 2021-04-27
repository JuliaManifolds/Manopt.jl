@doc raw"""
    y = prox_distance(M,λ,f,x [, p=2])
    prox_distance!(M, y, λ, f, x [, p=2])

compute the proximal map ``\operatorname{prox}_{λ\varphi}`` with
parameter λ of ``φ(x) = \frac{1}{p}d_{\mathcal M}^p(f,x)``.
For the mutating variant the computation is done in place of `y`.

# Input
* `M` – a [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) ``\mathcal M``
* `λ` – the prox parameter
* `f` – a point ``f ∈ \mathcal M`` (the data)
* `x` – the argument of the proximal map

# Optional argument
* `p` – (`2`) exponent of the distance.

# Ouput
* `y` – the result of the proximal map of ``φ``
"""
function prox_distance(M::Manifold, λ, f, x, p::Int=2)
    d = distance(M, f, x)
    if p == 2
        t = λ / (1 + λ)
    elseif p == 1
        t = (λ < d) ? λ / d : 1.0
    else
        throw(
            ErrorException(
                "Proximal Map of distance(M,f,x) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
    return exp(M, x, log(M, x, f), t)
end
function prox_distance!(M::Manifold, y, λ, f, x, p::Int=2)
    d = distance(M, f, x)
    if p == 2
        t = λ / (1 + λ)
    elseif p == 1
        t = (λ < d) ? λ / d : 1.0
    else
        throw(
            ErrorException(
                "Proximal Map of distance(M,f,x) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
    return exp!(M, y, x, log(M, x, f), t)
end

@doc raw"""
    [y1,y2] = prox_TV(M, λ, [x1,x2] [,p=1])
    prox_TV!(M, [y1,y2] λ, [x1,x2] [,p=1])

Compute the proximal map ``\operatorname{prox}_{λ\varphi}`` of
``φ(x,y) = d_{\mathcal M}^p(x,y)`` with
parameter `λ`.

# Input

* `M` – a [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)
* `λ` – a real value, parameter of the proximal map
* `(x1,x2)` – a tuple of two points,

# Optional

(default is given in brackets)
* `p` – (1) exponent of the distance of the TV term

# Ouput

* `(y1,y2)` – resulting tuple of points of the ``\operatorname{prox}_{λφ}(```(x1,x2)```)``.
  The result can also be computed in place.
"""
function prox_TV(M::Manifold, λ::Number, x::Tuple{T,T}, p::Int=1) where {T}
    d = distance(M, x[1], x[2])
    if p == 1
        t = min(0.5, λ / d)
    elseif p == 2
        t = λ / (1 + 2 * λ)
    else
        throw(
            ErrorException(
                "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
    return (exp(M, x[1], log(M, x[1], x[2]), t), exp(M, x[2], log(M, x[2], x[1]), t))
end
function prox_TV!(M::Manifold, y, λ::Number, x::Tuple{T,T}, p::Int=1) where {T}
    d = distance(M, x[1], x[2])
    if p == 1
        t = min(0.5, λ / d)
    elseif p == 2
        t = λ / (1 + 2 * λ)
    else
        throw(
            ErrorException(
                "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
    X1 = log(M, x[1], x[2])
    X2 = log(M, x[2], x[1])
    exp!(M, y[1], x[1], X1, t)
    exp!(M, y[2], x[2], X2, t)
    return y
end
@doc raw"""
    ξ = prox_TV(M,λ,x [,p=1])

compute the proximal maps ``\operatorname{prox}_{λ\varphi}`` of
all forward differences occurring in the power manifold array, i.e.
``\varphi(xi,xj) = d_{\mathcal M}^p(xi,xj)`` with `xi` and `xj` are array
elemets of `x` and `j = i+e_k`, where `e_k` is the ``k``th unitvector.
The parameter `λ` is the prox parameter.

# Input
* `M` – a [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)
* `λ` – a real value, parameter of the proximal map
* `x` – a point.

# Optional
(default is given in brackets)
* `p` – (1) exponent of the distance of the TV term

# Ouput
* `y` – resulting  point containing with all mentioned proximal
  points evaluated (in a cyclic order). The computation can also be done in place
"""
function prox_TV(M::PowerManifold, λ, x, p::Int=1)
    y = deepcopy(x)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R).I
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1
            for i in R # iterate over all pixel
                if (i[k] % 2) == l
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neigbbor index as Cartesian Index
                        (y[i], y[j]) = prox_TV(M.manifold, λ, (y[i], y[j]), p) # Compute TV on these
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end
function prox_TV!(M::PowerManifold, y, λ, x, p::Int=1)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    copyto!(M, y, x)
    maxInd = last(R).I
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1
            for i in R # iterate over all pixel
                if (i[k] % 2) == l # even/odd splitting
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neigbbor index as Cartesian Index
                        prox_TV!(M.manifold, [y[i], y[j]], λ, (y[i], y[j]), p) # Compute TV on these
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end
@doc raw"""
    y = prox_parallel_TV(M, λ, x [,p=1])
    prox_parallel_TV!(M, y, λ, x [,p=1])

compute the proximal maps ``\operatorname{prox}_{λφ}`` of
all forward differences occurring in the power manifold array, i.e.
``φ(x_i,x_j) = d_{\mathcal M}^p(x_i,x_j)`` with `xi` and `xj` are array
elements of `x` and `j = i+e_k`, where `e_k` is the ``k``th unit vector.
The parameter `λ` is the prox parameter.

# Input
* `M`     – a `PowerManifold` manifold
* `λ`     – a real value, parameter of the proximal map
* `x`     – a point

# Optional
(default is given in brackets)
* `p` – (`1`) exponent of the distance of the TV term

# Ouput
* `y`  – resulting Array of points with all mentioned proximal
  points evaluated (in a parallel within the arrays elements).
  The computation can also be done in place.

*See also* [`prox_TV`](@ref)
"""
function prox_parallel_TV(M::PowerManifold, λ, x::AbstractVector, p::Int=1)
    R = CartesianIndices(x[1])
    d = ndims(x[1])
    if length(x) != 2 * d
        throw(
            ErrorException(
                "The number of inputs from the array ($(length(x))) has to be twice the data dimensions ($(d)).",
            ),
        )
    end
    maxInd = Tuple(last(R))
    # create an array for even/odd splitted proxes along every dimension
    y = deepcopy(x)
    yV = reshape(y, d, 2)
    xV = reshape(x, d, 2)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1 # even odd
            for i in R # iterate over all pixel
                if (i[k] % 2) == l
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neigbbor index as Cartesian Index
                        # parallel means we apply each (direction even/odd) to a seperate copy of the data.
                        (yV[k, l + 1][i], yV[k, l + 1][j]) = prox_TV(
                            M.manifold, λ, (xV[k, l + 1][i], xV[k, l + 1][j]), p
                        ) # Compute TV on these
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end
function prox_parallel_TV!(
    M::PowerManifold, y::AbstractVector, λ, x::AbstractVector, p::Int=1
)
    R = CartesianIndices(x[1])
    d = ndims(x[1])
    if length(x) != 2 * d
        throw(
            ErrorException(
                "The number of inputs from the array ($(length(x))) has to be twice the data dimensions ($(d)).",
            ),
        )
    end
    maxInd = Tuple(last(R))
    # init y
    for i in 1:length(x)
        copyto!(M, y[i], x[i])
    end
    yV = reshape(y, d, 2)
    xV = reshape(x, d, 2)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1 # even odd
            for i in R # iterate over all pixel
                if (i[k] % 2) == l
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neigbbor index as Cartesian Index
                        # parallel means we apply each (direction even/odd) to a seperate copy of the data.
                        prox_TV!(
                            M.manifold,
                            [yV[k, l + 1][i], yV[k, l + 1][j]],
                            λ,
                            (xV[k, l + 1][i], xV[k, l + 1][j]),
                            p,
                        ) # Compute TV on these in place of y
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end

@doc raw"""
    (y1,y2,y3) = prox_TV2(M,λ,(x1,x2,x3),[p=1], kwargs...)
    prox_TV2!(M, y, λ,(x1,x2,x3),[p=1], kwargs...)

Compute the proximal map ``\operatorname{prox}_{λ\varphi}`` of
``\varphi(x_1,x_2,x_3) = d_{\mathcal M}^p(c(x_1,x_3),x_2)`` with
parameter `λ`>0, where ``c(x,z)`` denotes the mid point of a shortest
geodesic from `x1` to `x3` that is closest to `x2`.
The result can be computed in place of `y`.

# Input

* `M`          – a manifold
* `λ`          – a real value, parameter of the proximal map
* `(x1,x2,x3)` – a tuple of three points

* `p` – (`1`) exponent of the distance of the TV term

# Optional
`kwargs...` – parameters for the internal [`subgradient_method`](@ref)
    (if `M` is neither `Euclidean` nor `Circle`, since for these a closed form
    is given)

# Output
* `(y1,y2,y3)` – resulting tuple of points of the proximal map.
  The computation can also be done in place.
"""
function prox_TV2(
    M::Manifold,
    λ,
    x::Tuple{T,T,T},
    p::Int=1;
    stopping_criterion::StoppingCriterion=StopAfterIteration(5),
    kwargs...,
) where {T}
    (p != 1) && throw(
        ErrorException(
            "Proximal Map of TV2(M, λ, x, p) not implemented for p=$(p) (requires p=1) on general manifolds.",
        ),
    )
    PowX = SVector(x)
    PowM = PowerManifold(M, NestedPowerRepresentation(), 3)
    xR = PowX
    F(M, x) = 1 / 2 * distance(M, PowX, x)^2 + λ * costTV2(M, x)
    ∂F(PowM, x) = log(PowM, x, PowX) + λ * grad_TV2(PowM, x)
    subgradient_method!(PowM, F, ∂F, xR; stopping_criterion=stopping_criterion, kwargs...)
    return (xR...,)
end
function prox_TV2!(
    M::Manifold,
    y,
    λ,
    x::Tuple{T,T,T},
    p::Int=1;
    stopping_criterion::StoppingCriterion=StopAfterIteration(5),
    kwargs...,
) where {T}
    (p != 1) && throw(
        ErrorException(
            "Proximal Map of TV2(M, λ, x, p) not implemented for p=$(p) (requires p=1) on general manifolds.",
        ),
    )
    PowX = SVector(x)
    PowM = PowerManifold(M, NestedPowerRepresentation(), 3)
    copyto!(M, y, PowX)
    F(M, x) = 1 / 2 * distance(M, PowX, x)^2 + λ * costTV2(M, x)
    ∂F!(M, y, x) = log!(M, y, x, PowX) + λ * grad_TV2!(M, y, x)
    subgradient_method!(
        PowM,
        F,
        ∂F!,
        y;
        stopping_criterion=stopping_criterion,
        evaluation=MutatingEvaluation(),
        kwargs...,
    )
    return y
end
@doc raw"""
    y = prox_TV2(M, λ, x[, p=1])
    prox_TV2!(M, y, λ, x[, p=1])

compute the proximal maps ``\operatorname{prox}_{λ\varphi}`` of
all centered second order differences occuring in the power manifold array, i.e.
``\varphi(x_k,x_i,x_j) = d_2(x_k,x_i.x_j)``, where ``k,j`` are backward and forward
neighbors (along any dimension in the array of `x`).
The parameter `λ` is the prox parameter.

# Input
* `M` – a [Manifold](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold)
* `λ` – a real value, parameter of the proximal map
* `x` – a points.

# Optional
(default is given in brackets)
* `p` – (`1`) exponent of the distance of the TV term

# Output
* `y` – resulting point with all mentioned proximal points evaluated (in a cylic order).
  The computation can also be done in place.
"""
function prox_TV2(M::PowerManifold{N,T}, λ, x, p::Int=1) where {N,T}
    y = deepcopy(x)
    return prox_TV2!(M, y, λ, x, p)
end
function prox_TV2!(M::PowerManifold{N,T}, y, λ, x, p::Int=1) where {N,T}
    power_size = power_dimensions(M)
    R = CartesianIndices(power_size)
    d = length(size(x))
    minInd = first(R).I
    maxInd = last(R).I
    copyto!(M, y, x)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:2
            for i in R # iterate over all pixel
                if (i[k] % 3) == l
                    JForward = i.I .+ ek.I #i + e_k
                    JBackward = i.I .- ek.I # i - e_k
                    all(JForward .<= maxInd) &&
                        all(JBackward .>= minInd) &&
                        prox_TV2!(
                            M.manifold,
                            [y[M, JBackward...], y[M, i.I...], y[M, JForward...]],
                            λ,
                            (y[M, JBackward...], y[M, i.I...], y[M, JForward...]),
                            p,
                        )
                end # if mod 3
            end # i in R
        end # for mod 3
    end # directions
    return y
end
@doc raw"""
    project_collaborative_TV(M, λ, x, Ξ[, p=2,q=1])
    project_collaborative_TV!(M, Θ, λ, x, Ξ[, p=2,q=1])

compute the projection onto collaborative Norm unit (or α-) ball, i.e. of the function

```math
F^q(x) = \sum_{i∈\mathcal G}
  \Bigl( \sum_{j∈\mathcal I_i}
    \sum_{k=1^d} \lVert X_{i,j}\rVert_x^p\Bigr)^\frac{q/p},
```

where ``\mathcal G`` is the set of indices for ``x∈\mathcal M`` and ``\mathcal I_i``
is the set of its forward neighbors.
The computation can also be done in place of `Θ`.

This is adopted from the paper by Duran, Möller, Sbert, Cremers:
_Collaborative Total Variation: A General Framework for Vectorial TV Models_
(arxiv: [1508.01308](https://arxiv.org/abs/1508.01308)), where the most inner
norm is not on a manifold but on a vector space, see their Example 3 for
details.
"""
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p=2.0, q=1.0, α=1.0)
    pdims = power_dimensions(N)
    if length(pdims) == 1
        d = 1
        s = 1
        iRep = (1,)
    else
        d = pdims[end]
        s = length(pdims) - 1
        if s != d
            throw(
                ErrorException(
                    "the last dimension ($(d)) has to be equal to the number of the previous ones ($(s)) but its not.",
                ),
            )
        end
        iRep = (Integer.(ones(d))..., d)
    end
    if q == 1 # Example 3 case 2
        if p == 1
            normΞ = norm.(Ref(N.manifold), x, Ξ)
            return max.(normΞ .- λ, 0.0) ./ ((normΞ .== 0) .+ normΞ) .* Ξ
        end
        if p == 2 # Example 3 case 3
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims=d + 1))
            if length(iRep) > 1
                norms = repeat(norms; inner=iRep)
            end
            # if the norm is zero add 1 to avoid division by zero, also then the
            # nominator is already (max(-λ,0) = 0) so it stays zero then
            return max.(norms .- λ, 0.0) ./ ((norms .== 0) .+ norms) .* Ξ
        end
        throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
    elseif q == Inf
        if p == 2
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims=d + 1))
            if length(iRep) > 1
                norms = repeat(norms; inner=iRep)
            end
        elseif p == 1
            norms = sum(norm.(Ref(N.manifold), x, Ξ); dims=d + 1)
            if length(iRep) > 1
                norms = repeat(norms; inner=iRep)
            end
        elseif p == Inf
            norms = norm.(Ref(N.manifold), x, Ξ)
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
        return (α .* Ξ) ./ max.(Ref(α), norms)
    end # end q
    return throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
end
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p::Int, q::Float64=1.0, α=1.0)
    return project_collaborative_TV(N, λ, x, Ξ, Float64(p), q, α)
end
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p::Float64, q::Int, α=1.0)
    return project_collaborative_TV(N, λ, x, Ξ, p, Float64(q), α)
end
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p::Int, q::Int, α=1.0)
    return project_collaborative_TV(N, λ, x, Ξ, Float64(p), Float64(q), α)
end

function project_collaborative_TV!(N::PowerManifold, Θ, λ, x, Ξ, p=2.0, q=1.0, α=1.0)
    pdims = power_dimensions(N)
    if length(pdims) == 1
        d = 1
        s = 1
        iRep = (1,)
    else
        d = pdims[end]
        s = length(pdims) - 1
        if s != d
            throw(
                ErrorException(
                    "the last dimension ($d) has to be equal to the number of the previous ones ($s) but its not.",
                ),
            )
        end
        iRep = (Integer.(ones(d))..., d)
    end
    if q == 1 # Example 3 case 2
        if p == 1
            normΞ = norm.(Ref(N.manifold), x, Ξ)
            Θ .= max.(normΞ .- λ, 0.0) ./ ((normΞ .== 0) .+ normΞ) .* Ξ
            return Θ
        elseif p == 2 # Example 3 case 3
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims=d + 1))
            if length(iRep) > 1
                norms = repeat(norms; inner=iRep)
            end
            # if the norm is zero add 1 to avoid division by zero, also then the
            # nominator is already (max(-λ,0) = 0) so it stays zero then
            Θ .= max.(norms .- λ, 0.0) ./ ((norms .== 0) .+ norms) .* Ξ
            return Θ
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
    elseif q == Inf
        if p == 2
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims=d + 1))
            (length(iRep) > 1) && (norms = repeat(norms; inner=iRep))
        elseif p == 1
            norms = sum(norm.(Ref(N.manifold), x, Ξ); dims=d + 1)
            (length(iRep) > 1) && (norms = repeat(norms; inner=iRep))
        elseif p == Inf
            norms = norm.(Ref(N.manifold), x, Ξ)
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
        Θ .= (α .* Ξ) ./ max.(Ref(α), norms)
        return Θ
    end # end q
    return throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
end
function project_collaborative_TV!(
    N::PowerManifold, Θ, λ, x, Ξ, p::Int, q::Float64=1.0, α=1.0
)
    return project_collaborative_TV!(N, Θ, λ, x, Ξ, Float64(p), q, α)
end
function project_collaborative_TV!(N::PowerManifold, Θ, λ, x, Ξ, p::Float64, q::Int, α=1.0)
    return project_collaborative_TV!(N, Θ, λ, x, Ξ, p, Float64(q), α)
end
function project_collaborative_TV!(N::PowerManifold, Θ, λ, x, Ξ, p::Int, q::Int, α=1.0)
    return project_collaborative_TV!(N, Θ, λ, x, Ξ, Float64(p), Float64(q), α)
end
