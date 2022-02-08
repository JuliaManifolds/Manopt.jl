@doc raw"
    random_point(M::AbstractManifold)

generate a random point on a manifold. By default it uses `random_point(M,:Gaussian)`.
"
random_point(M::AbstractManifold) = random_point(M, Val(:Gaussian))
@doc raw"
    random_point(M::AbstractManifold, s::Symbol, options...)

generate a random point using a noise model given by `s` with its additional `options`
just passed on.
"
function random_point(M::AbstractManifold, s::Symbol, options...)
    return random_point(M, Val(s), options...)
end

@doc raw"""
    random_point(M::AbstractPowerManifold, options...)

generate a random point on the `AbstractPowerManifold` `M` given `options` that are
passed on.
"""
function random_point(
    M::AbstractPowerManifold{𝔽,Mt,NestedPowerRepresentation}, options...
) where {𝔽,Mt}
    return [random_point(M.manifold, options...) for i in get_iterator(M)]
end
function random_point(
    M::AbstractPowerManifold{𝔽,Mt,ArrayPowerRepresentation}, options...
) where {𝔽,Mt}
    return cat(
        [random_point(M.manifold, options...) for i in get_iterator(M)]...;
        dims=length(representation_size(M.manifold)) + 1,
    )
end

@doc raw"""
    random_point(M::AbstractGroupManifold, options...)

On an abstract group manifold, the random point is taken from the internally stored `M.manifold`.
"""
random_point(M::AbstractGroupManifold, kwargs...) = random_point(M.manifold, kwargs...)
function random_point(M::AbstractGroupManifold, s::Symbol, options...)
    return random_point(M, Val(s), options...)
end

@doc raw"""
    random_point(M::Circle, :Uniform)

return a random point on the [`Circle`](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) ``\mathbb S^1`` by
picking a random element from ``[-\pi,\pi)`` uniformly.
"""
random_point(::Circle, ::Val{:Uniform}) = sym_rem(rand() * 2 * π)
random_point(M::Circle) = random_point(M, Val(:Uniform)) # introduce different default

@doc raw"""
    random_point(M::Euclidean[,:Gaussian, σ::Float64=1.0])

generate a random point on the `Euclidean` manifold `M`, where the
optional parameter determines the type of the entries of the
resulting point on the Euclidean space d.
"""
random_point(M::Euclidean) = randn(representation_size(M))
random_point(M::Euclidean, ::Val{:Gaussian}, σ=1.0) = σ * randn(manifold_dimension(M))

@doc raw"""
    random_point(M::FixedRankMatrices, options...)

return a random point on the FixedRankMatrices manifold.
The orthogonal matrices are sampled from the Stiefel manifold
and the singular values are sampled uniformly at random.
"""
function random_point(::FixedRankMatrices{m,n,k}, options...) where {m,n,k}
    U = random_point(Stiefel(m, k), options...)
    S = sort(rand(k); rev=true)
    V = random_point(Stiefel(n, k), options...)
    return SVDMPoint(U, S, V')
end

@doc raw"""
    random_point(M::Grassmannian, :Gaussian [, σ=1.0])

return a random point `x` on `Grassmannian` manifold `M` by
generating a random (Gaussian) matrix with standard deviation `σ` in matching
size, which is orthonormal.
"""
function random_point(::Grassmann{n,k,𝔽}, ::Val{:Gaussian}, σ::Float64=1.0) where {n,k,𝔽}
    V = σ * randn(𝔽 === ℝ ? Float64 : ComplexF64, (n, k))
    A = qr(V).Q[:, 1:k]
    return A
end

@doc raw"""
    random_point(M::ProductManifold, options...)

return a random point `x` on `Grassmannian` manifold `M` by
generating a random (Gaussian) matrix with standard deviation `σ` in matching
size, which is orthonormal.
"""
function random_point(M::ProductManifold, o...)
    return ProductRepr([random_point(N, o...) for N in M.manifolds]...)
end

@doc raw"""
    random_point(M::Rotations, :Gaussian [, σ=1.0])

return a random point `p` on the manifold `Rotations`
by generating a (Gaussian) random orthogonal matrix with determinant ``+1``. Let ``QR = A``
be the QR decomposition of a random matrix ``A``, then the formula reads ``p = QD``
where ``D`` is a diagonal matrix with the signs of the diagonal entries of ``R``, i.e.
````math
D_{ij}=\begin{cases}
\operatorname{sgn}(R_{ij}) & \text{if} \; i=j \\
0 & \, \text{otherwise.}
\end{cases}
````
It can happen that the matrix gets -1 as a determinant. In this case, the first
and second columns are swapped.
"""
function random_point(M::Rotations, ::Val{:Gaussian}, σ::Real=1.0)
    # Special case: Rotations(1) is just zero-dimensional
    (manifold_dimension(M) == 0) && return ones(Float64, 1, 1)
    A = randn(Float64, representation_size(M))
    s = diag(sign.(qr(A).R))
    D = Diagonal(s)
    C = qr(A).Q * D
    if det(C) < 0
        C[:, [1, 2]] = C[:, [2, 1]]
    end
    return C
end

@doc raw"""
    random_point(M::SymmetricPositiveDefinite, :Gaussian[, σ=1.0])

generate a random symmetric positive definite matrix on the
`SymmetricPositiveDefinite` manifold `M`.
"""
function random_point(
    ::SymmetricPositiveDefinite{N}, ::Val{:Gaussian}, σ::Float64=1.0
) where {N}
    D = Diagonal(1 .+ rand(N)) # random diagonal matrix
    s = qr(σ * randn(N, N)) # random q
    return Matrix(Symmetric(s.Q * D * transpose(s.Q)))
end

@doc raw"""
    random_point(M::Stiefel, :Gaussian[, σ=1.0])

return a random (Gaussian) point `x` on the `Stiefel` manifold `M` by generating a (Gaussian)
matrix with standard deviation `σ` and return the orthogonalized version, i.e. return ​​the Q
component of the QR decomposition of the random matrix of size ``n×k``.
"""
function random_point(::Stiefel{n,k,𝔽}, ::Val{:Gaussian}, σ::Float64=1.0) where {n,k,𝔽}
    A = σ * randn(𝔽 === ℝ ? Float64 : ComplexF64, n, k)
    return Matrix(qr(A).Q)
end

@doc raw"""
    random_point(M::Sphere, :Gaussian[, σ=1.0])
return a random point on the Sphere by projecting a normal distributed vector
from within the embedding to the sphere.
"""
function random_point(M::Sphere, ::Val{:Gaussian}, σ::Float64=1.0)
    return project(M, σ * randn(manifold_dimension(M) + 1))
end

@doc raw"""
    random_point(M::TangentBundle, options...)

generate a random point on the tangent bundle by calling a [`random_point`](@ref) and a
[`random_tangent`](@ref) with the given `options...`
"""
function random_point(M::TangentBundle, options...)
    p = random_point(M.manifold, options...)
    X = random_tangent(M.manifold, p, options...)
    return ProductRepr(p, X)
end

@doc raw"""
    random_tangent(M::AbstractGroupManifold, p, options...)

On an abstract group manifold, the random tangent is taken from the internally stored `M.manifold`s tangent space at `p`.
"""
function random_tangent(M::AbstractGroupManifold, p, kwargs...)
    return random_tangent(M.manifold, p, kwargs...)
end
function random_tangent(M::AbstractGroupManifold, p, s::Symbol, options...)
    return random_tangent(M, p, Val(s), options...)
end

@doc raw"""
    random_tangent(M, p, options...)

generate a random tangent vector in the tangent space of `p` on `M`. By default
this is a `:Gaussian` distribution.
"""
function random_tangent(M::AbstractManifold, p, options...)
    return random_tangent(M, p, :Gaussian, options...)
end
function random_tangent(M::AbstractManifold, p, s::Symbol, options...)
    return random_tangent(M, p, Val(s), options...)
end

@doc raw"""
    random_tangent(M::Circle, p [, :Gaussian, σ=1.0])

return a random tangent vector from the tangent space of the point `p` on the
[Circle](https://juliamanifolds.github.io/Manifolds.jl/stable/interface.html#ManifoldsBase.Manifold) ``\mathbb S^1`` by using a normal distribution with
mean 0 and standard deviation 1.
"""
random_tangent(::Circle, p, ::Val{:Gaussian}, σ::Real=1.0) = σ * randn()

function random_tangent(M::Euclidean, p, ::Val{:Gaussian}, σ::Float64=1.0)
    return σ * randn(manifold_dimension(M))
end

@doc raw"""
    random_tangent(M::Grassmann, p[,type=:Gaussian, σ=1.0])

return a (Gaussian) random vector from the tangent space ``T_p\mathrm{Gr}(n,k)`` with mean
zero and standard deviation `σ` by projecting a random Matrix onto the  `p`.
"""
function random_tangent(M::Grassmann, p, ::Val{:Gaussian}, σ::Float64=1.0)
    Z = σ * randn(eltype(p), size(p))
    X = project(M, p, Z)
    X = X ./ norm(X)
    return X
end

@doc raw"""
    random_tangent(M::FixedRankMatrices, p, options...)

generate a random tangent vector in the tangent space of the point `p` on the
`FixedRankMatrices` manifold `M`.
"""
function random_tangent(::FixedRankMatrices{m,n,k}, p, options...) where {m,n,k}
    Up = randn(m, k, options...)
    Vp = randn(n, k, options...)
    A = randn(k, k, options...)
    return UMVTVector(Up - p.U * p.U' * Up, A, Vp' - Vp' * p.Vt' * p.Vt)
end

@doc raw"""
    random_tangent(M::Hyperbolic, p, :Gaussian [, σ=1.0])

generate a random point on the Hyperbolic manifold by projecting a point from the embedding
with respect to the Minkowski metric.
"""
function random_tangent(M::Hyperbolic, p, ::Val{:Gaussian}, σ=1.0)
    Y = σ * randn(eltype(p), size(p))
    X = project(M, p, Y)
    return X
end

function random_tangent(M::PowerManifold, p, options...)
    rep_size = representation_size(M.manifold)
    X = zero_vector(M, p)
    for i in get_iterator(M)
        X[M, i] = random_tangent(M.manifold, p[M, i], options...)
    end
    return X
end

@doc raw"""
    random_tangent(M::ProductManifold, p)

generate a random tangent vector in the tangent space of the point `p` on the
`ProductManifold` `M`.
"""
function random_tangent(M::ProductManifold, p, options...)
    X = map(
        (m, p) -> random_tangent(m, p, options...),
        M.manifolds,
        submanifold_components(M, p),
    )
    return ProductRepr(X...)
end

@doc raw"""
    random_tangent(M::Rotations, p[, type=:Gaussian, σ=1.0])

return a random tangent vector in the tangent space
``T_x\mathrm{SO}(n)`` of the point `x` on the `Rotations` manifold `M` by generating
a random skew-symmetric matrix. The function takes the real upper triangular matrix of a
(Gaussian) random matrix ``A`` with dimension ``n\times n`` and subtracts its transposed matrix.
Finally, the matrix is normalized.
"""
function random_tangent(M::Rotations, p, ::Val{:Gaussian}, σ::Real=1.0)
    T = number_eltype(p)
    # Special case: Rotations(1) is just zero-dimensional
    (manifold_dimension(M) == 0) && return zeros(T, 1, 1)
    A = σ .* randn(T, representation_size(M))
    A = triu(A, 1) - transpose(triu(A, 1))
    A = (1 / norm(A)) * A
    return A
end

@doc raw"""
    random_tangent(M::Sphere, p[, :Gaussian, σ=1.0])

return a random tangent vector in the tangent space of `p` on the `Sphere` `M`.
"""
function random_tangent(M::Sphere, p, ::Val{:Gaussian}, σ::Float64=1.0)
    n = σ * randn(size(p)) # Gaussian in embedding
    return project(M, p, n) #project to TpM (keeps Gaussianness)
end

@doc raw"""
    random_tangent(M::Stiefel, p[,type=:Gaussian, σ=1.0])

return a (Gaussian) random vector from the tangent space ``T_p\mathrm{St}(n,k)`` with mean
zero and standard deviation `σ` by projecting a random Matrix onto the  `p`.
"""
function random_tangent(M::Stiefel, p, ::Val{:Gaussian}, σ::Float64=1.0)
    Z = σ * randn(eltype(p), size(p))
    X = project(M, p, Z)
    X = X ./ norm(X)
    return X
end

@doc raw"""
    random_tangent(M::SymmetricPositiveDefinite, p[, :Gaussian, σ = 1.0])

generate a random tangent vector in the tangent space of the point `p` on the
`SymmetricPositiveDefinite` manifold `M` by using a Gaussian distribution
with standard deviation `σ` on an ONB of the tangent space.
"""
function random_tangent(M::SymmetricPositiveDefinite, p, ::Val{:Gaussian}, σ::Float64=0.01)
    # generate ONB in TxM
    I = one(p)
    B = get_basis(M, p, DiagonalizingOrthonormalBasis(I))
    Ξ = get_vectors(M, p, B)
    Ξx = vector_transport_to.(Ref(M), Ref(I), Ξ, Ref(p), Ref(ParallelTransport()))
    return sum(σ * randn(length(Ξx)) .* Ξx)
end

@doc raw"""
    random_tangent(M::SymmetricPositiveDefinite, p, :Rician [,σ = 0.01])

generate a random tangent vector in the tangent space of `p` on
the `SymmetricPositiveDefinite` manifold `M` by using a Rician distribution
with standard deviation `σ`.
"""
function random_tangent(::SymmetricPositiveDefinite, p, ::Val{:Rician}, σ::Real=0.01)
    # Rician
    C = cholesky(Hermitian(p))
    R = C.L + sqrt(σ) * triu(randn(size(p, 1), size(p, 2)), 0)
    return R * R'
end

@doc raw"""
    random_tangent(M::TangentBundle, p, options...)

generate a random tangent vector at `p` on the tangent bundle by calling
[`random_tangent`](@ref) with the given `options...` twice.
"""
function random_tangent(M::TangentBundle, p, options...)
    X = random_tangent(M.manifold, p[M, :point], options...)
    Y = random_tangent(M.manifold, p[M, :point], options...)
    return ProductRepr(X, Y)
end
