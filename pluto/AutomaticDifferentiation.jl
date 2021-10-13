### A Pluto.jl notebook ###
# v0.16.1

using Markdown
using InteractiveUtils

# ╔═╡ f88b15de-cec6-4bc8-9b68-2a407b5aeded
begin
    using Pkg: Pkg
    Pkg.activate(mktempdir())
    Pkg.add([
        Pkg.PackageSpec(; name="Manifolds", version="0.7"),
        Pkg.PackageSpec(; name="Manopt", version="0.3.13"),
        Pkg.PackageSpec(; name="FiniteDiff", version="2.8.1"),
        Pkg.PackageSpec(; name="ReverseDiff", version="1.9.0"),
        # Pkg.PackageSpec(; name="PlutoUI"),
    ])
end

# ╔═╡ 856f336c-e232-4f1f-b1ac-759b4558acd1
using Manifolds, Manopt, Random, LinearAlgebra

# ╔═╡ b0769dfa-28cf-440e-9ba2-1ef488f171a9
using FiniteDiff, ReverseDiff

# ╔═╡ 0213d26a-18ac-11ec-03fd-ada5992bcea8
md"""
# Using (Euclidean) AD in Manopt.jl
"""

# ╔═╡ f3bc91ee-5871-4cba-ac89-190deb71ad0f
md"""
Since [Manifolds.jl](https://juliamanifolds.github.io/Manifolds.jl/latest/) 0.7 the support of automatic differentiation support has been extended.

This tutorial explains how to use Euclidean tools to derive a gradient for a real-valued function ``F\colon \mathcal M → ℝ``. We will consider two methods: an intrinsic variant and a variant employing the embedding. These gradients can then be used within any gradient based optimisation algorithm in [Manopt.jl](https://manoptjl.org).

While by default we use [FiniteDifferences.jl](https://juliadiff.org/FiniteDifferences.jl/latest/), you can also use [FiniteDiff.jl](https://github.com/JuliaDiff/FiniteDiff.jl), [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/), [ReverseDiff.jl](https://juliadiff.org/ReverseDiff.jl/), or  [Zygote.jl](https://fluxml.ai/Zygote.jl/).
"""

# ╔═╡ d9be6c2f-65fd-4685-9005-da22bf985e28
md"""
In this Notebook we will take a look at a few possibilities to approximate or derive the gradient of a function ``f:\mathcal M \to ℝ`` on a Riemannian manifold, without computing it yourself. There is mainly two different philosophies:

1. Working _instrinsically_, i.e. stay on the manifold and in the tangent spaces. Here, we will consider approximating the gradient by forward differences.

2. Working in an embedding – there we can use all tools from functions on Euclidean spaces – finite differences or automatic differenciation – and then compute the corresponding Riemannian gradient from there.

Let's first load all packages we need.
"""

# ╔═╡ 18d7459f-eed6-489b-a096-ac77ccd781af
md"""
## 1. (Intrinsic) Forward Differences

A first idea is to generalise (multivariate) finite differences to Riemannian manifolds. Let ``X_1,\ldots,X_d ∈ T_p\mathcal M`` denote an orthonormal basis of the tangent space ``T_p\mathcal M`` at the point ``p∈\mathcal M`` on the Riemannian manifold.

We can generalise the notion of a directional derivative, i.e. for the “direction” ``Y∈T_p\mathcal M`` let ``c\colon [-ε,ε]``, ``ε>0``, be a curve with ``c(0) = p``, ``\dot c(0) = Y`` and we obtain

```math
	Df(p)[Y] = \frac{\mathrm{d}}{\mathrm{d}t} f(c(t)) = \lim_{h \to 0} \frac{1}{h}(f(\exp_p(hY))-f(p))
```

We can approximate ``Df(p)[X]`` by a finite difference scheme for an ``h>0`` as

```math
DF(p)[Y] ≈ G_h(Y) := \frac{1}{h}(f(\exp_p(hY))-f(p))
```

Furthermore the gradient ``\operatorname{grad}f`` is the Riesz representer of the differential, ie.

```math
	Df(p)[Y] = g_p(\operatorname{grad}f(p), Y),\qquad \text{ for all } Y ∈ T_p\mathcal M
```

and since it is a tangent vector, we can write it in terms of a basis as

```math
	\operatorname{grad}f(p) = \sum_{i=1}^{d} g_p(\operatorname{grad}f(p),X_i)X_i
	= \sum_{i=1}^{d} Df(p)[X_i]X_i
```

and perform the approximation from above to obtain
```math
	\operatorname{grad}f(p) ≈ \sum_{i=1}^{d} G_h(X_i)X_i
```
for some suitable step size ``h``.This comes at the cost of ``d+1`` function evaluations and ``d`` exponential maps.
"""

# ╔═╡ a3df142e-94df-48d2-be08-d1f1f3854c76
md"""
This is the first variant we can use. An advantage is, that it is _intrinsic_ in the sense that it does not require any embedding of the manifold.
"""

# ╔═╡ 9a030ac6-1f44-4fa6-8bc9-1c0278e97fe2
md""" ### An Example: The Rayleigh Quotient

The Rayleigh quotient is concerned with finding Eigenvalues (and Eigenvectors) of a symmetric matrix $A\in ℝ^{(n+1)×(n+1)}$. The optimisation problem reads

```math
F\colon ℝ^{n+1} \to ℝ,\quad F(\mathbf x) = \frac{\mathbf x^\mathrm{T}A\mathbf x}{\mathbf x^\mathrm{T}\mathbf x}
```

Minimizing this function yields the smallest eigenvalue ``\lambda_1`` as a value and the corresponding minimizer ``\mathbf x^*`` is a corresponding eigenvector.

Since the length of an eigenvector is irrelevant, there is an ambiguity in the cost function. It can be better phrased on the sphere ``𝕊^n`` of unit vectors in ``\mathbb R^{n+1}``, i.e.

```math
\operatorname*{arg\,min}_{p \in 𝕊^n} f(p) = \operatorname*{arg\,min}_{p \in 𝕊^n} p^\mathrm{T}Ap
```

We can compute the Riemannian gradient exactly as

```math
\operatorname{grad} f(p) = 2(Ap - pp^\mathrm{T}Ap)
```

so we can compare it to the approximation by finite differences.
"""

# ╔═╡ 19747159-d383-4547-9315-0ed2494904a6
begin
    Random.seed!(42)
    n = 200
    A = randn(n + 1, n + 1)
    A = Symmetric(A)
    M = Sphere(n)
    nothing
end

# ╔═╡ 41c204dd-6e4e-4a70-8f06-209a469e0680
f1(p) = p' * A'p

# ╔═╡ 2e33de5e-ffaa-422a-91d9-61f588ed1211
gradf1(p) = 2 * (A * p - p * p' * A * p)

# ╔═╡ bbd9a010-1981-45b3-bf7d-c04bcd2c2128
md"""Manifolds provides a finite difference scheme in Tangent spaces, that you can introduce to use an existing framework (if the wrapper is implemented) form Euclidean space. Here we use `FiniteDiff.jl`."""

# ╔═╡ 08456b40-74ec-4319-93e7-130b5cf70ac3
r_backend = Manifolds.TangentDiffBackend(Manifolds.FiniteDiffBackend())

# ╔═╡ 12327b62-7e79-4381-b6a7-f85b08a8251b
gradf1_FD(p) = Manifolds.gradient(M, f1, p, r_backend)

# ╔═╡ 07f9a630-e53d-45ea-b109-3d4de190723d
begin
    p = zeros(n + 1)
    p[1] = 1.0
    X1 = gradf1(p)
    X2 = gradf1_FD(p)
    norm(M, p, X1 - X2)
end

# ╔═╡ 8e5f677d-dafa-49b9-b678-3f129be31dcf
md"We obtain quite a good approximation of the gradient."

# ╔═╡ 77769eab-54dd-41dc-8125-0382e5ef0bf1
md"""
## 2. Conversion of an Euclidean Gradient in the Embedding to a Riemannian Gradient of an (not necessarily isometrically) embedded Manifold

Let ``\tilde f\colon\mathbb R^m \to \mathbb R`` be a function in the embedding of an ``n``-dimensional manifold ``\mathcal M \subset \mathbb R^m`` and ``f\colon \mathcal M \to \mathbb R`` denote the restriction of ``\tilde f`` to the manifold ``\mathcal M``.

Since we can use the push forward of the embedding to also embed the tangent space ``T_p\mathcal M``, ``p\in \mathcal M``, we can similarly obtain the differential ``Df(p)\colon T_p\mathcal M \to \mathbb R`` by restricting the differential ``D\tilde f(p)`` to the tangent space.

If both ``T_p\mathcal M`` and ``T_p\mathcal R^m`` have the same inner product, or in other words the manifold is isometrically embedded in ``R^m`` (like for example the sphere ``\mathbb S^n\subset\mathbb R^{m+1}`` then this restriction of the differential directly translates to a projection of the gradient, i.e.

```math
\operatorname{grad}f(p) = \operatorname{Proj}_{T_p\mathcal M}(\operatorname{grad} \tilde f(p))
```

More generally we might have to take a change of the metric into account, i.e.

```math
\langle  \operatorname{Proj}_{T_p\mathcal M}(\operatorname{grad} \tilde f(p)), X \rangle
= Df(p)[X] = g_p(\operatorname{grad}f(p), X)
```

or in words: we have to change the Riesz representer of the (restricted/projected) differential of ``f`` (``\tilde f``) to the one with respect to the Riemannian metric. This is done using [`change_representer`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/metric.html#Manifolds.change_representer-Tuple{AbstractManifold,%20AbstractMetric,%20Any,%20Any}).
"""

# ╔═╡ 57cda07f-e432-46af-b771-5e5a3067feac
md"""
### A continued Example
We continue with the Rayleigh Quotient from before, now just starting with the defintion of the Euclidean case in the embedding, the function ``F``.
"""

# ╔═╡ c3f3aeba-2849-4715-94e2-0c44613a2ce9
F(x) = x' * A * x / (x' * x);

# ╔═╡ 786fce04-53ef-448d-9657-31208b35fb7e
md"The cost function is the same by restriction"

# ╔═╡ c1341fef-adec-4574-a642-a1a8a9c1fee5
f2(M, p) = F(p);

# ╔═╡ 0818a62f-1bef-44f7-a33f-1ab0054e853c
md"The gradient is now computed combining our gradient scheme with ReverseDiff."

# ╔═╡ 89cd6b4b-f9ef-47ac-afd3-cf9aacf43256
function grad_f2_AD(M, p)
    return Manifolds.gradient(
        M, F, p, RiemannianProjectionBackend(Manifolds.ReverseDiffBackend())
    )
end

# ╔═╡ 7c5a8a17-6f63-4587-a94a-6936bdd3cec6
X3 = grad_f2_AD(M, p)

# ╔═╡ b3e7f57f-d87a-47c5-b8ad-48b6d205fa73
norm(M, p, X1 - X3)

# ╔═╡ 893db402-283f-4e3e-8bf7-c6f22e485efb
md"""
### An Example for a nonisometrically embedded Manifold

on the manifold ``\mathcal P(3)`` of symmetric positive definite matrices.
"""

# ╔═╡ 8494a0d6-dbf2-4eb0-a555-f00e446fbe38
md"""
The following function computes (half) the distance squared (with respect to the linear affine metric) on the manifold ``\mathcal P(3)`` to the identity, i.e. $I_3$. denoting the unit matrix we consider the function

```math
	G(q) = \frac{1}{2}d^2_{\mathcal P(3)}(q,I_3) = \lVert \operatorname{Log}(q) \rVert_F^2,
```
where $\operatorname{Log}$ denotes the matrix logarithm and ``\lVert \cdot \rVert_F`` is the Frobenius norm.
This can be computed for symmetric positive definite matrices by summing the squares of the ``\log``arithms of the eigenvalues of ``q`` and divide by two:
"""

# ╔═╡ c93eb2da-89df-4751-b086-62be604d41e6
G(q) = sum(log.(eigvals(Symmetric(q))) .^ 2) / 2

# ╔═╡ e2bf6f55-7235-4d75-8bee-a325434e32ad
md"""
We can also interpret this as a function on the space of matrices and apply the Euclidean finite differences machinery; in this way we can easily derive the Euclidean gradient. But when computing the Riemannian gradient, we have to change the representer (see again [`change_representer`](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/metric.html#Manifolds.change_representer-Tuple{AbstractManifold,%20AbstractMetric,%20Any,%20Any})) after projecting onto the tangent space ``T_p\mathcal P(n)`` at ``p``.

Let's first define a point and the manifold ``N=\mathcal P(3)``.
"""

# ╔═╡ 153378ca-703d-4a84-bc63-22347399a160
rotM(α) = [1.0 0.0 0.0; 0.0 cos(α) sin(α); 0.0 -sin(α) cos(α)]

# ╔═╡ 699f0177-2c5b-434b-9eca-b6fc573e497f
q = rotM(π / 6) * [1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 3.0] * transpose(rotM(π / 6))

# ╔═╡ 9b52d05c-4cba-4922-85a4-1a2a3c74823e
N = SymmetricPositiveDefinite(3)

# ╔═╡ cb3fe7aa-1262-48f2-9ebc-3e959c72a33e
is_point(N, q)

# ╔═╡ 13712c64-48fd-4f2a-9ee4-1949e51d316f
md"""We could first just compute the gradient using `FiniteDiff.jl`, but this yields the Euclidean gradient:"""

# ╔═╡ 64beb3dd-9507-4792-be02-ae1405704690
FiniteDiff.finite_difference_gradient(G, q)

# ╔═╡ 2be4f9e8-0331-44ac-839f-7bb71d9edef9
md"""Instead, we use the [`RiemannianProjectedBackend`](https://juliamanifolds.github.io/Manifolds.jl/latest/features/differentiation.html#Manifolds.RiemannianProjectionBackend) of `Manifolds.jl`, which in this case internally uses `FiniteDiff.jl` to compute a Euclidean gradient but then uses the conversion explained above to derive the Riemannian gradient.

We define this here again as a function `grad_G_FD` that could be used in the `Manopt.jl` framework within a gradient based optimisation.
"""

# ╔═╡ 6f1d748f-27ce-496b-8561-f16972da50cc
function grad_G_FD(N, q)
    return Manifolds.gradient(
        N, G, q, RiemannianProjectionBackend(Manifolds.FiniteDiffBackend())
    )
end

# ╔═╡ 7dd656ea-08de-4172-8a92-87ad2228ce69
G1 = grad_G_FD(N, q)

# ╔═╡ 219573d2-283f-456c-a5c3-fadd734fc157
md"""
Now, we can agaon compare this to the (known) solution of the gradient, namely the gradient of (a half) the distance suqared, i.e. ``G(q) = \frac{1}{2}d^2_{\mathcal P(3)}(q,I_3)`` is given by ``\operatorname{grad} G(q) = -\operatorname{log}_q I_3``, where ``\operatorname{log}`` is the [logarithmic map](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/symmetricpositivedefinite.html#Base.log-Tuple{SymmetricPositiveDefinite,%20Vararg{Any,%20N}%20where%20N}) on the manifold.
"""

# ╔═╡ e28a2752-877c-4ab4-a253-8d26fa9a73c2
G2 = -log(N, q, Matrix{Float64}(I, 3, 3))

# ╔═╡ 25c65878-1be6-4fec-b65e-9c1741320a41
md"""Both terms agree up to ``1.2×10^{-10}``:"""

# ╔═╡ 9a66d4f3-508d-4285-9a93-df1323575202
norm(G1 - G2)

# ╔═╡ c07fb3d0-d12f-44d7-bcab-7a0d39e6af8d
isapprox(M, q, G1, G2; atol=2 * 1e-10)

# ╔═╡ f47c70b6-ca05-498f-9e10-58c3839ca427
md""" In this case we can not use `ReverseDiff.jl`, since it can not handle the `eigvals!` function that is called internally."""

# ╔═╡ 32d8d025-3993-4d31-9eea-3463e0af1c12
md"""
## Summary

This tutorial illustrates how to use tools from Euclidean spaces, finite differences or automatic differentiation, to compute gradients on Riemannian manifolds. The scheme allows to use _any_ differentiation framework within the embedding to derive a Riemannian gradient.
"""

# ╔═╡ Cell order:
# ╟─0213d26a-18ac-11ec-03fd-ada5992bcea8
# ╟─f3bc91ee-5871-4cba-ac89-190deb71ad0f
# ╟─d9be6c2f-65fd-4685-9005-da22bf985e28
# ╟─f88b15de-cec6-4bc8-9b68-2a407b5aeded
# ╠═856f336c-e232-4f1f-b1ac-759b4558acd1
# ╠═b0769dfa-28cf-440e-9ba2-1ef488f171a9
# ╟─18d7459f-eed6-489b-a096-ac77ccd781af
# ╟─a3df142e-94df-48d2-be08-d1f1f3854c76
# ╟─9a030ac6-1f44-4fa6-8bc9-1c0278e97fe2
# ╠═19747159-d383-4547-9315-0ed2494904a6
# ╠═41c204dd-6e4e-4a70-8f06-209a469e0680
# ╠═2e33de5e-ffaa-422a-91d9-61f588ed1211
# ╟─bbd9a010-1981-45b3-bf7d-c04bcd2c2128
# ╠═08456b40-74ec-4319-93e7-130b5cf70ac3
# ╠═12327b62-7e79-4381-b6a7-f85b08a8251b
# ╠═07f9a630-e53d-45ea-b109-3d4de190723d
# ╟─8e5f677d-dafa-49b9-b678-3f129be31dcf
# ╟─77769eab-54dd-41dc-8125-0382e5ef0bf1
# ╟─57cda07f-e432-46af-b771-5e5a3067feac
# ╠═c3f3aeba-2849-4715-94e2-0c44613a2ce9
# ╟─786fce04-53ef-448d-9657-31208b35fb7e
# ╠═c1341fef-adec-4574-a642-a1a8a9c1fee5
# ╟─0818a62f-1bef-44f7-a33f-1ab0054e853c
# ╠═89cd6b4b-f9ef-47ac-afd3-cf9aacf43256
# ╠═7c5a8a17-6f63-4587-a94a-6936bdd3cec6
# ╠═b3e7f57f-d87a-47c5-b8ad-48b6d205fa73
# ╟─893db402-283f-4e3e-8bf7-c6f22e485efb
# ╟─8494a0d6-dbf2-4eb0-a555-f00e446fbe38
# ╠═c93eb2da-89df-4751-b086-62be604d41e6
# ╟─e2bf6f55-7235-4d75-8bee-a325434e32ad
# ╠═153378ca-703d-4a84-bc63-22347399a160
# ╠═699f0177-2c5b-434b-9eca-b6fc573e497f
# ╠═9b52d05c-4cba-4922-85a4-1a2a3c74823e
# ╠═cb3fe7aa-1262-48f2-9ebc-3e959c72a33e
# ╟─13712c64-48fd-4f2a-9ee4-1949e51d316f
# ╠═64beb3dd-9507-4792-be02-ae1405704690
# ╟─2be4f9e8-0331-44ac-839f-7bb71d9edef9
# ╠═6f1d748f-27ce-496b-8561-f16972da50cc
# ╠═7dd656ea-08de-4172-8a92-87ad2228ce69
# ╟─219573d2-283f-456c-a5c3-fadd734fc157
# ╠═e28a2752-877c-4ab4-a253-8d26fa9a73c2
# ╟─25c65878-1be6-4fec-b65e-9c1741320a41
# ╠═9a66d4f3-508d-4285-9a93-df1323575202
# ╠═c07fb3d0-d12f-44d7-bcab-7a0d39e6af8d
# ╟─f47c70b6-ca05-498f-9e10-58c3839ca427
# ╟─32d8d025-3993-4d31-9eea-3463e0af1c12
