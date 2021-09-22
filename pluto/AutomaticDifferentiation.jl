### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ b0769dfa-28cf-440e-9ba2-1ef488f171a9
begin # dev mode – use global env on my machine
    import Pkg
    # careful: this is _not_ a reproducible environment
    # activate the global environment
    Pkg.activate()
   using ReverseDiff, Manifolds, Manopt, Test, Random, LinearAlgebra, FiniteDiff
end

# ╔═╡ 0213d26a-18ac-11ec-03fd-ada5992bcea8
md"""
# Using (Euclidean) AD in Manopt.jl
"""

# ╔═╡ f3bc91ee-5871-4cba-ac89-190deb71ad0f
md"""
Since Manifolds.jl 0.6.7 the support of automatic differentiation support has been extended. With [ForwardDiff.jl](https://juliadiff.org/ForwardDiff.jl/stable/) or [ReverseDiff.jl](https://juliadiff.org/ReverseDiff.jl/) you can now use automatic differentiation on manifolds for embedded manifolds.
"""

# ╔═╡ 35f02ab2-30d1-4c3c-8ba7-d07f391019e0
md"""
In general there is currenlty two ways to get the gradient of a function ``f\colon\mathcal M \to ℝ`` on a Riemannian manifold ``\mathcal M``.
"""

# ╔═╡ d9be6c2f-65fd-4685-9005-da22bf985e28
md"""
In this Notebook we will take a look at a few possibilities to approximate or derive the gradient of a function ``f:\mathcal M \to ℝ`` on a Riemannian manifold, without computing it yourself. There is mainly two different philosophies:

1. Working _instrinsically_, i.e. stay on the manifold and in the tangent spaces. Here, we will consider approximating the gradient by forward differences.

2. Working in an embedding – there we can use all tools from functions on Euclidean spaces – finite differences or automatic differenciation – and then compute the corresponding Riemannian gradient from there.

Let's first load all packages we need.
"""

# ╔═╡ f88b15de-cec6-4bc8-9b68-2a407b5aeded
# using ReverseDiff, Manifolds, Manopt, Test, Random, LinearAlgebra

# ╔═╡ 18d7459f-eed6-489b-a096-ac77ccd781af
md"""
## 1. (Intrinsic) Forward Differences

A first idea is to generalise (multivariate) finite differences to Riemannian manifolds. Let ``X_1,\ldots,X_d ∈ T_p\mathcal M`` denote an orthonormal basis of the tangent space ``T_p\mathcal M`` at the point ``p∈\mathcal M`` on the Riemannian manifold.

We can generalise the notion of a directional derivative, i.e. for the “direction” ``Y∈T_p\mathcal M`` let ``c\colon [-ε,ε]``, ``ε>0``, be a curve with ``c(0) = p``, ``\dot c(0) = Y`` and we obtain  

```math
	Df(p)[Y] = \frac{\mathrm{d}}{\mathrm{d}t} f(c(t)) = \lim_{h \to 0} \frac{1}{h}(f(\exp_p(hY)-f(p))
```

We can approximate ``Df(p)[X]`` by a finite difference scheme for an ``h>0`` as

```math
DF(p)[Y] ≈ G_h(Y) := \frac{1}{h}(f(\exp_p(hY)-f(p))
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
	n = 20
	A = randn(n+1,n+1)
	A = Symmetric(A)
	M = Sphere(n)
	nothing
end

# ╔═╡ 41c204dd-6e4e-4a70-8f06-209a469e0680
f1(p) = p'*A'p

# ╔═╡ 2e33de5e-ffaa-422a-91d9-61f588ed1211
gradf1(p) = 2*(A*p - p*p'*A*p) 

# ╔═╡ bbd9a010-1981-45b3-bf7d-c04bcd2c2128
md"""Manifolds provides a finite difference scheme"""

# ╔═╡ 0c823b57-a009-44b7-9901-4b1a1ed7e103
finite_diff = Manifolds.FiniteDiffBackend(Val(:forward))

# ╔═╡ 47b536ea-cd9c-4083-bbd0-f69904a1307d


# ╔═╡ 08456b40-74ec-4319-93e7-130b5cf70ac3
r_backend = RiemannianONBDiffBackend(finite_diff, ExponentialRetraction(), LogarithmicInverseRetraction(), DefaultOrthonormalBasis())

# ╔═╡ 12327b62-7e79-4381-b6a7-f85b08a8251b
gradf1_FD(p) = Manifolds.gradient(M, f1, p, r_backend)

# ╔═╡ 07f9a630-e53d-45ea-b109-3d4de190723d
begin
	p = zeros(n+1)
	p[1] = 1.0
	X1 = gradf1(p)
	X2 = gradf1_FD(p)
	[X1-X2]
end

# ╔═╡ 8e5f677d-dafa-49b9-b678-3f129be31dcf
is_vector(M, p, X1)

# ╔═╡ ab5faa5f-1394-40c2-8c81-0e0f5449cd72
is_vector(M, p, X2)

# ╔═╡ 77769eab-54dd-41dc-8125-0382e5ef0bf1
md"""
## 2. Conversion of an Euclidean Gradient in the Embedding to a Riemannian Gradient of an (not necessarily isometrically) embedded Manifold

Let ``\tilde f\colon\mathbb R^m \to \mathbb R`` be a function un the embedding of an ``n``-dimensional manifold ``\mathcal M \subset \mathbb R^m`` and ``f\colon \mathcal M \to \mathbb R`` denote the restriction of ``\tilde f`` to the manifold ``\mathcal M``.

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
## Example
As an example we use the Rayleigh quotient ``f(x) = \frac{x^{\mathrm{T}}Ax}{x^{\mathrm{T}}x}`` for a given symmetric matrix ``A``.
"""

# ╔═╡ e3b955b1-c780-4302-a605-190c3d10cd6f
Random.seed!(42)

# ╔═╡ c3f3aeba-2849-4715-94e2-0c44613a2ce9
f̃(x) = x'*A*x/(x'*x);

# ╔═╡ 786fce04-53ef-448d-9657-31208b35fb7e
md"The cost function is the same by restriction"

# ╔═╡ c1341fef-adec-4574-a642-a1a8a9c1fee5
f2(p) = f̃(p);

# ╔═╡ 0818a62f-1bef-44f7-a33f-1ab0054e853c
md"The gradient is now computed combining our gradient scheme with ReverseDiff."

# ╔═╡ 89cd6b4b-f9ef-47ac-afd3-cf9aacf43256
Manifolds.rgradient_backend!(Manifolds.RiemannianProjectionGradientBackend(???SomeMagic???)

# ╔═╡ 9cd489f1-0cfa-4ab8-bc1f-d5d4b4a4cb39
gradf(M, p) = gradient(M, f, p, ???SomeFurtherMagic???)

# ╔═╡ 558dc14f-00f8-4aab-bc0f-6ce132068259
p0 = zeros(n); p0[1] = 1.;

# ╔═╡ 5dcab4ea-8ecc-46ae-ad98-1670ee795a4a
gradient_descent(M, f, gradf, p0)

# ╔═╡ Cell order:
# ╟─0213d26a-18ac-11ec-03fd-ada5992bcea8
# ╟─f3bc91ee-5871-4cba-ac89-190deb71ad0f
# ╟─35f02ab2-30d1-4c3c-8ba7-d07f391019e0
# ╟─d9be6c2f-65fd-4685-9005-da22bf985e28
# ╠═f88b15de-cec6-4bc8-9b68-2a407b5aeded
# ╠═b0769dfa-28cf-440e-9ba2-1ef488f171a9
# ╟─18d7459f-eed6-489b-a096-ac77ccd781af
# ╟─a3df142e-94df-48d2-be08-d1f1f3854c76
# ╟─9a030ac6-1f44-4fa6-8bc9-1c0278e97fe2
# ╠═19747159-d383-4547-9315-0ed2494904a6
# ╠═41c204dd-6e4e-4a70-8f06-209a469e0680
# ╠═2e33de5e-ffaa-422a-91d9-61f588ed1211
# ╠═bbd9a010-1981-45b3-bf7d-c04bcd2c2128
# ╠═0c823b57-a009-44b7-9901-4b1a1ed7e103
# ╠═47b536ea-cd9c-4083-bbd0-f69904a1307d
# ╠═08456b40-74ec-4319-93e7-130b5cf70ac3
# ╠═12327b62-7e79-4381-b6a7-f85b08a8251b
# ╠═07f9a630-e53d-45ea-b109-3d4de190723d
# ╠═8e5f677d-dafa-49b9-b678-3f129be31dcf
# ╠═ab5faa5f-1394-40c2-8c81-0e0f5449cd72
# ╠═77769eab-54dd-41dc-8125-0382e5ef0bf1
# ╟─57cda07f-e432-46af-b771-5e5a3067feac
# ╟─e3b955b1-c780-4302-a605-190c3d10cd6f
# ╠═c3f3aeba-2849-4715-94e2-0c44613a2ce9
# ╟─786fce04-53ef-448d-9657-31208b35fb7e
# ╠═c1341fef-adec-4574-a642-a1a8a9c1fee5
# ╟─0818a62f-1bef-44f7-a33f-1ab0054e853c
# ╠═89cd6b4b-f9ef-47ac-afd3-cf9aacf43256
# ╠═9cd489f1-0cfa-4ab8-bc1f-d5d4b4a4cb39
# ╠═558dc14f-00f8-4aab-bc0f-6ce132068259
# ╠═5dcab4ea-8ecc-46ae-ad98-1670ee795a4a
