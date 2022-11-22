### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ 39dcf482-5b7c-437d-b000-b0766a1e3fc7
using Pkg; Pkg.activate();

# ╔═╡ ff6edcbd-b70d-4c5f-a5da-d3a221c7d595
using Distributions, LinearAlgebra, Manifolds, Manopt, Random

# ╔═╡ 23c48862-6984-11ed-0a6d-9f6c98ae7134
md"""
# Constraint Optimisation in Manopt.jl

This tutorial is a short introduction to using solvers for constraint optimisation in [`Manopt.jl`](https://manoptjl.org).
"""

# ╔═╡ d7bbeb6b-709a-451b-9eb2-2eb7245dbe03
md"""
## Constrained Optimisation on Riemannian manifolds

A constraint optimisation problem is given by

```math
\tag{P}
\begin{align*}
\operatorname*{arg\,min}_{p\in\mathcal M} & f(p)\\
\text{such that} &\quad g(p) \leq 0\\
&\quad h(p) = 0,\\ 
\end{align*}
```
where ``f\colon \mathcal M → ℝ`` is a cost function, and ``g\colon \mathcal M → ℝ^m`` and ``h\colon \mathcal M → ℝ^n`` are the inequality and equality constraints, respectively. The ``\leq`` and ``=`` in (P) are meant elementwise.

This can be seen as a balance between moving constraints into the geometry of a manifold ``\mathcal M`` and keeping some, since they can be handled well in algorithms, see [^BergmannHerzog2019], [^LiuBoumal2020] for details.
"""

# ╔═╡ 7c0b460e-34e4-4d7b-be51-71f4a38f28e3
md"Let's first again load the necessary packages"

# ╔═╡ 8b0eae06-2218-4a22-a7ae-fc2344ab09f1
Random.seed!(42);

# ╔═╡ 6db180cf-ccf9-4a9e-b69f-1b39db6703d3
md"""
In this tutorial we want to look at different ways to specify the problem and its implications. We start with specifying an example problems to illustrayte the different available forms.
"""

# ╔═╡ 4045214f-dada-4211-afe7-056f78492d8c
md"""
We will consider the problem of a Nonnegative PCA, cf. Section 5.1.2 in [^LiuBoumal2020]:

let ``v_0 ∈ ℝ^d``, ``\lVert v_0 \rVert=1`` be given spike signal, that is a signal that is sparse with only ``s=\lfloor δd \rfloor`` nonzero entries.

```math
  Z = \sqrt{σ} v_0v_0^{\mathrm{T}}+N,
```
where ``\sigma`` is a signal-to-noise ratio and ``N`` is a matrix with random entries, where the diagonal entries are distributed with zero mean and standard deviation ``1/d`` on the off-diagonals and ``2/d`` on the daigonal
"""

# ╔═╡ cd9f3e01-bd76-4972-a04e-028758baa9a3
d = 17 # dimension of v0

# ╔═╡ bd27b323-3571-4cb8-91a1-67fae56ef43b
σ = 0.1 # SNR

# ╔═╡ d463b59a-69b3-4b7f-883b-c65ffe46efe9
δ = 0.1; s = Int(floor(δ * d)); # Sparsity

# ╔═╡ 7eaab1cc-9238-44cf-b57f-4321a486cdaa
S = sample(1:d, s; replace=false)

# ╔═╡ 5654d627-25bc-4d6c-a562-49dad43799da
v0 = [i ∈ S ? 1 / sqrt(s) : 0.0 for i in 1:d];

# ╔═╡ e906df2b-5db9-4540-9200-5e3e0671e269
N = rand(Normal(0, 1 / d), (d, d)); N[diagind(N, 0)] .= rand(Normal(0, 2 / d), d);

# ╔═╡ 155a1401-79aa-4305-81f7-505e43c73094
Z = sqrt(σ) * v0 * transpose(v0) + N;

# ╔═╡ dd078404-8c89-4655-b8ac-e9c816318361
md"""
In order to recover ``v_0`` we consider the constrained optimisation problem on the sphere ``\mathcal S^{d-1}`` given by

```math
\begin{align*}
\operatorname*{arg\,min}_{p\in\mathcal S^{d-1}} & -p^{\mathrm{T}}Zp^{\mathrm{T}}\\
\text{such that} &\quad p \geq 0\\
\end{align*}
```

or in the previous notation ``f(p) = -p^{\mathrm{T}}Zp^{\mathrm{T}}`` and ``g(p) = -p``.
"""

# ╔═╡ 54756d9f-1807-45fc-aa72-40bec10d022d
M = Sphere(d - 1)

# ╔═╡ 0ee60004-e7e5-4c9a-8f1d-e493217f11be
md"""
## a first test run

We first defined ``f``  and ``g`` as usual functions
"""

# ╔═╡ 8be00c3b-4385-449b-b58f-3d3ef972c3c3
f(M, p) = -transpose(p) * Z * p

# ╔═╡ 950dab06-b89e-41c7-9d81-3e9f3fb51b4d
g(M, p) = -p

# ╔═╡ 6218866b-18b5-47f7-98a6-6e1192cb1c24
md"""
since ``f`` is a functions defined in the embedding ``ℝ^d`` as well, we obtain its gradient by projection.
"""

# ╔═╡ 92ee76b4-e132-44c7-9ab3-ef4227969fa2
grad_f(M, p) = project(M, p, -transpose(Z) * p / 2 - Z * p / 2)

# ╔═╡ 0f71531d-b292-477d-b108-f45dc4e680ad
md"""
For the constraints this is a little more involved, since each function ``g_i = g(p)_i = p_i`` has to return its own gradient. These are again in the embedding just ``\operatorname{grad} g_i(p) = -e_i`` the ``i`` th unit vector. We can project these again onto the tangent space at ``p``:
"""

# ╔═╡ 9e7028c9-0f15-4245-a089-2670c26b3b40
grad_g(M, p) = project.(Ref(M), Ref(p), [[i == j ? -1.0 : 0.0 for j in 1:d] for i in 1:d])

# ╔═╡ 72e99369-165b-494e-9acc-7719a12d9d8d
x0 = random_point(M);

# ╔═╡ c72709e1-7bae-4345-b29b-4ef1e791292b
md"""
Now this is a little slow, so we can modify two things, that we will directly do both – but one could also just change one of these – :

1. Gradients should be evaluated in place, so for example 
"""

# ╔═╡ 717bd019-2978-4e55-a586-ed876cefa65d
grad_f!(M, X, p) = project!(M, X, p, -transpose(Z) * p / 2 - Z * p / 2)

# ╔═╡ db35ae71-c96e-4432-a7d5-3df9f6c0f9fb
md"""
2. The constraints are currently always evaluated all together, since the function `grad_g` always returns a vector of gradients.
We change this _both_ into a vector of gradient functions `\operatorname{grad} g_i` _as well as_ gradients that are computed in place.
"""

# ╔═╡ fb86f597-f8af-4c98-b5b1-4db0dfc06199
g2 = [(M, p) -> -p[i] for i in 1:d];

# ╔═╡ 1d427174-57da-41d6-8577-d97d643a2142
grad_g2! = [
    (M, X, p) -> project!(M, X, p, -[i == j ? -1.0 : 0.0 for j in 1:d]) for i in 1:d
];

# ╔═╡ ce8f1156-a350-4fde-bd39-b08a16b2821d
@time v2 = augmented_Lagrangian_method(
    M,
    f,
    grad_f!,
    x0;
    evaluation=MutatingEvaluation(),
    G=g2,
    gradG=grad_g2!,
    debug=[:Iteration, :Cost, :Stop, 10, "\n"],
    sub_kwargs=[
        :debug =>
            ["   ", :Iteration, :Cost, " | ", DebugStepsize(), :Change, :Stop, 50, "\n"],
    ],
)

# ╔═╡ 78d055e8-d5c8-4cdf-a706-3089368397bd
md"""
## Literature
[^BergmannHerzog2019]:
    > R. Bergmann, R. Herzog, __Intrinsic formulation of KKT conditions and constraint qualifications on smooth manifolds__,
	> In: SIAM Journal on Optimization 29(4), pp. 2423–2444 (2019)
	> doi: [10.1137/18M1181602](https://doi.org/10.1137/18M1181602),
	> arXiv: [1804.06214](https://arxiv.org/abs/1804.06214).
[^LiuBoumal2020]:
    > C. Liu, N. Boumal, __Simple Algorithms for Optimization on Riemannian Manifolds with Constraints__,
    > In: Applied Mathematics & Optimization 82, pp. 949–981 (2020),
    > doi [10.1007/s00245-019-09564-3](https://doi.org/10.1007/s00245-019-09564-3),
    > arXiv: [1901.10000](https://arxiv.org/abs/1901.10000).
"""

# ╔═╡ Cell order:
# ╟─23c48862-6984-11ed-0a6d-9f6c98ae7134
# ╟─d7bbeb6b-709a-451b-9eb2-2eb7245dbe03
# ╟─7c0b460e-34e4-4d7b-be51-71f4a38f28e3
# ╠═39dcf482-5b7c-437d-b000-b0766a1e3fc7
# ╠═ff6edcbd-b70d-4c5f-a5da-d3a221c7d595
# ╠═8b0eae06-2218-4a22-a7ae-fc2344ab09f1
# ╟─6db180cf-ccf9-4a9e-b69f-1b39db6703d3
# ╟─4045214f-dada-4211-afe7-056f78492d8c
# ╠═cd9f3e01-bd76-4972-a04e-028758baa9a3
# ╠═bd27b323-3571-4cb8-91a1-67fae56ef43b
# ╠═d463b59a-69b3-4b7f-883b-c65ffe46efe9
# ╠═7eaab1cc-9238-44cf-b57f-4321a486cdaa
# ╠═5654d627-25bc-4d6c-a562-49dad43799da
# ╠═e906df2b-5db9-4540-9200-5e3e0671e269
# ╠═155a1401-79aa-4305-81f7-505e43c73094
# ╟─dd078404-8c89-4655-b8ac-e9c816318361
# ╠═54756d9f-1807-45fc-aa72-40bec10d022d
# ╠═0ee60004-e7e5-4c9a-8f1d-e493217f11be
# ╠═8be00c3b-4385-449b-b58f-3d3ef972c3c3
# ╠═950dab06-b89e-41c7-9d81-3e9f3fb51b4d
# ╠═6218866b-18b5-47f7-98a6-6e1192cb1c24
# ╠═92ee76b4-e132-44c7-9ab3-ef4227969fa2
# ╟─0f71531d-b292-477d-b108-f45dc4e680ad
# ╠═9e7028c9-0f15-4245-a089-2670c26b3b40
# ╠═72e99369-165b-494e-9acc-7719a12d9d8d
# ╟─c72709e1-7bae-4345-b29b-4ef1e791292b
# ╠═717bd019-2978-4e55-a586-ed876cefa65d
# ╟─db35ae71-c96e-4432-a7d5-3df9f6c0f9fb
# ╠═fb86f597-f8af-4c98-b5b1-4db0dfc06199
# ╠═1d427174-57da-41d6-8577-d97d643a2142
# ╠═ce8f1156-a350-4fde-bd39-b08a16b2821d
# ╟─78d055e8-d5c8-4cdf-a706-3089368397bd
