### A Pluto.jl notebook ###
# v0.19.13

using Markdown
using InteractiveUtils

# ╔═╡ ff6edcbd-b70d-4c5f-a5da-d3a221c7d595
using Distributions, LinearAlgebra, Manifolds, Manopt, Random, PlutoUI

# ╔═╡ 23c48862-6984-11ed-0a6d-9f6c98ae7134
md"""
# How to do Constrained Optimization

This tutorial is a short introduction to using solvers for constraint optimisation in [`Manopt.jl`](https://manoptjl.org).
"""

# ╔═╡ d7bbeb6b-709a-451b-9eb2-2eb7245dbe03
md"""
## Introduction
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
d = 150; # dimension of v0

# ╔═╡ bd27b323-3571-4cb8-91a1-67fae56ef43b
σ = 0.1^2; # SNR

# ╔═╡ d463b59a-69b3-4b7f-883b-c65ffe46efe9
δ = 0.1; s = Int(floor(δ * d)); # Sparsity

# ╔═╡ 7eaab1cc-9238-44cf-b57f-4321a486cdaa
S = sample(1:d, s; replace=false);

# ╔═╡ 5654d627-25bc-4d6c-a562-49dad43799da
v0 =  [i ∈ S ? 1 / sqrt(s) : 0.0 for i in 1:d];

# ╔═╡ e906df2b-5db9-4540-9200-5e3e0671e269
N = rand(Normal(0, 1 / d), (d, d)); N[diagind(N, 0)] .= rand(Normal(0, 2 / d), d);

# ╔═╡ 155a1401-79aa-4305-81f7-505e43c73094
 Z = Z = sqrt(σ) * v0 * transpose(v0) + N;

# ╔═╡ dd078404-8c89-4655-b8ac-e9c816318361
md"""
In order to recover ``v_0`` we consider the constrained optimisation problem on the sphere ``\mathcal S^{d-1}`` given by

```math
\begin{align*}
\operatorname*{arg\,min}_{p\in\mathcal S^{d-1}} & -p^{\mathrm{T}}Zp^{\mathrm{T}}\\
\text{such that} &\quad p \geq 0\\
\end{align*}
```

or in the previous notation ``f(p) = -p^{\mathrm{T}}Zp^{\mathrm{T}}`` and ``g(p) = -p``. We first initialize the manifold under consideration
"""

# ╔═╡ 54756d9f-1807-45fc-aa72-40bec10d022d
M = Sphere(d - 1)

# ╔═╡ 0ee60004-e7e5-4c9a-8f1d-e493217f11be
md"""
## A first Augmented Lagrangian Run

We first defined ``f``  and ``g`` as usual functions
"""

# ╔═╡ 8be00c3b-4385-449b-b58f-3d3ef972c3c3
f(M, p) = -transpose(p) * Z * p;

# ╔═╡ 950dab06-b89e-41c7-9d81-3e9f3fb51b4d
g(M, p) = -p;

# ╔═╡ 6218866b-18b5-47f7-98a6-6e1192cb1c24
md"""
since ``f`` is a functions defined in the embedding ``ℝ^d`` as well, we obtain its gradient by projection.
"""

# ╔═╡ 92ee76b4-e132-44c7-9ab3-ef4227969fa2
grad_f(M, p) = project(M, p, -transpose(Z) * p - Z * p);

# ╔═╡ 0f71531d-b292-477d-b108-f45dc4e680ad
md"""
For the constraints this is a little more involved, since each function ``g_i = g(p)_i = p_i`` has to return its own gradient. These are again in the embedding just ``\operatorname{grad} g_i(p) = -e_i`` the ``i`` th unit vector. We can project these again onto the tangent space at ``p``:
"""

# ╔═╡ 9e7028c9-0f15-4245-a089-2670c26b3b40
grad_g(M, p) = project.(
	Ref(M), Ref(p), [[i == j ? -1.0 : 0.0 for j in 1:d] for i in 1:d]
);

# ╔═╡ cd7ab9a5-1db8-4819-952a-6322f22a0654
md"We further start in a random point:"

# ╔═╡ 72e99369-165b-494e-9acc-7719a12d9d8d
x0 = random_point(M);

# ╔═╡ aedd3604-7df6-46c5-a393-213da785b84f
md" Let's check a few things for the initial point"

# ╔═╡ 317403e3-8816-4761-b229-798710f3ed43
f(M, x0)

# ╔═╡ 00d8dd65-e57c-4810-80a3-28878a1938ea
md" How much the function g is positive"

# ╔═╡ eaad9d23-403c-4050-9b4f-097d28329591
maximum(g(M, x0))

# ╔═╡ 08468240-a353-4097-b26e-5fe14be039e3
md"""
Now as a first method we can just call the [Augmented Lagrangian Method](https://manoptjl.org/stable/solvers/augmented_Lagrangian_method/) with a simple call:
"""

# ╔═╡ eba57714-59f0-4a36-b9e5-929fe11a9e59
with_terminal() do
	@time global v1 = augmented_Lagrangian_method(
    	M, f, grad_f, x0; G=g, gradG=grad_g,
    	debug=[:Iteration, :Cost, :Stop, " | ", :Change, 50, "\n"],
	);
end

# ╔═╡ 70cb2bda-43f9-44b3-a48a-41a3b71023e3
md"Now we have both a lower function value and the point is nearly within the constraints, ... up to numerical inaccuracies"

# ╔═╡ 52e26eae-1a78-4ee5-8ad2-d7f3e85d2204
f(M, v1)

# ╔═╡ dd38b344-d168-4519-9837-69a90943c3dc
maximum( g(M, v1) )

# ╔═╡ 25ae1129-916b-4b3e-a317-74ecf41995dd
md" ## A faster Augmented Lagrangian Run "

# ╔═╡ c72709e1-7bae-4345-b29b-4ef1e791292b
md"""
Now this is a little slow, so we can modify two things, that we will directly do both – but one could also just change one of these – :

1. Gradients should be evaluated in place, so for example
"""

# ╔═╡ 717bd019-2978-4e55-a586-ed876cefa65d
grad_f!(M, X, p) = project!(M, X, p, -transpose(Z) * p - Z * p);

# ╔═╡ db35ae71-c96e-4432-a7d5-3df9f6c0f9fb
md"""
2. The constraints are currently always evaluated all together, since the function `grad_g` always returns a vector of gradients.
We first change the constraints function into a vector of functions.
We further change the gradient _both_ into a vector of gradient functions ``\operatorname{grad} g_i, i=1,\ldots,d``, _as well as_ gradients that are computed in place.
"""

# ╔═╡ fb86f597-f8af-4c98-b5b1-4db0dfc06199
g2 = [(M, p) -> -p[i] for i in 1:d];

# ╔═╡ 1d427174-57da-41d6-8577-d97d643a2142
grad_g2! = [
    (M, X, p) -> project!(M, X, p, [i == j ? -1.0 : 0.0 for j in 1:d]) for i in 1:d
];

# ╔═╡ ce8f1156-a350-4fde-bd39-b08a16b2821d
with_terminal() do
	@time global v2 = augmented_Lagrangian_method(
    	M, f, grad_f!, x0; G=g2, gradG=grad_g2!, evaluation=InplaceEvaluation(),
    	debug=[:Iteration, :Cost, :Stop, " | ", :Change, 50, "\n"],
	);
end

# ╔═╡ f6617b0f-3688-4429-974b-990e0279cb38
md"""
As a technical remark: Note that (by default) the change to [`InplaceEvaluation`](https://manoptjl.org/stable/plans/problem/#Manopt.InplaceEvaluation)s affects both the constrained solver as well as the inner solver of the subproblem in each iteration.
"""

# ╔═╡ f4b3f8c4-8cf8-493e-a6ac-ea9673609a9c
f(M, v2)

# ╔═╡ 2cbaf932-d7e8-42df-b0b4-0974a98818ca
maximum(g(M, v2))

# ╔═╡ ff01ff09-d0da-4159-8597-de2853944bcf
md" These are the very similar to the previous values but the solver took much less time and less memory allocations."

# ╔═╡ c918e591-3806-475e-8f0b-d50896d243ee
md"## Exact Penalty Method"

# ╔═╡ 6dc2aa06-7239-41c5-b296-4aa5a048444c
md"""
As a second solver, we have the [Exact Penalty Method](https://manoptjl.org/stable/solvers/exact_penalty_method/), which currenlty is available with two smoothing variants, which make an inner solver for smooth optimisationm, that is by default again [quasi Newton] possible:
[`LogarithmicSumOfExponentials`](https://manoptjl.org/stable/solvers/exact_penalty_method/#Manopt.LogarithmicSumOfExponentials)
and [`LinearQuadraticHuber`](https://manoptjl.org/stable/solvers/exact_penalty_method/#Manopt.LinearQuadraticHuber). We compare both here as well. The first smoothing technique is the default, so we can just call
"""

# ╔═╡ e9847e43-d4ef-4a90-a51d-ce527787d467
with_terminal() do
	@time global v3 = exact_penalty_method(
    	M, f, grad_f!, x0; G=g2, gradG=grad_g2!, evaluation=InplaceEvaluation(),
    	debug=[:Iteration, :Cost, :Stop, " | ", :Change, 50, "\n"],
	);
end

# ╔═╡ 5a4b79ff-6cac-4978-9a82-f561484846a8
md"We obtain a similar cost value as for the Augmented Lagrangian Solver above, but here the constraint is actually fulfilled and not just numerically “on the boundary”."

# ╔═╡ 2812f7bb-ad5b-4e1d-8dbd-239129a3facd
f(M, v3)

# ╔═╡ 5e48f24d-94e3-4ad1-ae78-cf65fe2d9caf
maximum(g(M, v3))

# ╔═╡ 6c5c8ed9-da01-4565-8b97-d8465b7f7e9f
md"""
The second smoothing technique is often beneficial, when we have a lot of constraints (in the above mentioned vectorial manner), since we can avoid several gradient evaluations for the constraint functions here. This leads to a faster iteration time.
"""

# ╔═╡ fac5894c-250e-447d-aab8-1bfab7aae78c
with_terminal() do
	@time global v4 = exact_penalty_method(
    	M, f, grad_f!, x0; G=g2, gradG=grad_g2!, evaluation=InplaceEvaluation(),
		smoothing=LinearQuadraticHuber(),
    	debug=[:Iteration, :Cost, :Stop, " | ", :Change, 50, "\n"],
	);
end

# ╔═╡ f0356469-5bdf-49b4-b7a9-83cc987cb368
md"""
For the result we see the same behaviour as for the other smoothing.
"""

# ╔═╡ bd276d17-2df0-4732-ae94-340f1e4a54f9
f(M, v4)

# ╔═╡ 8a3ed988-89c7-49d7-ac9f-301ff5f246ef
maximum(g(M, v4))

# ╔═╡ 51c4ac18-d829-4297-b7c9-3058fdd10555
md"""
## Comparing to the unconstraint solver

We can compare this to the _global_ optimum on the sphere, which is the unconstraint optimisation problem; we can just use Quasi Newton.

Note that this is much faster, since every iteration of the algorithms above does a quasi-Newton call as well.
"""

# ╔═╡ 70fd7a56-ebce-43b9-b75e-f47c7a277a07
with_terminal() do
	@time global w1 = quasi_Newton(
		M, f, grad_f!, x0; evaluation=InplaceEvaluation()
	);
end

# ╔═╡ 7062ba30-6f7e-42f0-9396-ab2821a22f52
f(M, w1)

# ╔═╡ 4c82e8b7-4fb8-4e4d-8a95-64b42471dc13
md" But for sure here the constraints here are not fulfilled and we have veru positive entries in ``g(w_1)``"

# ╔═╡ e56fc1dc-4146-42a2-89e3-0566eb0d16f5
maximum(g(M, w1))

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

# ╔═╡ 39dcf482-5b7c-437d-b000-b0766a1e3fc7


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Manifolds = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
Manopt = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Distributions = "~0.25.79"
Manifolds = "~0.8.40"
Manopt = "~0.3.46"
PlutoUI = "~0.7.48"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "00f2f33c06007e33258fb4ee1bf8c35feef33d22"

[[deps.AbstractFFTs]]
deps = ["ChainRulesCore", "LinearAlgebra"]
git-tree-sha1 = "69f7020bd72f069c219b5e8c236c1fa90d2cb409"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.2.1"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "195c5505521008abea5aee4f96930717958eac6f"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "62e51b39331de8911e4a7ff6f5aaf38a5f4cc0ae"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.2.0"

[[deps.ArrayInterfaceCore]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "c46fb7dd1d8ca1d213ba25848a5ec4e47a1a1b08"
uuid = "30b0a656-2188-435a-8636-2ec0e6a096e2"
version = "0.1.26"

[[deps.ArrayInterfaceStaticArraysCore]]
deps = ["Adapt", "ArrayInterfaceCore", "LinearAlgebra", "StaticArraysCore"]
git-tree-sha1 = "93c8ba53d8d26e124a5a8d4ec914c3a16e6a0970"
uuid = "dd5226c6-a4d4-4bc7-8575-46859f9c95b9"
version = "0.1.3"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "e7ff6cadf743c098e08fca25c91103ee4303c9bb"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.15.6"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "38f7a08f19d8810338d4f5085211c7dfa5d5bdd8"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.4"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random"]
git-tree-sha1 = "1fd869cc3875b57347f7027521f561cf46d1fcd8"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.19.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "d08c20eef1f2cbc6e60fd3612ac4340b89fea322"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.9"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "aaabba4ce1b7f8a9b34c015053d3b1edf60fa49c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.4.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "3c8de95b4e932d76ec8960e12d681eba580e9674"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.8"

[[deps.DataAPI]]
git-tree-sha1 = "e08915633fcb3ea83bf9d6126292e5bc5c739922"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.13.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DensityInterface]]
deps = ["InverseFunctions", "Test"]
git-tree-sha1 = "80c3e8639e3353e5d2912fb3a1916b8455e2494b"
uuid = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
version = "0.4.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["ChainRulesCore", "DensityInterface", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Test"]
git-tree-sha1 = "a7756d098cbabec6b3ac44f369f74915e8cfd70a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.79"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "c36550cb29cbe373e95b3f40486b9a4148f89ffd"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.2"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.Einsum]]
deps = ["Compat"]
git-tree-sha1 = "4a6b3eee0161c89700b6c1949feae8b851da5494"
uuid = "b7d42ee7-0b51-5a75-98ca-779d3107e4c0"
version = "0.4.1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "802bfc139833d2ba893dd9e62ba1767c88d708ae"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.13.5"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "6872f5ec8fd1a38880f027a26739d42dcda6691f"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.2"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "ba2d094a88b6b287bd25cfa86f301e7693ffae2f"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.7.4"

[[deps.HybridArrays]]
deps = ["LinearAlgebra", "Requires", "StaticArrays"]
git-tree-sha1 = "0de633a951f8b5bd32febc373588517aa9f2f482"
uuid = "1baab800-613f-4b0a-84e4-9cd3431bfbb9"
version = "0.4.13"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions", "Test"]
git-tree-sha1 = "709d864e3ed6e3545230601f94e11ebc65994641"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.11"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[deps.Inflate]]
git-tree-sha1 = "5cd07aab533df5170988219191dfad0519391428"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.3"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "49510dfcb407e572524ba94aeae2fced1f3feb0f"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.8"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[deps.Kronecker]]
deps = ["LinearAlgebra", "NamedDims", "SparseArrays", "StatsBase"]
git-tree-sha1 = "78d9909daf659c901ae6c7b9de7861ba45a743f4"
uuid = "2c470bb0-bcc8-11e8-3dad-c9649493f05e"
version = "0.5.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearMaps]]
deps = ["LinearAlgebra", "SparseArrays", "Statistics"]
git-tree-sha1 = "d1b46faefb7c2f48fdec69e6f3cc34857769bc15"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.8.0"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "946607f84feb96220f480e0422d3484c49c00239"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.19"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.Manifolds]]
deps = ["Colors", "Distributions", "Einsum", "Graphs", "HybridArrays", "Kronecker", "LinearAlgebra", "ManifoldsBase", "Markdown", "MatrixEquations", "Quaternions", "Random", "RecipesBase", "RecursiveArrayTools", "Requires", "SimpleWeightedGraphs", "SpecialFunctions", "StaticArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "52ae2d59b106d9557243ddc1cdb9aea0a7081099"
uuid = "1cead3c2-87b3-11e9-0ccd-23c62b72b94e"
version = "0.8.40"

[[deps.ManifoldsBase]]
deps = ["LinearAlgebra", "Markdown"]
git-tree-sha1 = "0a17f21f8a544642c276111513deaf9bbf391217"
uuid = "3362f125-f0bb-47a3-aa74-596ffd7ef2fb"
version = "0.13.26"

[[deps.Manopt]]
deps = ["ColorSchemes", "ColorTypes", "Colors", "DataStructures", "Dates", "LinearAlgebra", "ManifoldsBase", "Markdown", "Printf", "Random", "Requires", "SparseArrays", "StaticArrays", "Statistics", "Test"]
git-tree-sha1 = "b1fbcf0c5ec31cb04671abd5baccba9d3dfcbc7e"
uuid = "0fc0a36d-df90-57f3-8f93-d78a9fc72bb5"
version = "0.3.46"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MatrixEquations]]
deps = ["LinearAlgebra", "LinearMaps"]
git-tree-sha1 = "3b284e9c98f645232f9cf07d4118093801729d43"
uuid = "99c1a7ee-ab34-5fd5-8076-27c950a045f4"
version = "2.2.2"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.0+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.2.1"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "a7c3d1da1189a1c2fe843a3bfa04d18d20eb3211"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.1"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "cb8ebcee2b4e07b72befb9def593baef8aa12f07"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.50"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "cf494dca75a69712a72b80bc48f59dcf3dea63ec"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.16"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "b64719e8b4504983c7fca6cc9db3ebc8acc2a4d6"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.8.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "efc140104e6d0ae3e7e30d56c98c4a927154d684"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.48"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "97aa253e65b784fd13e83774cadc95b38011d734"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.6.0"

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "fcebf40de9a04c58da5073ec09c1c1e95944c79b"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.6.1"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
deps = ["SnoopPrecompile"]
git-tree-sha1 = "d12e612bba40d189cead6ff857ddb67bd2e6a387"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.1"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterfaceCore", "ArrayInterfaceStaticArraysCore", "ChainRulesCore", "DocStringExtensions", "FillArrays", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "StaticArraysCore", "Statistics", "Tables", "ZygoteRules"]
git-tree-sha1 = "4c7a6462350942da60ea5749afd7cea58017301f"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.32.2"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays", "Test"]
git-tree-sha1 = "a6f404cc44d3d3b28c793ec0eb59af709d827e4e"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.2.1"

[[deps.SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "d75bda01f8c31ebb72df80a46c88b25d1c79c56d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.1.7"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "4e051b85454b4e4f66e6a6b7bdc452ad9da3dcf6"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.10"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["ChainRulesCore", "HypergeometricFunctions", "InverseFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "5783b877201a82fc0014cbf381e7e6eb130473a4"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.0.1"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "c79322d36826aa2f4fd8ecfa96ddb47b174ac78d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.1"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[deps.URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─23c48862-6984-11ed-0a6d-9f6c98ae7134
# ╟─d7bbeb6b-709a-451b-9eb2-2eb7245dbe03
# ╟─7c0b460e-34e4-4d7b-be51-71f4a38f28e3
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
# ╟─0ee60004-e7e5-4c9a-8f1d-e493217f11be
# ╠═8be00c3b-4385-449b-b58f-3d3ef972c3c3
# ╠═950dab06-b89e-41c7-9d81-3e9f3fb51b4d
# ╟─6218866b-18b5-47f7-98a6-6e1192cb1c24
# ╠═92ee76b4-e132-44c7-9ab3-ef4227969fa2
# ╟─0f71531d-b292-477d-b108-f45dc4e680ad
# ╠═9e7028c9-0f15-4245-a089-2670c26b3b40
# ╟─cd7ab9a5-1db8-4819-952a-6322f22a0654
# ╠═72e99369-165b-494e-9acc-7719a12d9d8d
# ╟─aedd3604-7df6-46c5-a393-213da785b84f
# ╠═317403e3-8816-4761-b229-798710f3ed43
# ╟─00d8dd65-e57c-4810-80a3-28878a1938ea
# ╠═eaad9d23-403c-4050-9b4f-097d28329591
# ╟─08468240-a353-4097-b26e-5fe14be039e3
# ╠═eba57714-59f0-4a36-b9e5-929fe11a9e59
# ╟─70cb2bda-43f9-44b3-a48a-41a3b71023e3
# ╠═52e26eae-1a78-4ee5-8ad2-d7f3e85d2204
# ╠═dd38b344-d168-4519-9837-69a90943c3dc
# ╟─25ae1129-916b-4b3e-a317-74ecf41995dd
# ╟─c72709e1-7bae-4345-b29b-4ef1e791292b
# ╠═717bd019-2978-4e55-a586-ed876cefa65d
# ╟─db35ae71-c96e-4432-a7d5-3df9f6c0f9fb
# ╠═fb86f597-f8af-4c98-b5b1-4db0dfc06199
# ╠═1d427174-57da-41d6-8577-d97d643a2142
# ╠═ce8f1156-a350-4fde-bd39-b08a16b2821d
# ╟─f6617b0f-3688-4429-974b-990e0279cb38
# ╠═f4b3f8c4-8cf8-493e-a6ac-ea9673609a9c
# ╠═2cbaf932-d7e8-42df-b0b4-0974a98818ca
# ╟─ff01ff09-d0da-4159-8597-de2853944bcf
# ╟─c918e591-3806-475e-8f0b-d50896d243ee
# ╟─6dc2aa06-7239-41c5-b296-4aa5a048444c
# ╠═e9847e43-d4ef-4a90-a51d-ce527787d467
# ╟─5a4b79ff-6cac-4978-9a82-f561484846a8
# ╠═2812f7bb-ad5b-4e1d-8dbd-239129a3facd
# ╠═5e48f24d-94e3-4ad1-ae78-cf65fe2d9caf
# ╟─6c5c8ed9-da01-4565-8b97-d8465b7f7e9f
# ╠═fac5894c-250e-447d-aab8-1bfab7aae78c
# ╟─f0356469-5bdf-49b4-b7a9-83cc987cb368
# ╠═bd276d17-2df0-4732-ae94-340f1e4a54f9
# ╠═8a3ed988-89c7-49d7-ac9f-301ff5f246ef
# ╟─51c4ac18-d829-4297-b7c9-3058fdd10555
# ╠═70fd7a56-ebce-43b9-b75e-f47c7a277a07
# ╠═7062ba30-6f7e-42f0-9396-ab2821a22f52
# ╟─4c82e8b7-4fb8-4e4d-8a95-64b42471dc13
# ╠═e56fc1dc-4146-42a2-89e3-0566eb0d16f5
# ╟─78d055e8-d5c8-4cdf-a706-3089368397bd
# ╠═39dcf482-5b7c-437d-b000-b0766a1e3fc7
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
