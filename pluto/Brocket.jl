### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 6e0fc36e-db37-11eb-153a-a79fe149b82a
using Manifolds, Manopt, LinearAlgebra, Random

# ╔═╡ 9b5cf8fa-b358-4966-baf9-8a6abc86bd5d
md"
# The Brocket function

this example considers the Brocket function on the Stiefel Manifold, i.e. for a given symmetric matrix ``A \in \mathbb R^{n\times n}`` and a positive integer ``k < n`` we consider the cost function

```math
F(p) = \operatorname{tr}(p^\mathrm{T}ApN),\qquad p \in \mathrm{St}(n,k),
```
where ``\mathrm{St}(n,k)`` denotes the [Stiefel](https://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/stiefel.html#Manifolds.Stiefel) manifold and ``N=\operatorname{diag}(k,k-1,\ldots,1)``.
"

# ╔═╡ 4f6af949-9e3e-4b58-abc8-a8c74ff109ae
begin
    Random.seed!(42)
    n = 10
    k = 5
    nothing
end

# ╔═╡ 8ad35f54-d774-47a4-8dd0-d62ad05be20b
md"## Gradient of the cost function"

# ╔═╡ 944f7644-2428-4f79-885b-54ff326eb331
begin
    struct GradF!
        A::Matrix{Float64}
        N::Diagonal{Float64,Vector{Float64}}
    end
    function (gradF!::GradF!)(::Stiefel, X, p)
        Ap = gradF!.A * p
        pTAp = p' * Ap
        print(size(X))
        print(size(
            2 .* Ap * gradF!.N, #.- p * pTAp * gradF!.N .- p * gradF!.N * pTAp
        ))
        copyto!(X, 2 .* Ap * gradF!.N .- p * pTAp * gradF!.N .- p * gradF!.N * pTAp)
        return X
    end
end

# ╔═╡ c10796ce-1f12-4d70-b643-9c0b44caa853
begin
    A = randn(n, n)
    A = (A + A') / 2
end

# ╔═╡ de711f30-6c88-4eb8-bd31-2dcfdcfb6ab6
F(::Stiefel, p) = tr((p' * A * p) * Diagonal(k:-1:1))

# ╔═╡ fe4155cc-ff04-4d56-a254-6d642a5c570b
gradF! = GradF!(A, Diagonal(Float64.(collect(k:-1:1))))

# ╔═╡ bdb2e4b0-7a14-4bd4-83ee-72b31d509196
M = Stiefel(n, k)

# ╔═╡ e60da1b8-04b1-45fd-9926-f326f69e5fa6
x0 = copy(M, A)

# ╔═╡ 3198c2b2-3c38-46f6-a6a3-cecc5ae4e1e0
begin
    X = zero_vector(M, x0)
    gradF!(M, X, x0)
end

# ╔═╡ c95073cc-7447-4217-a4e7-f477f76fd292
quasi_Newton(
    M,
    F,
    gradF!,
    x0;
    memory_size=m,
    vector_transport_method=ProjectionTransport(),
    retraction_method=QRRetraction(),
    stopping_criterion=StopWhenGradientNormLess(norm(M, x0, gradF(M, x0)) * 10^(-6)),
    cautious_update=true,
    evaluation=MutatingEvaluation(),
    #    debug = [:Iteration," ", :Cost, " ", DebugGradientNorm(), "\n", 10],
)

# ╔═╡ Cell order:
# ╠═9b5cf8fa-b358-4966-baf9-8a6abc86bd5d
# ╠═6e0fc36e-db37-11eb-153a-a79fe149b82a
# ╠═4f6af949-9e3e-4b58-abc8-a8c74ff109ae
# ╠═8ad35f54-d774-47a4-8dd0-d62ad05be20b
# ╠═944f7644-2428-4f79-885b-54ff326eb331
# ╠═de711f30-6c88-4eb8-bd31-2dcfdcfb6ab6
# ╠═c10796ce-1f12-4d70-b643-9c0b44caa853
# ╠═fe4155cc-ff04-4d56-a254-6d642a5c570b
# ╠═bdb2e4b0-7a14-4bd4-83ee-72b31d509196
# ╠═e60da1b8-04b1-45fd-9926-f326f69e5fa6
# ╠═3198c2b2-3c38-46f6-a6a3-cecc5ae4e1e0
# ╠═c95073cc-7447-4217-a4e7-f477f76fd292
