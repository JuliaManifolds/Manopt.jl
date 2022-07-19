### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ e2dd0dd2-0444-11ed-1e55-5b3126a23f62
begin
	using Pkg
	Pkg.activate()#use branch in dev mode for now
	using Manopt, Manifolds, LinearAlgebra
end

# ╔═╡ 285f33c1-e8ed-45d4-ba33-9e5eb17ce422
md""" # Constraint Mean using Frank–Wolfe

This example illustrates the use of the Riemannian Frank-Wolfe algorithm by reimplementing the example given in [^WeberSra2022], Section 4.2, that is the geometric mean (in Alg. 3 GM).

[^WeberSra2022]:
    > M. Weber, S. Sra: _Riemannian Optimization via Frank-Wolfe Methods_,
    > Math. Prog., 2022, doi: [10.1007/s10107-022-01840-5](https://doi.org/10.1007/s10107-022-01840-5).
"""

# ╔═╡ c355af21-9912-4de2-9a3e-21c3dd7d2fdd
md"""
We first define some variables.
* `n` is the dimension of our SPD matrices
* `N` is the number of data matrices we will generate
"""

# ╔═╡ 397f4aa4-bd8b-4f98-9c5a-c3514e50ef37
begin
	# dimension of the SPD matrices
	n = 50
	# number of data points
	N = 100	
end

# ╔═╡ 8e59bb69-e073-40cc-a304-d1256463dcbb
md"""
We now generate the $N data points and $N random weights that sum up to 1.
"""

# ╔═╡ 1a96dee3-f2b2-42a4-ac5a-2af399e86463
@doc raw"""
	harmonic_mean(pts,w=1/N.*ones(N))

for `N` points `pts` on the SPD matrices, this computes the weighted harmonic mean, i.e.

```math
	\biggl( \sum_{i=1}^N w_i p_i^{-1} \biggr)^{-1}
```
"""
function harmonic_mean(pts, w=1/length(pts) .* ones(length(pts)))
	return inv(sum( [wi*inv(pi) for (wi,pi) in zip(w,pts)]  ))
end

# ╔═╡ a56c69d9-768b-4ce5-99d7-986820b8b53c
@doc raw"""
	arithmetic_mean(pts,w=1/N.*ones(N))

for `N` points `pts` on the SPD matrices, this computes the weighted arithmetic mean, i.e.

```math
	\sum_{i=1}^N w_i p_i
```
"""
function arithmetic_mean(pts, w=1/length(pts) .* ones(length(pts)))
	return sum( [wi*pi for (wi,pi) in zip(w,pts)]  )
end

# ╔═╡ 58594aa5-1dc8-4193-914f-98ab4bfcdc03
M = SymmetricPositiveDefinite(n)

# ╔═╡ 09e304f8-b153-4076-81d7-a3a01644d65a
begin
	data = [random_point(M) for _ ∈ 1:N]
	weights = rand(N)
	weights ./= sum(weights)
end

# ╔═╡ b2fdb060-31c6-4d2c-9e04-e5fa0caab5d0
function weighted_mean_cost(M, p)
	return sum([ wi*distance(M,p,di)^2 for (wi,di) in zip(weights,data) ] )
end

# ╔═╡ e5fc5216-5aab-4638-9444-02dd9b1cb4e3
function grad_weighted_mean_cost(M, p)
	return sum([ wi*grad_distance(M,p,di) for (wi,di) in zip(weights,data) ] )
end

# ╔═╡ 701ace87-ef6e-42a6-9e81-563c3abc55b4
@doc raw"""
	FW_oracle(M::SymmetricPositiveDefinite, L, U, p, X)

Given a lower bound `L` and an upper bound `U` (spd matrices),
a point `p` and a tangent vector (e.g. the gradient at p),
this oracle solves the subproblem related to the constraint problem

```math
	\operatorname{arg\,min}_{L\preceq q \preceq U} ⟨ X, \log_p q⟩
```
which has a closed form solution, cf. (38) in Weber & Sra.

"""
function FW_oracle(M::SymmetricPositiveDefinite, L, U, p, X)
	e = eigen(Symmetric(p))
    U = e.vectors
    Ud = max.(e.values, floatmin(eltype(e.values)))
    Dsqrt = Diagonal(sqrt.(Ud))
    DsqrtInv = Diagonal(1 ./ sqrt.(Ud))
    pSqrt = Symmetric(U * Dsqrt * transpose(U))
    pSqrtInv = Symmetric(U * DsqrtInv * transpose(U))

	e2 = eigen(pSqrt*X*pSqrt)
	D2 = Diagonal(1.0 .* (e2.values .< 0))
	Q = e2.vectors
	
	Uprime = Q'*pSqrtInv*U*pSqrtInv*Q
	Lprime = Q'*pSqrtInv*L*pSqrtInv*Q
	P = cholesky(Uprime - Lprime)
	z =  P'*D*P+Lprima
	return pSqrt*Q*z*Q'*pSqrt
end

# ╔═╡ 41ad71e7-708f-42e9-a92b-902c6324215f
H = harmonic_mean(data, weights);

# ╔═╡ d77613d4-ede8-44c7-bc3f-46ab4c828b90
A = arithmetic_mean(data, weights);

# ╔═╡ 17ec6a97-d7af-4b35-b388-523537e88a0f
special_oracle(M,p,X) = FW_oracle(M, H, A, p, X)

# ╔═╡ 3d369862-dc1c-4e06-acde-9eb1baa1a425
Frank_Wolfe_algorithm(M, weighted_mean_cost, grad_weighted_mean_cost, data[1]; subtask=special_oracle)

# ╔═╡ Cell order:
# ╠═285f33c1-e8ed-45d4-ba33-9e5eb17ce422
# ╠═e2dd0dd2-0444-11ed-1e55-5b3126a23f62
# ╠═c355af21-9912-4de2-9a3e-21c3dd7d2fdd
# ╠═397f4aa4-bd8b-4f98-9c5a-c3514e50ef37
# ╟─8e59bb69-e073-40cc-a304-d1256463dcbb
# ╠═09e304f8-b153-4076-81d7-a3a01644d65a
# ╠═1a96dee3-f2b2-42a4-ac5a-2af399e86463
# ╠═a56c69d9-768b-4ce5-99d7-986820b8b53c
# ╠═b2fdb060-31c6-4d2c-9e04-e5fa0caab5d0
# ╠═e5fc5216-5aab-4638-9444-02dd9b1cb4e3
# ╠═58594aa5-1dc8-4193-914f-98ab4bfcdc03
# ╠═701ace87-ef6e-42a6-9e81-563c3abc55b4
# ╠═41ad71e7-708f-42e9-a92b-902c6324215f
# ╠═d77613d4-ede8-44c7-bc3f-46ab4c828b90
# ╠═17ec6a97-d7af-4b35-b388-523537e88a0f
# ╠═3d369862-dc1c-4e06-acde-9eb1baa1a425
