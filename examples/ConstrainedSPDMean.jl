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
md""" # Constraint Mean using Frank–Wolfe."""

# ╔═╡ 58594aa5-1dc8-4193-914f-98ab4bfcdc03
M = SymmetricPositiveDefinite(3)

# ╔═╡ 5467ca6d-7bf3-42db-9a52-93464d809f6c
md"""this example recreates the Example from Section 4.2.1 by Weber and Sra, 2017"""

# ╔═╡ 701ace87-ef6e-42a6-9e81-563c3abc55b4
function FW_oracle(M::SymmetricPositiveDefinite, L, U, S, X)
	e = eigen(Symmetric(X))
    U = e.vectors
    Ud = max.(e.values, floatmin(eltype(e.values)))
    Dsqrt = Diagonal(sqrt.(x))
    DsqrtInv = Diagonal(1 ./ sqrt.(x))
    XSqrt = Symmetric(U * Ssqrt * transpose(U))
    XSqrtInv = Symmetric(U * SsqrtInv * transpose(U))

	e2 = eigen(Xsqrt*S*Xsqrt)
	D2 = Diagonal(1.0 .* (e2.values .< 0))
	Q = e2.vectors
	
	Uprime = Q'*XSqrtInv*U*XSqrtInv*Q
	Lprime = Q'*XSqrtInv*L*XSqrtInv*Q
	P = cholesky(Uprime - Lprime)
	z =  P'*D*P+Lprima
	return XSqrt*Q*z*Q'*XSqrt
end

# ╔═╡ 3d369862-dc1c-4e06-acde-9eb1baa1a425


# ╔═╡ Cell order:
# ╠═285f33c1-e8ed-45d4-ba33-9e5eb17ce422
# ╠═e2dd0dd2-0444-11ed-1e55-5b3126a23f62
# ╠═58594aa5-1dc8-4193-914f-98ab4bfcdc03
# ╠═5467ca6d-7bf3-42db-9a52-93464d809f6c
# ╠═701ace87-ef6e-42a6-9e81-563c3abc55b4
# ╠═3d369862-dc1c-4e06-acde-9eb1baa1a425
