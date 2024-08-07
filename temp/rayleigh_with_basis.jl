### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ d9b8ed2e-48e5-11ef-28d1-1370b29e394d
using Pkg; Pkg.activate();

# ╔═╡ 8c7093a3-bad5-4513-9407-bda76668752e
begin
    using LinearAlgebra
    #using RandomMatrices
    using Manopt
    using Manifolds
    using PlutoUI
	using Random
end;

# ╔═╡ c9598db9-9e9a-4882-838b-ddc753fa4b1f
begin

    #Random.seed!(42)

    n = 3

    # Eigenwerte vorgeben und orthogonal transformieren:

    #Diagonalmatrix mit Eigenwerten:
    D = Matrix{Float64}(I, n, n)
    for i in 1:n
        D[i, i] = i
    end

    # Zufällige orthogonale Matrix:
    #O = rand(Haar(1),n)

    # Transformation von D:
    #A = O' * D * O
    A = D
    for i in 1:n
		for j in 1:n
			if i != j
				A[i,j] = 0.1
			end
		end
	end
	println(eigvals(A))
end;

# ╔═╡ bfadf38e-46e8-4f63-a905-dc150925102b
function solve(mp, s, k)
	E = get_vectorbundle(mp)
    M = get_manifold(mp)
    o = get_objective(mp)
    F_p = get_bundle_map(M, E, o, s.p)
    Fprime_p = get_derivative(M, E, o, s.p)
    connection = get_connection_map(mp, F_p)

	covariant_derivative = Fprime_p - connection#*p'(F_p)

	# basis representation of covariant_derivative
	B = get_basis(M, s.p, DefaultOrthogonalBasis())
	b = get_vectors(M, s.p, B)

	newton_matrix = Matrix{Float64}(I, n-1, n-1)
	for i in 1:n-1
		for j in 1:n-1
			newton_matrix[i,j] = (covariant_derivative*b[j])'*b[i]
		end
	end

	# basis representation of F_p
	#f = get_coordinates(M, s.p, F_p, DefaultOrthogonalBasis()) funktioniert nicht, da wir hier eine duale Paarung brauchen
	f = zeros(n-1)
	for i in 1:n-1
		f[i] = F_p[E,:vector]*b[i]
	end

	# solve linear system -> get coefficients of tangent vector
	deltax_basis = (newton_matrix)\(-f)

	deltax = get_vector(M, s.p, deltax_basis, B)
	return deltax
end

# ╔═╡ f46e8e2a-9f01-42d3-af27-cb54981dd189
function f(M, p)
    return p' * A * p
end

# ╔═╡ 71c8e8a8-8254-45e4-8837-58d096aee7d8
function f_prime(M, p)
    return ArrayPartition(p, (2.0 * A * p)')
end

# ╔═╡ 29dac6c4-aa14-433c-9972-dfd3e95a86e1
function f_second_derivative(M, p)
    return 2.0 * A
end

# ╔═╡ 72635088-2096-437f-9f0b-9bbd63b77dbf
function connection_map(E, q)
    return q[E, :vector] * bundle_projection(E, q) * Matrix{Float64}(I, n, n)
end

# ╔═╡ f25decc6-7fc0-46a6-8b22-cbcc913813fc
M = Sphere(n - 1)

# ╔═╡ 7d3696e4-195c-4cc9-b0da-be648fb6a870
E = TangentBundle(M)

# ╔═╡ 00255503-1b2f-4a58-83cf-8027f55be5ff
begin
#p = [zeros(n - 1)..., 1]
Random.seed!(40)
p=rand(M)
println(p)
end;

# ╔═╡ 6e0d91a9-83b7-4930-bcc4-86620088a1b7
p_res = vectorbundle_newton(M, E, f_prime, f_second_derivative, connection_map, p;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=StopAfterIteration(15),
	retraction_method=ProjectionRetraction(),
	debug=[:Iteration, (:Change, "Change: %1.8e"), 1, "\n", :Stop]
)

# ╔═╡ bc89b23b-7a44-4b55-9e32-8ef3f8a4b323
f(M, p_res)

# ╔═╡ Cell order:
# ╠═d9b8ed2e-48e5-11ef-28d1-1370b29e394d
# ╠═8c7093a3-bad5-4513-9407-bda76668752e
# ╠═bfadf38e-46e8-4f63-a905-dc150925102b
# ╠═c9598db9-9e9a-4882-838b-ddc753fa4b1f
# ╠═f46e8e2a-9f01-42d3-af27-cb54981dd189
# ╠═71c8e8a8-8254-45e4-8837-58d096aee7d8
# ╠═29dac6c4-aa14-433c-9972-dfd3e95a86e1
# ╠═72635088-2096-437f-9f0b-9bbd63b77dbf
# ╠═f25decc6-7fc0-46a6-8b22-cbcc913813fc
# ╠═7d3696e4-195c-4cc9-b0da-be648fb6a870
# ╠═00255503-1b2f-4a58-83cf-8027f55be5ff
# ╠═6e0d91a9-83b7-4930-bcc4-86620088a1b7
# ╠═bc89b23b-7a44-4b55-9e32-8ef3f8a4b323
