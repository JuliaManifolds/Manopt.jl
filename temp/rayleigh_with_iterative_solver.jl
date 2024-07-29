### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 3534e21a-4dbf-11ef-390f-71c07e69d7cc
using Pkg; Pkg.activate();

# ╔═╡ fcf9017e-e85b-4078-957d-04814303606a
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using Random
	using IterativeSolvers
end

# ╔═╡ d95cc388-48d7-467e-9aed-2610edefa6d2
function solve(mp, s, k)
	E = get_vectorbundle(mp)
    M = get_manifold(mp)
    o = get_objective(mp)
    F_p = get_bundle_map(M, E, o, s.p)
    Fprime_p = get_derivative(M, E, o, s.p)
	
    connection = get_connection_map(mp, F_p)
	covariant_derivative = Fprime_p - connection
	
	deltax = cg(covariant_derivative, -F_p[E, :vector]')
	println(deltax)
	return deltax
end

# ╔═╡ ca5bf240-de50-4b05-b4c7-561034edb0e3
begin

    #Random.seed!(42)

    n = 5

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

# ╔═╡ 5d5edb4b-e9c2-4c12-ba24-6e6ac5b25a8b
function f(M, p)
    return p' * A * p
end

# ╔═╡ 288fe34c-3940-4870-80a3-a6c4c509787d
function f_prime(M, p)
    return ArrayPartition(p, (2.0 * A * p)')
end

# ╔═╡ dae48164-51be-4d9e-bb3b-d574e45f170c
function f_second_derivative(M, p)
    return 2.0 * A
end

# ╔═╡ 6a98891e-92ca-4b4c-930d-8c2a9de87b38
function connection_map(E, q)
    return q[E, :vector] * bundle_projection(E, q) * Matrix{Float64}(I, n, n)
end

# ╔═╡ a9160f3e-e65b-47c1-aa27-dcfb3ef7d589
M = Sphere(n - 1)

# ╔═╡ 31059363-9ba3-4a53-b016-4ffd71eb5d66
E = TangentBundle(M)

# ╔═╡ 157196d9-d81d-4473-a6b3-cd41a1c5c014
begin
#p = [zeros(n - 1)..., 1]
Random.seed!(40)
p=rand(M)
println(p)
end;

# ╔═╡ 377b5ae8-379d-4e79-9812-d51cad263969
p_res = vectorbundle_newton(M, E, f_prime, f_second_derivative, connection_map, p;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=StopAfterIteration(15),
	retraction_method=ProjectionRetraction(),
	debug=[:Iteration, :Change, 1, "\n", :Stop]
)

# ╔═╡ 09a6eabd-22a8-41c1-8252-7d26fb237087
f(M, p_res)

# ╔═╡ Cell order:
# ╠═3534e21a-4dbf-11ef-390f-71c07e69d7cc
# ╠═fcf9017e-e85b-4078-957d-04814303606a
# ╠═d95cc388-48d7-467e-9aed-2610edefa6d2
# ╠═ca5bf240-de50-4b05-b4c7-561034edb0e3
# ╠═5d5edb4b-e9c2-4c12-ba24-6e6ac5b25a8b
# ╠═288fe34c-3940-4870-80a3-a6c4c509787d
# ╠═dae48164-51be-4d9e-bb3b-d574e45f170c
# ╠═6a98891e-92ca-4b4c-930d-8c2a9de87b38
# ╠═a9160f3e-e65b-47c1-aa27-dcfb3ef7d589
# ╠═31059363-9ba3-4a53-b016-4ffd71eb5d66
# ╠═157196d9-d81d-4473-a6b3-cd41a1c5c014
# ╠═377b5ae8-379d-4e79-9812-d51cad263969
# ╠═09a6eabd-22a8-41c1-8252-7d26fb237087
