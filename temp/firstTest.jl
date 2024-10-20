### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ 9c6adf5c-d521-4227-bf44-64f317bccba5
using Pkg; Pkg.activate();

# ╔═╡ af19557a-9a39-4861-bc89-5fa932c26364
begin
    using LinearAlgebra
    #using RandomMatrices
    using Manopt
    using Manifolds
	using Random
    using PlutoUI
end;

# ╔═╡ 8d501a3e-2835-469b-a29c-2a314406ae5d
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

# ╔═╡ 9910fbe2-d9b9-48b6-96da-7db072ca6b71
function f(M, p)
    return p' * A * p
end

# ╔═╡ 256f6350-3c69-4eec-bcea-6e8617ede1a2
function f_prime(M, p)
    return ArrayPartition(p, (2.0 * A * p)')
end

# ╔═╡ ae45b8f3-1984-4002-8fa8-e16c260d8a22
function f_second_derivative(M, p)
    return 2.0 * A
end

# ╔═╡ 743ef6ed-41d8-45a9-973b-b5b4867dd043
function connection_map(E, q)
    return q[E, :vector] * bundle_projection(E, q) * Matrix{Float64}(I, n, n)
end

# ╔═╡ 6a07e0c2-ba0f-4d23-8e98-3f86138298be
begin
    import Manopt.get_submersion
    function get_submersion(M::Sphere, p)
        return p'p - 1
    end
end

# ╔═╡ d1e0f7aa-f1cc-4ef6-90de-a8dbc0aec0b4
begin
    import Manopt.get_submersion_derivative
    function get_submersion_derivative(M::Sphere, p)
        return p'
    end
end

# ╔═╡ 2f11d1ca-4434-11ef-2ca7-f52cd378d278
function solve(mp, s, k)
    #(E, X, F, Fprime, rS, x)
    E = get_vectorbundle(mp)
    M = get_manifold(mp)
    o = get_objective(mp)
    F_p = get_bundle_map(M, E, o, s.p)
    Fprime_p = get_derivative(M, E, o, s.p)
    connection = get_connection_map(mp, F_p)
    submersion_derivative = get_submersion_derivative(M, s.p)

    #covDeriv = connection(F, Fprime)
    covariant_derivative = Fprime_p - connection#*p'(F_p)
    #newton_matrix = vcat(covariant_derivative, s.p')
    newton_matrix = vcat(covariant_derivative, submersion_derivative)
    #ptilde = vcat(s.p, 0)
    ptilde = vcat(submersion_derivative', 0)
    newton_matrix = hcat(newton_matrix, ptilde)

	#println("is_same = ", s.is_same)
	if (s.is_same == true)
   	 	rhs = -1.0 * F_p[E, :vector]'
    	rhs = vcat(rhs, 0)
    	deltaxlambda = newton_matrix \ rhs
		#println("Hallo")
	else
		rhs = ((1-s.stepsize.alpha)*F_p[E, :vector]' - vector_transport_to(M, s.p, get_bundle_map(M, E, o, s.p_trial)[E, :vector]', s.p_trial,  ProjectionTransport()))
    	rhs = vcat(rhs, 0)
    	deltaxlambda = newton_matrix \ rhs
		#println("Hallo")
	end
    return deltaxlambda[1:n]
end

# ╔═╡ 5b2750cc-92a2-404a-bc76-c7d540021c5d
obj = VectorbundleObjective(f_prime, f_second_derivative, connection_map)

# ╔═╡ aa7d9421-ae18-4564-bbc2-a8d437f9248c
M = Sphere(n - 1)

# ╔═╡ 71c472a3-e500-433d-ba39-54c0e6f74174
E = TangentBundle(M)

# ╔═╡ 2913b52c-950f-4027-b552-b07f1cca1b76
begin
#p = [zeros(n - 1)..., 1]
Random.seed!(40)
p=rand(M)
p_trial = p
println(p)
end;

# ╔═╡ 6d416d63-230b-4ee7-8e49-fb0a60a6778a
#problem = VectorbundleManoptProblem(M, E, obj)

# ╔═╡ 1583e00e-85ec-4faa-9429-a3bcd804aa36
#state = VectorbundleNewtonState(M, E, f_prime, p, solve, AllocatingEvaluation())

# ╔═╡ 6656986c-ebea-4850-9daa-8e453fda9bac
#solve!(problem, state)

# ╔═╡ 5d019884-80a4-46b2-a2d0-bf0f2771db20
#f(M, state.p)

# ╔═╡ 9b49618b-2034-4ff2-9e8d-90d5127ae411
md"""
Instead of creating objective, problem, and state manually, we now also have a high-level interface:
"""

# ╔═╡ 310ac543-b448-4776-a66b-62d0de5695d0
p_res = vectorbundle_newton(M, E, f_prime, f_second_derivative, connection_map, p;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=(StopAfterIteration(15)|StopWhenChangeLess(M,1e-13)),
	retraction_method=ProjectionRetraction(),
	vector_transport_method=ProjectionTransport(),
	debug=[:Iteration, :Change, 1, "\n", :Stop]
)

# ╔═╡ e112fd45-dacc-49af-89ad-993d6a064860
f(M, p_res)

# ╔═╡ 4855e08c-4464-4a9b-bd68-de59a7c4884c
# We might have started at an eigenvalue already?

# ╔═╡ Cell order:
# ╠═9c6adf5c-d521-4227-bf44-64f317bccba5
# ╠═af19557a-9a39-4861-bc89-5fa932c26364
# ╠═2f11d1ca-4434-11ef-2ca7-f52cd378d278
# ╠═8d501a3e-2835-469b-a29c-2a314406ae5d
# ╠═9910fbe2-d9b9-48b6-96da-7db072ca6b71
# ╠═256f6350-3c69-4eec-bcea-6e8617ede1a2
# ╠═ae45b8f3-1984-4002-8fa8-e16c260d8a22
# ╠═743ef6ed-41d8-45a9-973b-b5b4867dd043
# ╠═6a07e0c2-ba0f-4d23-8e98-3f86138298be
# ╠═d1e0f7aa-f1cc-4ef6-90de-a8dbc0aec0b4
# ╠═5b2750cc-92a2-404a-bc76-c7d540021c5d
# ╠═aa7d9421-ae18-4564-bbc2-a8d437f9248c
# ╠═71c472a3-e500-433d-ba39-54c0e6f74174
# ╠═2913b52c-950f-4027-b552-b07f1cca1b76
# ╠═6d416d63-230b-4ee7-8e49-fb0a60a6778a
# ╠═1583e00e-85ec-4faa-9429-a3bcd804aa36
# ╠═6656986c-ebea-4850-9daa-8e453fda9bac
# ╠═5d019884-80a4-46b2-a2d0-bf0f2771db20
# ╟─9b49618b-2034-4ff2-9e8d-90d5127ae411
# ╠═310ac543-b448-4776-a66b-62d0de5695d0
# ╠═e112fd45-dacc-49af-89ad-993d6a064860
# ╠═4855e08c-4464-4a9b-bd68-de59a7c4884c
