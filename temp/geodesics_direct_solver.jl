### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 1adeba2e-53cf-11ef-15b8-9146d96dd44c
using Pkg; Pkg.activate();

# ╔═╡ b35fec03-4536-4612-ac48-20f620f3c315
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using OffsetArrays
	using Plots
	using Random
end;

# ╔═╡ b457e285-89de-410b-8b6a-f83b2c87e8b3
begin
	N=25
	h = 1/(N+2)*π/2
	Omega = range(; start=0.0, stop = π/2, length=N+2)[2:end-1]
	y0 = [0,0,1] # startpoint of geodesic
	yT = [1,0,0] # endpoint of geodesic
end;

# ╔═╡ 2ccd8a64-92a8-42e3-92f3-b7741f9d1a60
M3 = PowerManifold(Euclidean(4), NestedPowerRepresentation(), N);

# ╔═╡ 3c46069e-27db-462c-bcce-bfefca8974bb
function A(M, y, X)
	Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
	Ay = zero_vector(M, y)
	C = -1/h*Diagonal([ones(3)..., 0])
	for i in 1:N
#		E = Diagonal([1/h * (2+(-Oy[j-1][1:3]+2*Oy[j][1:3]-Oy[j+1][1:3])'*(-Matrix{Float64}(I, 3, 3)[1:3,j]*Oy[i][j] - Oy[i][1:3])) for j in 1:3])
		E = Diagonal([1/h * (2+(-Oy[i-1][1:3]+2*Oy[i][1:3]-Oy[i+1][1:3])'*([ - (i==j ? 2.0 : 1.0) * Oy[i][j] for i=1:3])) for j in 1:3])
		E = vcat(E, y[i][1:3]')
		E = hcat(E, [y[i][1:3]...,0])
		if i == 1
			Ay[M, i] = E*X[i] + C*X[i+1]
		elseif i == N
			Ay[M, i] = C*X[i-1] + E*X[i]
		else
			Ay[M, i] = C*X[i-1] + E*X[i] + C*X[i+1]
		end
	end
	return Ay
end

# ╔═╡ 67103dc4-0d0b-4f74-a0b8-020802e89e7b
function b(M, y)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			c = [ 1/h * (2*Oy[i][1:3] - Oy[i-1][1:3] - Oy[i+1][1:3])'* Matrix{Float64}(I, 3, 3)[1:3,j] + Oy[i][1:3]'*Matrix{Float64}(I, 3, 3)[1:3,j]*Oy[i][4] for j in 1:3]
			X[M, i] = [c...,0]
		end
		return X
end

# ╔═╡ 98ab46c5-9366-4e75-bb84-9d622dae8e1b
function y(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ 71bdb6e1-904a-48db-bc50-75d800804b3d
discretized_ylambda = [[y(Ωi)...,1] for Ωi in Omega]

# ╔═╡ 91c4d59f-bbb7-467d-89af-9327b219c6d8
function connection_map(E, q)
    return
end

# ╔═╡ 1814b9b3-4b1c-42a1-ace6-c065e7cf817e
function solve_linear_system(M, A, b, p)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	base = get_vectors(M, p, B)
	Ac = zeros(manifold_dimension(M),manifold_dimension(M));
    for (i,basis_vector) in enumerate(base)
	  Ac[:,i] = get_coordinates(M, p, A(M, p, basis_vector), B)
	end
	bc = get_coordinates(M, p, b(M, p), B)
	Xc = Ac \ (-bc)
	res_c = get_vector(M, p, Xc, B)
	return res_c
end

# ╔═╡ e7f593e8-9305-4e3a-956f-1e3754a4ac80
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, A, b, newtonstate.p)

# ╔═╡ 0a79f7c2-d016-4186-b52c-3f0ebc7a829f
p_res = vectorbundle_newton(M3, TangentBundle(M3), b, A, connection_map, discretized_ylambda;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=StopAfterIteration(10),
	#retraction_method=ProjectionRetraction(),
	debug=[:Iteration, :Change, 1, "\n", :Stop]
)

# ╔═╡ 3a22cc9a-770e-4494-9287-dd8fc2238ee7
begin
	Random.seed!(42)
	p = rand(M3)
	y_0 = [1/norm(p_res[i]+0.001*p[M3,i])*(p_res[i]+0.001*p[M3,i]) for i in 1:N]
end;

# ╔═╡ d3c50114-acc3-4b60-b1c1-3658032b4b7d
p_res2 = vectorbundle_newton(M3, TangentBundle(M3), b, A, connection_map, y_0;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=StopAfterIteration(10),
	#retraction_method=ProjectionRetraction(),
	debug=[:Iteration, :Change, 1, "\n", :Stop]
)

# ╔═╡ Cell order:
# ╠═1adeba2e-53cf-11ef-15b8-9146d96dd44c
# ╠═b35fec03-4536-4612-ac48-20f620f3c315
# ╠═b457e285-89de-410b-8b6a-f83b2c87e8b3
# ╠═2ccd8a64-92a8-42e3-92f3-b7741f9d1a60
# ╠═3c46069e-27db-462c-bcce-bfefca8974bb
# ╠═67103dc4-0d0b-4f74-a0b8-020802e89e7b
# ╠═98ab46c5-9366-4e75-bb84-9d622dae8e1b
# ╠═71bdb6e1-904a-48db-bc50-75d800804b3d
# ╠═91c4d59f-bbb7-467d-89af-9327b219c6d8
# ╠═e7f593e8-9305-4e3a-956f-1e3754a4ac80
# ╠═1814b9b3-4b1c-42a1-ace6-c065e7cf817e
# ╠═0a79f7c2-d016-4186-b52c-3f0ebc7a829f
# ╠═3a22cc9a-770e-4494-9287-dd8fc2238ee7
# ╠═d3c50114-acc3-4b60-b1c1-3658032b4b7d
