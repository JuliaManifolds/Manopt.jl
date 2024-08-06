### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ b7726008-53f6-11ef-216f-c1984c3e1e7b
using Pkg; Pkg.activate();

# ╔═╡ 64eb4ec9-2b11-42ed-987a-f72066adefe1
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using OffsetArrays
	using Plots
	using Random
end;

# ╔═╡ caf81526-dfb2-438e-99d2-03c6b60405af
begin
	N=300
	h = 1/(N+2)*π/2
	Omega = range(; start=0.0, stop = π/2, length=N+2)[2:end-1]
	y0 = [0,0,1] # startpoint of geodesic
	yT = [1,0,0] # endpoint of geodesic
end;

# ╔═╡ 42c54278-2ae7-4f75-b391-9011d2154dad
M3 = PowerManifold(Euclidean(4), NestedPowerRepresentation(), N);

# ╔═╡ c4664660-037f-4711-8d08-24f4c404979c
function y(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ 15611bda-f70a-4df7-ba35-2cda58870f75
discretized_ylambda = [[y(Ωi)...,1] for Ωi in Omega]

# ╔═╡ 7e6db5df-7f76-4235-94e9-e7561b0c3e06
begin
	Random.seed!(4)
	f = 10*rand(TangentBundle(M3))
	#f = [1/norm(0.1*p[TangentBundle(M3),i])*(+0.1*p[TangentBundle(M3),i]) for i in 1:N]
end;

# ╔═╡ 572cddcc-bc6a-4c92-b649-b109c7233f00
function A(M, y, X)
	Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
	Ay = zero_vector(M, y)
	C = -1/h*Diagonal([ones(3)..., 0])
	for i in 1:N
#		E = Diagonal([1/h * (2+(-Oy[j-1][1:3]+2*Oy[j][1:3]-Oy[j+1][1:3])'*(-Matrix{Float64}(I, 3, 3)[1:3,j]*Oy[i][j] - Oy[i][1:3])) for j in 1:3])
		E = Diagonal([1/h * (2+(-Oy[i-1][1:3]+2*Oy[i][1:3]-Oy[i+1][1:3])'*([ - (i==j ? 2.0 : 1.0) * Oy[i][j] for i=1:3])) - h * f[i][1:3]'*([ - (i==j ? 2.0 : 1.0) * Oy[i][j] for i=1:3]) for j in 1:3])
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

# ╔═╡ 2d418421-2558-418d-a260-225836dad049
function b(M, y)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			c = [ 1/h * (2*Oy[i][1:3] - Oy[i-1][1:3] - Oy[i+1][1:3])'* Matrix{Float64}(I, 3, 3)[1:3,j] + Oy[i][1:3]'*Matrix{Float64}(I, 3, 3)[1:3,j]*Oy[i][4] + h * f[i][j] for j in 1:3]
			X[M, i] = [c...,0]
		end
		return X
end

# ╔═╡ 0828cdb7-1a18-4059-81c6-dd7509d7de4e
function connection_map(E, q)
    return q
end

# ╔═╡ a99c6ea8-203d-4440-ac1e-2ce9c342f3e4
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

# ╔═╡ c6c8216f-7f71-4f04-a766-edadd38000fa
solve(problem, newtonstate, k) = solve_linear_system(problem.manifold, A, b, newtonstate.p)

# ╔═╡ 5596b132-af0f-4896-8617-bbe1d7c9bf41
p_res = vectorbundle_newton(M3, TangentBundle(M3), b, A, connection_map, discretized_ylambda;
	sub_problem=solve,
	sub_state=AllocatingEvaluation(),
	stopping_criterion=StopAfterIteration(10),
	#retraction_method=ProjectionRetraction(),
	debug=[:Iteration, (:Change, "Change: %1.8e"), 1, "\n", :Stop]
)

# ╔═╡ 5d0a6ccb-a0a0-44b9-a80a-cb244db00af7
is_point(M3, p_res)

# ╔═╡ b083f8e4-0dff-43d5-8bce-0f4dd85d9569
discretized_ylambda

# ╔═╡ ea267cbc-0ad4-4fb8-b378-eff8be2d7c3f
begin
	pythonplot()
	S = Sphere(2)
	pts = [ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0] ]
	scene = plot(S, p_res; wireframe_color=colorant"#CCCCCC", markersize=1)
	#plot(S, discretized_ylambda; wireframe_color=colorant"#CCCCCC", markersize=1)
	#plot!(scene, S, pts; wireframe=false, geodesic_interpolation=100, linewidth=2)
end

# ╔═╡ Cell order:
# ╠═b7726008-53f6-11ef-216f-c1984c3e1e7b
# ╠═64eb4ec9-2b11-42ed-987a-f72066adefe1
# ╠═caf81526-dfb2-438e-99d2-03c6b60405af
# ╠═42c54278-2ae7-4f75-b391-9011d2154dad
# ╠═c4664660-037f-4711-8d08-24f4c404979c
# ╠═15611bda-f70a-4df7-ba35-2cda58870f75
# ╠═7e6db5df-7f76-4235-94e9-e7561b0c3e06
# ╠═572cddcc-bc6a-4c92-b649-b109c7233f00
# ╠═2d418421-2558-418d-a260-225836dad049
# ╠═0828cdb7-1a18-4059-81c6-dd7509d7de4e
# ╠═c6c8216f-7f71-4f04-a766-edadd38000fa
# ╠═a99c6ea8-203d-4440-ac1e-2ce9c342f3e4
# ╠═5596b132-af0f-4896-8617-bbe1d7c9bf41
# ╠═5d0a6ccb-a0a0-44b9-a80a-cb244db00af7
# ╠═b083f8e4-0dff-43d5-8bce-0f4dd85d9569
# ╠═ea267cbc-0ad4-4fb8-b378-eff8be2d7c3f
