### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 3c4e4c22-53cc-11ef-0af6-078d9d74dc7e
using Pkg; Pkg.activate();

# ╔═╡ 48e07d18-6359-40a1-8d58-d5fae2a3def6
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using OffsetArrays
	using Plots
	using Random
end;

# ╔═╡ 0758d0e4-509b-4e2b-9e84-ade9d367a041
begin
	N=4
	h = 1/(N+2)*π/2
	Omega = range(; start=0.0, stop = π/2, length=N+2)[2:end-1]
	y0 = [0,0,1] # startpoint of geodesic
	yT = [1,0,0] # endpoint of geodesic
end;

# ╔═╡ bd75a594-0a1b-4e4d-b182-ce2778f60a02
function discretized_energy_derivative(M, y)
	# Include boundary points
	Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))	
	X = zero_vector(M,y)
	for i in 1:length(Omega)
		c = [ -1.0 * (2*Oy[i][1:3] - Oy[i-1][1:3] - Oy[i+1][1:3])'* Matrix{Float64}(I, 3, 3)[1:3,j] for j in 1:3]
		X[M, i] = [c...,0]
	end	
	return X
end;

# ╔═╡ 1ebca723-8442-42c8-b3e1-6b69d267ab46
function y(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ b6b568ea-0760-4829-a960-6eb8e76d2f71
M = Sphere(2)

# ╔═╡ 7b1c83d1-1358-4b90-bf5d-ca30b187b7cb
function A(M, y, X)
	Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
	Ay = zero_vector(M, y)
	C = -1*Diagonal([ones(3)..., 0])
	for i in 1:N
#		E = Diagonal([1/h * (2+(-Oy[j-1][1:3]+2*Oy[j][1:3]-Oy[j+1][1:3])'*(-Matrix{Float64}(I, 3, 3)[1:3,j]*Oy[i][j] - Oy[i][1:3])) for j in 1:3])
		E = Diagonal([1 * (2+(-Oy[i-1][1:3]+2*Oy[i][1:3]-Oy[i+1][1:3])'*([ - (i==j ? 2.0 : 1.0) * Oy[i][j] for i=1:3])) for j in 1:3])
		E = vcat(E, h*y[i][1:3]')
		E = hcat(E, [h*y[i][1:3]...,0])
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

# ╔═╡ 05598f1c-96d5-41a5-9634-44992b1dcb4e
function discretized_energy_second_derivative(M, E, p)
	B = get_basis(M, p, DefaultOrthonormalBasis())
	base = get_vectors(M, p, B)
	Ac = zeros(manifold_dimension(M),manifold_dimension(M));
    #for (i,basis_vector) in enumerate(base)
	#  Ac[:,i] = get_coordinates(M, p, A(M, p, basis_vector), B)
	#end
	for i in 1:manifold_dimension(M)
		for j in 1:manifold_dimension(M)
			Ac[i,j] = A(M, p, base[i])'*base[j]
		end
	end
	return Ac
end

# ╔═╡ 6b609b7f-1c9d-48a6-b33d-d40447def264
function b(M, y)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			c = [(2*Oy[i][1:3] - Oy[i-1][1:3] - Oy[i+1][1:3])'* Matrix{Float64}(I, 3, 3)[1:3,j] + h*Oy[i][1:3]'*Matrix{Float64}(I, 3, 3)[1:3,j]*Oy[i][4] for j in 1:3]
			X[M, i] = [-c...,0]
		end
		return X
end

# ╔═╡ c1408f56-fafb-4580-bbad-5e7c6c8a613c
discretized_ylambda = [[y(Ωi)...,0] for Ωi in Omega];

# ╔═╡ c5a084e7-d654-46fe-ba44-0563e7446195
M3 = PowerManifold(Euclidean(4), NestedPowerRepresentation(), N);

# ╔═╡ c51df677-1c05-4637-8670-894b2611fda2
TyM = TangentSpace(M3, discretized_ylambda);

# ╔═╡ 5a229c03-4dc8-4346-bbec-f6d5812ade10
function connection_map(E, q)
    return
end

# ╔═╡ 844c027c-5f42-4d27-a1fb-2b01ec4404ac
p_res = vectorbundle_newton(M3, TangentBundle(M3), discretized_energy_derivative, discretized_energy_second_derivative, connection_map, discretized_ylambda;
    sub_problem=DefaultManoptProblem(TyM, SymmetricLinearSystemObjective(A,b)),
	sub_state=ConjugateResidualState(TyM, SymmetricLinearSystemObjective(A,b)),
	stopping_criterion=StopAfterIteration(15),
	#retraction_method=ProjectionRetraction(),
	debug=[:Iteration, :Change, 1, "\n", :Stop]
)

# ╔═╡ Cell order:
# ╠═3c4e4c22-53cc-11ef-0af6-078d9d74dc7e
# ╠═48e07d18-6359-40a1-8d58-d5fae2a3def6
# ╠═0758d0e4-509b-4e2b-9e84-ade9d367a041
# ╠═bd75a594-0a1b-4e4d-b182-ce2778f60a02
# ╠═1ebca723-8442-42c8-b3e1-6b69d267ab46
# ╠═b6b568ea-0760-4829-a960-6eb8e76d2f71
# ╠═7b1c83d1-1358-4b90-bf5d-ca30b187b7cb
# ╠═05598f1c-96d5-41a5-9634-44992b1dcb4e
# ╠═6b609b7f-1c9d-48a6-b33d-d40447def264
# ╠═c1408f56-fafb-4580-bbad-5e7c6c8a613c
# ╠═c5a084e7-d654-46fe-ba44-0563e7446195
# ╠═c51df677-1c05-4637-8670-894b2611fda2
# ╠═5a229c03-4dc8-4346-bbec-f6d5812ade10
# ╠═844c027c-5f42-4d27-a1fb-2b01ec4404ac
