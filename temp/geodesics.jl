### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 6e502c97-0b1a-4403-8f81-6c15c832ce97
using Pkg; Pkg.activate();

# ╔═╡ 441ed744-8225-417b-9ee7-258b5dc11a78
begin
	using LinearAlgebra
	using Manopt
	using Manifolds
	using OffsetArrays
	using Plots
	using Random
end;

# ╔═╡ b099f6dc-6434-44e1-a4c5-03b9f1bcab0d
begin
	N=25
	h = 1/(N+2)*π/2
	Omega = range(; start=0.0, stop = π/2, length=N+2)[2:end-1]
	y0 = [0,0,1] # startpoint of geodesic
	yT = [1,0,0] # endpoint of geodesic
end;

# ╔═╡ a1a216ce-3cab-493c-99a1-9ea37a986e17
begin
	function discretized_y(y)
		return [y(Ωi) for Ωi in Omega]
	end
end;

# ╔═╡ 44630e2e-439e-42be-ab12-8366ba2e7335
function discretized_energy(M, y)
	Oy = OffsetArray([y0, y...,yT], 0:(length(Omega)+1))

	return 1/(2.0*h) * sum(
		(Oy[i+1] - Oy[i])'* (Oy[i+1] - Oy[i]) for i=0:length(Omega)
	)
	#E = 0
	#for i in 0:length(Omega)
	#	E = E + (Oy[i+1] - Oy[i])'* (Oy[i+1] - Oy[i])
	#end
	#E = 1/(2.0*stepsize) * E
end;

# ╔═╡ ca8817dd-117c-4f98-b5c3-df3167adacdd
length(Omega)

# ╔═╡ 42ab08f5-c7e6-4a54-8df1-81df0e75c566
collect(Omega)/π

# ╔═╡ 026fb1e8-0ca2-4b02-bc70-1d625c907ea2
function y(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ 4bfb39d0-a561-4dec-a59a-9c4bffc15b8d
begin
	discretizedy = discretized_y(y)
	#println(discretizedy[length(discretizedy)])
end;

# ╔═╡ f42ccb22-16aa-4857-b3e2-5c480d3615da
[y(0.0), y(π/2)]

# ╔═╡ dce98ded-fbd7-49c3-a3bd-fc9bae4bdacb
collect(Omega)

# ╔═╡ 31552a08-9596-49ad-be6e-ab2209be4c0e
print(abs(discretized_energy(Omega, discretized_y(y)) - pi/4))

# ╔═╡ 700b7f9f-93cb-4c01-b470-6a42fb129af4
M = Sphere(2)

# ╔═╡ 29417198-97f6-44d9-8e8b-df92a93ac1d5
begin
	function discretized_energy_derivative(M, y)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		X = zero_vector(M,y)
		#derivative_energy_discretized = Vector[]
		for i in 1:length(Omega) # RB: inner points?
			# On product
			# B = get_basis(submanifold(M, i), Oy[i], DefaultOrthogonalBasis())
			# b = get_vectors(submanifold(M, 1), Oy[i], B) # basis of tangent space at y_i
			B = get_basis(M.manifold, Oy[i], DefaultOrthogonalBasis())
			b = get_vectors(M.manifold, Oy[i], B) # basis of tangent space
			# Ob = OffsetArray([[0,0,0], b..., [0,0,0]], 0:length(b)+1)
			#for j in 0:length(b)+1
			#	val = 1/h * ((2*Oy[i] - Oy[i-1] - Oy[i+1])'*Ob[j])
			#   push!(derivative_energy_discretized, val)
			#end
			# coeffs:
			c = [ 1/h * (2*Oy[i] - Oy[i-1] - Oy[i+1])'*bj for bj in b]
			X[M, i] = get_vector(M.manifold, Oy[i], c, B)
		end
		return X
		#return derivative_energy_discretized
	end
end;

# ╔═╡ 56128130-b9a4-4fed-8da5-16d869fc3621
length(Omega)

# ╔═╡ 9325a36b-0ae0-45b3-ba6f-9465e3097b46
M2 = PowerManifold(M, NestedPowerRepresentation(), N)

# ╔═╡ 6111683a-ee67-4ce6-b7e8-b5fd0c4aab72
norm(discretized_energy_derivative(M2, discretized_y(y)))

# ╔═╡ a6bb5c25-22fc-4ffc-a32c-dfd30a3519bc
discretized_energy_derivative(M2, discretized_y(y))

# ╔═╡ 5493684b-bc44-48bd-b9e0-6783a406d054
M2.manifold

# ╔═╡ 24a4c417-2f74-4418-86cf-b6c980a70149
begin
	Random.seed!(42)
	p = rand(M2)
	y_0 = [1/norm(discretized_y(y)[i]+0.0001*p[M2,i])*(discretized_y(y)[i]+0.0001*p[M2,i]) for i in 1:N]
end;
#y_0 = rand(M2)

# ╔═╡ 792e458c-a7e0-4c22-af23-1b0cf532743a
check_gradient(M2, discretized_energy, discretized_energy_derivative, y_0; plot=true, rtol=1e-12)

# ╔═╡ dc2f8de4-f6af-400a-9e16-2b079a7697db
y_opt = gradient_descent(M2, discretized_energy, discretized_energy_derivative, y_0; debug=[:Iteration, :Cost, " | ",:GradientNorm, 25,"\n",:Stop])

# ╔═╡ e8bf7131-31f0-4bf5-b7ad-8826ebd8fb37
is_point(M2,y_opt)

# ╔═╡ ab589ad2-76aa-48d0-8b5d-df685b6efbb6
# The problem with the current y is, that is is on a NestedPowerManifold not on a Product Manifold. But we can discuss that tomorrow, when discussing Data structures anyways.

# ╔═╡ 998b1fd7-6574-43e6-9d41-3d2aa7e51927
# Für product bräuchtest du ArrayPartition(y1, y2,...,yN)

# ╔═╡ d12d9db3-bbdd-4779-a5d0-fc85448d7290
# auf PowerManifold(M, NestedPowerRepresentation(), 10)

# ╔═╡ 8d48f6e9-65a9-40e8-8fba-598a1bf8cb45
# M^10 (matrix representation)

# ╔═╡ 53cad866-4efd-4558-ab84-938a0ab8f726
#M3 = (ProductManifold(Euclidean(3), Euclidean(1)))^N
M3 = PowerManifold(Euclidean(4), NestedPowerRepresentation(), N);

# ╔═╡ bf2d374a-ae6a-44d7-a9ff-0316785bf545
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

# ╔═╡ 13981f8e-0e10-4432-954c-6d5d79cc26e6
function b(M, y)
		# Include boundary points
		Oy = OffsetArray([y0, y..., yT], 0:(length(Omega)+1))
		X = zero_vector(M,y)
		for i in 1:length(Omega)
			c = [ -1.0*1/h * (2*Oy[i][1:3] - Oy[i-1][1:3] - Oy[i+1][1:3])'* Matrix{Float64}(I, 3, 3)[1:3,j] for j in 1:3]
			X[M, i] = [c...,0]
		end
		return X
end

# ╔═╡ 56fd52a0-e09c-4c78-b4d7-515b7519219e
begin
	discretized_ylambda = [[y(Ωi)...,0] for Ωi in Omega]
end

# ╔═╡ 439574d4-b78b-4610-81f9-143c6d049573
B3 = get_basis(M3, discretized_ylambda, DefaultOrthonormalBasis());

# ╔═╡ 04e55f95-94c3-4f0f-a51c-db74b6ae429a
b3 = get_vectors(M3, discretized_ylambda, B3);

# ╔═╡ add85a20-58b0-4b74-a66c-98cc3b74cfa6
begin
    Ac = zeros(manifold_dimension(M3),manifold_dimension(M3));
    for (i,b) in enumerate(b3)
	  Ac[:,i] = get_coordinates(M3, discretized_ylambda, A(M3, discretized_ylambda, b), B3)
   end
end

# ╔═╡ 1ed8903b-98dd-4e88-97e4-66ffc357506c
det(Ac)

# ╔═╡ 000bcde0-5a29-4ba5-9c7c-e2fc40ea199e
Ac

# ╔═╡ 74848c8a-fe1d-49d2-873f-c6f7fd081d39
det(Ac)

# ╔═╡ df272fef-a844-4853-a9a8-87d5db0b6ef2
bc = get_coordinates(M3, discretized_ylambda, b(M3, discretized_ylambda), B3)

# ╔═╡ 02faf6d3-8a19-4cb7-9824-f035d02e2b4d
Xc = Ac \ (-bc)

# ╔═╡ c0aad8b6-7f95-431e-ae9d-0215b9a80bda
res_c = get_vector(M3, discretized_ylambda, Xc, B3)

# ╔═╡ c2251ffd-2248-4845-9b64-4f58a7d18809
TyM = TangentSpace(M3, discretized_ylambda);

# ╔═╡ ccc7f9c1-1677-4205-a1eb-ca03521a64e2
manifold_dimension(M3)

# ╔═╡ 374c7849-ddc4-4564-9f98-d2d0607e3774
@time res = conjugate_residual(TyM, A, b; stopping_criterion=StopAfterIteration(300), debug=[:Iteration, :Cost, " | ",:GradientNorm, 100,"\n",:Stop]);

# ╔═╡ 57e24ad7-a076-4c8e-80ac-5acad19a2203
res;

# ╔═╡ 64b5e297-5186-4488-b8d8-03aab78d89d1
res_c;

# ╔═╡ ba242dea-2c53-4d89-9bc2-3b88f75722da
norm(M3, discretized_ylambda, A(M3,discretized_ylambda,res) + b(M3,discretized_ylambda))

# ╔═╡ 87667dcb-1f13-436e-80b3-6d91d9922d0e
norm(M3, discretized_ylambda, A(M3,discretized_ylambda,res_c) + b(M3,discretized_ylambda))

# ╔═╡ Cell order:
# ╠═6e502c97-0b1a-4403-8f81-6c15c832ce97
# ╠═441ed744-8225-417b-9ee7-258b5dc11a78
# ╠═b099f6dc-6434-44e1-a4c5-03b9f1bcab0d
# ╠═a1a216ce-3cab-493c-99a1-9ea37a986e17
# ╠═4bfb39d0-a561-4dec-a59a-9c4bffc15b8d
# ╠═44630e2e-439e-42be-ab12-8366ba2e7335
# ╠═ca8817dd-117c-4f98-b5c3-df3167adacdd
# ╠═42ab08f5-c7e6-4a54-8df1-81df0e75c566
# ╠═026fb1e8-0ca2-4b02-bc70-1d625c907ea2
# ╠═f42ccb22-16aa-4857-b3e2-5c480d3615da
# ╠═dce98ded-fbd7-49c3-a3bd-fc9bae4bdacb
# ╠═31552a08-9596-49ad-be6e-ab2209be4c0e
# ╠═700b7f9f-93cb-4c01-b470-6a42fb129af4
# ╠═29417198-97f6-44d9-8e8b-df92a93ac1d5
# ╠═56128130-b9a4-4fed-8da5-16d869fc3621
# ╠═6111683a-ee67-4ce6-b7e8-b5fd0c4aab72
# ╠═a6bb5c25-22fc-4ffc-a32c-dfd30a3519bc
# ╠═9325a36b-0ae0-45b3-ba6f-9465e3097b46
# ╠═5493684b-bc44-48bd-b9e0-6783a406d054
# ╠═24a4c417-2f74-4418-86cf-b6c980a70149
# ╠═792e458c-a7e0-4c22-af23-1b0cf532743a
# ╠═dc2f8de4-f6af-400a-9e16-2b079a7697db
# ╠═e8bf7131-31f0-4bf5-b7ad-8826ebd8fb37
# ╠═ab589ad2-76aa-48d0-8b5d-df685b6efbb6
# ╠═998b1fd7-6574-43e6-9d41-3d2aa7e51927
# ╠═d12d9db3-bbdd-4779-a5d0-fc85448d7290
# ╠═8d48f6e9-65a9-40e8-8fba-598a1bf8cb45
# ╠═53cad866-4efd-4558-ab84-938a0ab8f726
# ╠═bf2d374a-ae6a-44d7-a9ff-0316785bf545
# ╠═439574d4-b78b-4610-81f9-143c6d049573
# ╠═04e55f95-94c3-4f0f-a51c-db74b6ae429a
# ╠═add85a20-58b0-4b74-a66c-98cc3b74cfa6
# ╠═1ed8903b-98dd-4e88-97e4-66ffc357506c
# ╠═000bcde0-5a29-4ba5-9c7c-e2fc40ea199e
# ╠═13981f8e-0e10-4432-954c-6d5d79cc26e6
# ╠═df272fef-a844-4853-a9a8-87d5db0b6ef2
# ╠═02faf6d3-8a19-4cb7-9824-f035d02e2b4d
# ╠═74848c8a-fe1d-49d2-873f-c6f7fd081d39
# ╠═c0aad8b6-7f95-431e-ae9d-0215b9a80bda
# ╠═56fd52a0-e09c-4c78-b4d7-515b7519219e
# ╠═c2251ffd-2248-4845-9b64-4f58a7d18809
# ╠═ccc7f9c1-1677-4205-a1eb-ca03521a64e2
# ╠═374c7849-ddc4-4564-9f98-d2d0607e3774
# ╠═57e24ad7-a076-4c8e-80ac-5acad19a2203
# ╠═64b5e297-5186-4488-b8d8-03aab78d89d1
# ╠═ba242dea-2c53-4d89-9bc2-3b88f75722da
# ╠═87667dcb-1f13-436e-80b3-6d91d9922d0e
