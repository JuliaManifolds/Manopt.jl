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
end;

# ╔═╡ b099f6dc-6434-44e1-a4c5-03b9f1bcab0d
begin
	h = 1/10
	stepsize = h*pi/2
	Omega = range(start=stepsize, step=stepsize, stop=pi/2)

	y0 = [0,0,1] # startpoint of geodesic
	yT = [1,0,0] # endpoint of geodesic
end;

# ╔═╡ a1a216ce-3cab-493c-99a1-9ea37a986e17
begin
	function discretized_y(y)
		List_of_y = []
		for i in 1:length(Omega)
			push!(List_of_y, y(Omega[i]))
		end
		return List_of_y
	end
end;

# ╔═╡ 44630e2e-439e-42be-ab12-8366ba2e7335
function discretized_energy(M, y)
	Oy = OffsetArray([y0, y...], 0:length(Omega))
	push!(Oy, yT)
	E = 0
	for i in 0:length(Omega)
		E = E + (Oy[i+1] - Oy[i])'* (Oy[i+1] - Oy[i])
	end
	E = 1/(2.0*stepsize) * E
end;

# ╔═╡ 026fb1e8-0ca2-4b02-bc70-1d625c907ea2
function y(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ 4bfb39d0-a561-4dec-a59a-9c4bffc15b8d
begin
	discretizedy = discretized_y(y)
	#println(discretizedy[length(discretizedy)])
end;

# ╔═╡ 31552a08-9596-49ad-be6e-ab2209be4c0e
print(abs(discretized_energy(Omega, discretized_y(y)) - pi/4))

# ╔═╡ 700b7f9f-93cb-4c01-b470-6a42fb129af4
M = Sphere(2)

# ╔═╡ 29417198-97f6-44d9-8e8b-df92a93ac1d5
begin
	function discretized_energy_derivative(M, y)
		Oy = OffsetArray([y0, y...], 0:length(Omega))
		push!(Oy, yT)
		derivative_energy_discretized = []
		for i in 1:length(Omega)-1
			B = get_basis(submanifold(M, i), Oy[i], DefaultOrthogonalBasis())
			b = get_vectors(submanifold(M, 1), Oy[i], B) # basis of tangent space at y_i
			Ob = OffsetArray([[0,0,0], b...], 0:length(b))
			push!(Ob, [0,0,0])
			for j in 0:length(b)+1
				val = 1/h * ((2*Oy[i] - Oy[i-1] - Oy[i+1])'*Ob[j])
				push!(derivative_energy_discretized, val)
			end
		end
		return derivative_energy_discretized
	end
end;

# ╔═╡ 01912d65-df98-4d29-b60e-7466c069972c
begin
	Mproduct = ProductManifold(M,M,M,M,M,M,M,M,M,M)
	#for i in 1:length(Omega)-1
	#	Mproduct = ProductManifold(Mproduct, M)
	#end
	#rand(Mproduct)
end;

# ╔═╡ 6111683a-ee67-4ce6-b7e8-b5fd0c4aab72
norm(discretized_energy_derivative(Mproduct, discretized_y(y)))

# ╔═╡ dc2f8de4-f6af-400a-9e16-2b079a7697db
#gradient_descent(Mproduct, discretized_energy, discretized_energy_derivative, discretized_y(y))

# ╔═╡ Cell order:
# ╠═6e502c97-0b1a-4403-8f81-6c15c832ce97
# ╠═441ed744-8225-417b-9ee7-258b5dc11a78
# ╠═b099f6dc-6434-44e1-a4c5-03b9f1bcab0d
# ╠═a1a216ce-3cab-493c-99a1-9ea37a986e17
# ╠═4bfb39d0-a561-4dec-a59a-9c4bffc15b8d
# ╠═44630e2e-439e-42be-ab12-8366ba2e7335
# ╠═026fb1e8-0ca2-4b02-bc70-1d625c907ea2
# ╠═31552a08-9596-49ad-be6e-ab2209be4c0e
# ╠═700b7f9f-93cb-4c01-b470-6a42fb129af4
# ╠═29417198-97f6-44d9-8e8b-df92a93ac1d5
# ╠═01912d65-df98-4d29-b60e-7466c069972c
# ╠═6111683a-ee67-4ce6-b7e8-b5fd0c4aab72
# ╠═dc2f8de4-f6af-400a-9e16-2b079a7697db
