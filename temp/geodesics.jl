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
	h = 1/100
	stepsize = h*pi/2
	Omega = range(start=0, step=stepsize, stop=pi/2)
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
	discretizedy = discretized_y(y)
	Oy = OffsetArray([[0,0,1], discretizedy...], 0:length(Omega))
	push!(Oy, [1,0,0])
	E = 0
	for i in 1:length(Omega)
		E = E + (Oy[i+1] - Oy[i])'* (Oy[i+1] - Oy[i])
	end
	E = 1/(2.0*stepsize) * E
end;

# ╔═╡ 026fb1e8-0ca2-4b02-bc70-1d625c907ea2
function y(t)
	return [sin(t), 0, cos(t)]
end;

# ╔═╡ 31552a08-9596-49ad-be6e-ab2209be4c0e
print(abs(discretized_energy(Omega, y) - pi/4))

# ╔═╡ 700b7f9f-93cb-4c01-b470-6a42fb129af4
M = Sphere(2)

# ╔═╡ 29417198-97f6-44d9-8e8b-df92a93ac1d5
begin
	function discretized_energy_derivative(M, y)
		discretizedy = discretized_y(y)
		Oy = OffsetArray([[0,0,1], discretizedy...], 0:length(Omega))
		push!(Oy, [1,0,0])
		derivative_energy_discretized = []
		for i in 1:length(Omega)
			B = get_basis(M, Oy[i], DefaultOrthogonalBasis())
			b = get_vectors(M, Oy[i], B) # basis of tangent space at y_i
			for j in 1:length(b)
				val = 1/h * ((2*Oy[i] - Oy[i-1] - Oy[i+1])'*b[j])
				push!(derivative_energy_discretized, val)
			end
		end
		return derivative_energy_discretized
	end
end;

# ╔═╡ 91b514d2-fa28-4cf5-a4b6-550dbbe6d7dc
norm(discretized_energy_derivative(M, y)) # sollte 0 sein, glaub das liegt daran, dass der erste punkt (also y(0)) zweimal vorkommt 

# ╔═╡ dc2f8de4-f6af-400a-9e16-2b079a7697db
#gradient_descent(M, discretized_energy, discretized_energy_derivative)

# ╔═╡ Cell order:
# ╠═6e502c97-0b1a-4403-8f81-6c15c832ce97
# ╠═441ed744-8225-417b-9ee7-258b5dc11a78
# ╠═b099f6dc-6434-44e1-a4c5-03b9f1bcab0d
# ╠═a1a216ce-3cab-493c-99a1-9ea37a986e17
# ╠═44630e2e-439e-42be-ab12-8366ba2e7335
# ╠═026fb1e8-0ca2-4b02-bc70-1d625c907ea2
# ╠═31552a08-9596-49ad-be6e-ab2209be4c0e
# ╠═700b7f9f-93cb-4c01-b470-6a42fb129af4
# ╠═29417198-97f6-44d9-8e8b-df92a93ac1d5
# ╠═91b514d2-fa28-4cf5-a4b6-550dbbe6d7dc
# ╠═dc2f8de4-f6af-400a-9e16-2b079a7697db
